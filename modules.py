import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import fftshift, ifftshift

class CustomLSTM(nn.Module):
    """
    텍스트 임베딩을 recurrent하게 처리하는 LSTM
    """
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()
        self.noise2h = nn.Linear(512, hidden_sz)  # CLIP embedding (512D) -> hidden
        self.noise2c = nn.Linear(512, hidden_sz)  # CLIP embedding -> cell state
        #print("CustomLSTM input_sz =",input_sz)
        #print("CustomLSTM hidden_sz =",hidden_sz)
        
    def init_weights(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def init_hidden(self, cond):
        """CLIP 텍스트 임베딩으로 hidden/cell state 초기화"""
        h_t = self.noise2h(cond)
        c_t = self.noise2c(cond)
        self.c_t = c_t
        self.h_t = h_t
    
    def forward(self, x):
        """LSTM forward pass"""
        c_t = self.c_t
        h_t = self.h_t
        HS = self.hidden_size
        x_t = x
        
        gates = x_t @ self.W + h_t @ self.U + self.bias
        i_t, f_t, g_t, o_t = (
            torch.sigmoid(gates[:, :HS]),
            torch.sigmoid(gates[:, HS:HS*2]),
            torch.tanh(gates[:, HS*2:HS*3]),
            torch.sigmoid(gates[:, HS*3:]),
        )
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        self.c_t = c_t
        self.h_t = h_t
        return h_t

class FreeSpaceProp(nn.Module):
    def __init__(self, 
                 wlength_vc, 
                 ridx_air,
                 total_x_num, total_y_num,
                 dx, dy,
                 prop_z):
        super(FreeSpaceProp, self).__init__()
        self.prop_z = prop_z
        self.total_x_num = total_x_num
        self.total_y_num = total_y_num
        self.dx = dx
        self.dy = dy

        wlengtheff = wlength_vc / ridx_air # effect wavelength
        
        y, x = (dy * float(total_y_num), dx * float(total_x_num))

        fy = torch.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), total_y_num)
        fx = torch.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), total_x_num)

        fx, fy = torch.meshgrid(fx, fy, indexing='ij')

        self.f0 = 1 / wlengtheff # to ensure coherent? It is the diffraction limitation.
        fy_max = 1 / np.sqrt((2 * prop_z * (1 / y))**2 + 1) / wlengtheff
        fx_max = 1 / np.sqrt((2 * prop_z * (1 / x))**2 + 1) / wlengtheff
        Q = torch.tensor(((np.abs(np.array(fx)) < fx_max) & (np.abs(np.array(fy)) < fy_max)).astype(np.uint8), 
                         dtype=torch.float32)

        # Angle Spectrum
        prop_window = Q * (fx**2 + fy**2) * (wlengtheff**2)
        phase_change = 2 * np.pi * self.f0 * prop_z * torch.sqrt(torch.clamp(1 - prop_window, 
                                                                        min=0, max=1))
        phase_change_cplx = torch.complex(torch.cos(phase_change),
                                          torch.sin(phase_change)) * Q # as exp(1i*...)
        
        shifted_phase_change_cplx = torch.fft.ifftshift(phase_change_cplx)
        shifted_phase_change_cplx = shifted_phase_change_cplx[None, None, ...]

        self.register_buffer('shifted_phase_change_cplx',
                             shifted_phase_change_cplx)
        
        
    def forward(self, x):
        
        ASpectrum = torch.fft.fft2(x)
        ASpectrum_z = torch.mul(self.shifted_phase_change_cplx, ASpectrum)
        output = torch.fft.ifft2(ASpectrum_z)
        return output
    
    def update_proplocation(self, x_shift, y_shift, z_shift):
        
        self.prop_z = self.prop_z + z_shift.item()
        Wz =  self.f0 * self.prop_z * torch.sqrt(torch.clamp(1 - self.prop_window, min=0, max=1))
        
        phase_change = 2 * np.pi * (Wz + self.fx * x_shift.item() + self.fy * y_shift.item())
        phase_change_cplx = torch.complex(torch.cos(phase_change),
                                          torch.sin(phase_change)) * self.Q
        
        # band pass for off axis numerical propagation
        phase_change_cplx = self._bandpass(phase_change_cplx, self.fx, self.fy, 
                                           self.total_x_num*self.dx, self.total_y_num*self.dy,
                                           x_shift.item(), y_shift.item(), self.prop_z, self.wlengtheff)

        shifted_phase_change_cplx = torch.fft.ifftshift(phase_change_cplx)
        shifted_phase_change_cplx = shifted_phase_change_cplx[None, None, ...].to(z_shift.device)

        self.register_buffer('shifted_phase_change_cplx',
                             shifted_phase_change_cplx)
        
    def _bandpass(self, H, fX, fY, Sx, Sy, x0, y0, z0, wv):
        """
        Table 1 of "Shifted angular spectrum method for off-axis numerical propagation" (2010).
        :param Sx:
        :param Sy:
        :param x0:
        :param y0:
        :return:
        """

        du = 1 / (2 * Sx)
        u_limit_p = ((x0 + 1 / (2 * du)) ** (-2) * z0 ** 2 + 1) ** (-1 / 2) / wv
        u_limit_n = ((x0 - 1 / (2 * du)) ** (-2) * z0 ** 2 + 1) ** (-1 / 2) / wv

        if Sx < x0:
            u0 = (u_limit_p + u_limit_n) / 2
            u_width = u_limit_p - u_limit_n
        elif x0 <= -Sx:
            u0 = -(u_limit_p + u_limit_n) / 2
            u_width = u_limit_n - u_limit_p
        else:
            u0 = (u_limit_p - u_limit_n) / 2
            u_width = u_limit_p + u_limit_n

        dv = 1 / (2 * Sy)
        v_limit_p = ((y0 + 1 / (2 * dv)) ** (-2) * z0 ** 2 + 1) ** (-1 / 2) / wv
        v_limit_n = ((y0 - 1 / (2 * dv)) ** (-2) * z0 ** 2 + 1) ** (-1 / 2) / wv

        if Sy < y0:
            # print('Sy < y0')
            v0 = (v_limit_p + v_limit_n) / 2
            v_width = v_limit_p - v_limit_n
        elif y0 <= -Sy:
            # print('y0 <= -Sy')
            v0 = -(v_limit_p + v_limit_n) / 2
            v_width = v_limit_n - v_limit_p
        else:
            # print('else')
            v0 = (v_limit_p - v_limit_n) / 2
            v_width = v_limit_p + v_limit_n

        fx_max = u_width / 2
        fy_max = v_width / 2

        H_filter = (torch.abs(fX - u0) <= fx_max) * (torch.abs(fY - v0) < fy_max)

        H_final = H * H_filter

        return H_final

class FreeSpaceProp_Multich(nn.Module):
    def __init__(self, 
                 wlength_vc, 
                 ridx_air,
                 total_x_num, total_y_num,
                 dx, dy,
                 prop_z):
        super(FreeSpaceProp_Multich, self).__init__()
        self.prop_z = prop_z
        self.total_x_num = total_x_num
        self.total_y_num = total_y_num
        self.dx = dx
        self.dy = dy

        wlengtheff = torch.tensor(wlength_vc, dtype=torch.float32)[..., None, None].repeat(1, total_x_num, total_y_num) / ridx_air # effect wavelength
        
        y, x = (dy * float(total_y_num), dx * float(total_x_num))

        # NOTE: this frequency sampling is for even sampling
        fy = torch.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), total_y_num)
        fx = torch.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), total_x_num)

        fx, fy = torch.meshgrid(fx, fy, indexing='ij')
        fx = fx[None, ...].repeat(len(wlength_vc), 1, 1)
        fy = fy[None, ...].repeat(len(wlength_vc), 1, 1)

        self.f0 = 1 / wlengtheff # to ensure coherent? It is the diffraction limitation.
        fy_max = 1 / np.sqrt((2 * prop_z * (1 / y))**2 + 1) / wlengtheff
        fx_max = 1 / np.sqrt((2 * prop_z * (1 / x))**2 + 1) / wlengtheff
        Q = (fx < fx_max) & (fy < fy_max)

        # Angle Spectrum
        prop_window = Q * (fx**2 + fy**2) * (wlengtheff**2)
        phase_change = 2 * np.pi * self.f0 * prop_z * torch.sqrt(torch.clamp(1 - prop_window, 
                                                                        min=0, max=1))
        phase_change_cplx = torch.complex(torch.cos(phase_change),
                                          torch.sin(phase_change)) * Q # as exp(1i*...)
        
        shifted_phase_change_cplx = torch.fft.ifftshift(phase_change_cplx)
        shifted_phase_change_cplx = shifted_phase_change_cplx[None, ...]

        self.register_buffer('shifted_phase_change_cplx',
                             shifted_phase_change_cplx)
        
        
    def forward(self, x):
        
        # ASpectrum = torch.fft.fft2(x)
        # ASpectrum_z = torch.mul(self.shifted_phase_change_cplx, ASpectrum)
        # output = torch.fft.ifft2(ASpectrum_z)
        
        U1 = torch.fft.fftn(ifftshift(x), dim=(-2, -1), norm='ortho')
    
        U2 = self.shifted_phase_change_cplx * U1

        output = fftshift(torch.fft.ifftn(U2, dim=(-2, -1), norm='ortho'))
        return output
    
    def update_proplocation(self, x_shift, y_shift, z_shift):
        
        self.prop_z = self.prop_z + z_shift.item()
        Wz =  self.f0 * self.prop_z * torch.sqrt(torch.clamp(1 - self.prop_window, min=0, max=1))
        
        phase_change = 2 * np.pi * (Wz + self.fx * x_shift.item() + self.fy * y_shift.item())
        phase_change_cplx = torch.complex(torch.cos(phase_change),
                                          torch.sin(phase_change)) * self.Q
        
        # band pass for off axis numerical propagation
        phase_change_cplx = self._bandpass(phase_change_cplx, self.fx, self.fy, 
                                           self.total_x_num*self.dx, self.total_y_num*self.dy,
                                           x_shift.item(), y_shift.item(), self.prop_z, self.wlengtheff)

        shifted_phase_change_cplx = torch.fft.ifftshift(phase_change_cplx)
        shifted_phase_change_cplx = shifted_phase_change_cplx[None, ...].to(z_shift.device)

        self.register_buffer('shifted_phase_change_cplx',
                             shifted_phase_change_cplx)
        
    def _bandpass(self, H, fX, fY, Sx, Sy, x0, y0, z0, wv):
        """
        Table 1 of "Shifted angular spectrum method for off-axis numerical propagation" (2010).
        :param Sx:
        :param Sy:
        :param x0:
        :param y0:
        :return:
        """

        du = 1 / (2 * Sx)
        u_limit_p = ((x0 + 1 / (2 * du)) ** (-2) * z0 ** 2 + 1) ** (-1 / 2) / wv
        u_limit_n = ((x0 - 1 / (2 * du)) ** (-2) * z0 ** 2 + 1) ** (-1 / 2) / wv

        if Sx < x0:
            u0 = (u_limit_p + u_limit_n) / 2
            u_width = u_limit_p - u_limit_n
        elif x0 <= -Sx:
            u0 = -(u_limit_p + u_limit_n) / 2
            u_width = u_limit_n - u_limit_p
        else:
            u0 = (u_limit_p - u_limit_n) / 2
            u_width = u_limit_p + u_limit_n

        dv = 1 / (2 * Sy)
        v_limit_p = ((y0 + 1 / (2 * dv)) ** (-2) * z0 ** 2 + 1) ** (-1 / 2) / wv
        v_limit_n = ((y0 - 1 / (2 * dv)) ** (-2) * z0 ** 2 + 1) ** (-1 / 2) / wv

        if Sy < y0:
            # print('Sy < y0')
            v0 = (v_limit_p + v_limit_n) / 2
            v_width = v_limit_p - v_limit_n
        elif y0 <= -Sy:
            # print('y0 <= -Sy')
            v0 = -(v_limit_p + v_limit_n) / 2
            v_width = v_limit_n - v_limit_p
        else:
            # print('else')
            v0 = (v_limit_p - v_limit_n) / 2
            v_width = v_limit_p + v_limit_n

        fx_max = u_width / 2
        fy_max = v_width / 2

        H_filter = (torch.abs(fX - u0) <= fx_max) * (torch.abs(fY - v0) < fy_max)

        H_final = H * H_filter

        return H_final
        
class MaskBlockPhase(nn.Module):
    def __init__(self, 
                 in_channel, 
                 mask_x_num, mask_y_num, mask_base_thick,
                 mask_init_method,
                 total_x_num, total_y_num,
                 ridx_mask, freq, c, attenu_factor):
        super(MaskBlockPhase, self).__init__()

        self.register_parameter('mask_phase',
                                nn.Parameter(torch.Tensor(1, in_channel, mask_x_num, mask_y_num),
                                             requires_grad=True))
        
        freq = np.array(freq, dtype=np.float32)
        ridx_mask = np.array(ridx_mask, dtype=np.float32)
        attenu_factor = np.array(attenu_factor, dtype=np.float32)

        if mask_init_method == 'zero':
            nn.init.zeros_(self.mask_phase.data)
        elif mask_init_method == 'normal':
            nn.init.normal_(self.mask_phase.data, mean=0., std=0.5)

        self.pad_x = total_x_num // 2 - mask_x_num // 2
        self.pad_y = total_y_num // 2 - mask_y_num // 2 # needs to pad zero
        
        self.phase_change_factor = 2 * np.pi * (ridx_mask - 1) \
            * freq / c # (n-1)*k used to calculate the height from phase
        self.amp_decay_factor = 2 * np.pi * attenu_factor \
            * freq / c # amplitude decay
        
        self.mask_base_thick = mask_base_thick

    def forward(self, x):
        # sigmoid on the mask phase, to prevent out of [0, 2pi] phase
        mask_phase = torch.sigmoid(self.mask_phase) * 2 * np.pi
        mask_amp = torch.ones_like(mask_phase)

        mask_phase = F.pad(mask_phase, (self.pad_y, self.pad_y, self.pad_x, self.pad_x))
        mask_amp = F.pad(mask_amp, (self.pad_y, self.pad_y, self.pad_x, self.pad_x))

        # NOTE: it is the phase of the mask, not the height
        mask_cplx = torch.complex(torch.cos(mask_phase),
                                  torch.sin(mask_phase)) * torch.abs(mask_amp)
        
        # NOTE: for physical layer, please apply phase change

        # if amplitude decay
        if self.amp_decay_factor.any() > 0.:
            mask_height = self.mask_base_thick + \
                mask_phase / self.phase_change_factor
            mask_amp = torch.exp(- mask_height * self.amp_decay_factor)
            mask_cplx = mask_amp * mask_cplx

        output = torch.mul(mask_cplx, x) # should be mul not matmul
        return output
    
class Digital_Encoder(nn.Module):
    def __init__(self, img_size, in_channel=3):
        super().__init__()
        self.img_size = img_size
        self.in_channel = in_channel

        self.l1 = nn.Sequential(nn.Linear(img_size*img_size*in_channel, (img_size)*(img_size)*in_channel),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear(img_size*img_size*in_channel, (img_size)*(img_size)*in_channel),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear(img_size*img_size*in_channel, (img_size)*(img_size)*in_channel+in_channel))

    def forward(self, x):
        b, _, _, _ = x.shape
        x_flat = x.view(b, -1).contiguous()
        out = self.l1(x_flat)
        out_img = out[:, :-self.in_channel]
        if self.in_channel == 1:
            out_scale = out[:, -self.in_channel:].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            out_scale = out[:, -self.in_channel:].unsqueeze(-1).unsqueeze(-1)
        img = out_img.view(out.shape[0], self.in_channel, self.img_size, self.img_size).contiguous()
        return img, out_scale

class Digital_Encoder_ClsEmd(nn.Module):
    def __init__(self, img_size, in_channel=1, num_classes=10, dim_expand_ratio=8):
        super().__init__()
        self.img_size = img_size
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.dim_expand_ratio = dim_expand_ratio

        self.class_embedding = nn.Embedding(num_classes, dim_expand_ratio)

        # # preprocessing
        self.l1 = nn.Sequential(nn.Linear(img_size*img_size*in_channel+dim_expand_ratio, (img_size)*(img_size)*in_channel),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear(img_size*img_size*in_channel, (img_size)*(img_size)*in_channel),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear(img_size*img_size*in_channel, (img_size)*(img_size)*in_channel+in_channel))

    def forward(self, x, classes):
        b, _, _, _ = x.shape
        x_flat = torch.cat((x.view(b, -1).contiguous(), self.class_embedding(classes)), -1)
        out = self.l1(x_flat)
        out_img = out[:, :-self.in_channel]
        if self.in_channel == 1:
            out_scale = out[:, -self.in_channel:].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            out_scale = out[:, -self.in_channel:].unsqueeze(-1).unsqueeze(-1)
        img = out_img.view(out.shape[0], self.in_channel, self.img_size, self.img_size).contiguous()
        return img, out_scale
    
class Digital_Encoder_TimEmd(nn.Module):
    def __init__(self, img_size, in_channel=1, Timemd_dim=8):
        super().__init__()
        self.img_size = img_size
        self.in_channel = in_channel
        self.Timemd_dim = Timemd_dim

        self.l0 = nn.Sequential(nn.Linear(img_size*img_size*in_channel, (img_size)*(img_size)*in_channel),
                                nn.LeakyReLU(0.2, inplace=True))
        # # preprocessing
        self.l1 = nn.Sequential(nn.Linear(img_size*img_size*in_channel+Timemd_dim, (img_size)*(img_size)*in_channel),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear(img_size*img_size*in_channel, (img_size)*(img_size)*in_channel),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear(img_size*img_size*in_channel, (img_size)*(img_size)*in_channel+in_channel))

    def forward(self, x, t_emb):
        b, _, _, _ = x.shape
        x_flat = x.view(b, -1).contiguous()
        x0 = self.l0(x_flat)
        hidden_states = torch.cat((x0, t_emb), -1)
        out = self.l1(hidden_states)
        out_img = out[:, :-self.in_channel]
        if self.in_channel == 1:
            out_scale = out[:, -self.in_channel:].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            out_scale = out[:, -self.in_channel:].unsqueeze(-1).unsqueeze(-1)
        img = out_img.view(out.shape[0], self.in_channel, self.img_size, self.img_size).contiguous()
        return img, out_scale

class Digital_Encoder_TimClsEmd(nn.Module):
    def __init__(self, img_size, in_channel=1, Timemd_dim=8, num_classes=10, Clsemd_dim=8):
        super().__init__()
        self.img_size = img_size
        self.in_channel = in_channel
        self.Timemd_dim = Timemd_dim
        self.Clsemd_dim = Clsemd_dim

        self.class_embedding = nn.Embedding(num_classes, Clsemd_dim)

        self.l0 = nn.Sequential(nn.Linear(img_size*img_size*in_channel, (img_size)*(img_size)*in_channel),
                                nn.LeakyReLU(0.2, inplace=True))
        # # preprocessing
        self.l1 = nn.Sequential(nn.Linear(img_size*img_size*in_channel+Timemd_dim+Clsemd_dim, (img_size)*(img_size)*in_channel),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear(img_size*img_size*in_channel, (img_size)*(img_size)*in_channel),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear(img_size*img_size*in_channel, (img_size)*(img_size)*in_channel+in_channel))

    def forward(self, x, t_emb, classes):
        b, _, _, _ = x.shape
        c_emb = self.class_embedding(classes)
        x_flat = x.view(b, -1).contiguous()
        x0 = self.l0(x_flat)
        hidden_states = torch.cat((x0, t_emb, c_emb), -1)
        out = self.l1(hidden_states)
        out_img = out[:, :-self.in_channel]
        if self.in_channel == 1:
            out_scale = out[:, -self.in_channel:].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            out_scale = out[:, -self.in_channel:].unsqueeze(-1).unsqueeze(-1)
        img = out_img.view(out.shape[0], self.in_channel, self.img_size, self.img_size).contiguous()
        return img, out_scale
'''    
class Digital_Encoder_TextTimEmd(nn.Module):
    """
    텍스트(CLIP) + Timestep 임베딩을 받는 Digital Encoder
    RAT-Diffusion 스타일로 텍스트를 LSTM으로 처리
    """
    def __init__(self, img_size, in_channel=3, Timemd_dim=128, text_hidden_dim=512):
        super().__init__()
        self.img_size = img_size
        self.in_channel = in_channel
        self.Timemd_dim = Timemd_dim
        self.text_hidden_dim = text_hidden_dim
        
        # CLIP 텍스트 임베딩 처리용 LSTM
        self.text_lstm = CustomLSTM(input_sz=512, hidden_sz=text_hidden_dim)
        
        # 이미지 전처리
        self.l0 = nn.Sequential(
            nn.Linear(img_size*img_size*in_channel, img_size*img_size*in_channel),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Timestep + Text embedding 결합 후 처리
        # text_hidden_dim이 LSTM output, Timemd_dim이 timestep embedding
        self.l1 = nn.Sequential(
            nn.Linear(img_size*img_size*in_channel + Timemd_dim + text_hidden_dim, 
                     img_size*img_size*in_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(img_size*img_size*in_channel, img_size*img_size*in_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(img_size*img_size*in_channel, img_size*img_size*in_channel + in_channel)
        )
    
    def forward(self, x, t_emb, text_emb):
        """
        Args:
            x: noisy image (B, C, H, W)
            t_emb: timestep embedding (B, Timemd_dim)
            text_emb: CLIP text embedding (B, 512)
        Returns:
            img: encoded image
            out_scale: scaling factor
        """
        b, _, _, _ = x.shape
        
        # 텍스트 임베딩을 LSTM으로 처리
        self.text_lstm.init_hidden(text_emb)
        text_hidden = self.text_lstm(text_emb)  # (B, text_hidden_dim)
        
        # 이미지 flatten 및 전처리
        x_flat = x.view(b, -1).contiguous()
        x0 = self.l0(x_flat)
        
        # Timestep + Text embedding 결합
        hidden_states = torch.cat((x0, t_emb, text_hidden), dim=-1)
        
        # 최종 인코딩
        out = self.l1(hidden_states)
        out_img = out[:, :-self.in_channel]
        
        if self.in_channel == 1:
            out_scale = out[:, -self.in_channel:].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            out_scale = out[:, -self.in_channel:].unsqueeze(-1).unsqueeze(-1)
        
        img = out_img.view(b, self.in_channel, self.img_size, self.img_size).contiguous()
        
        return img, out_scale
'''

class Digital_Encoder_TextTimEmd(nn.Module):
    # 텍스트(CLIP) + Timestep 임베딩을 받는 Digital Encoder
    # RAT-Diffusion 스타일로 텍스트를 LSTM으로 처리

    def __init__(self, img_size, in_channel=3, Timemd_dim=128, text_hidden_dim=512):
        super().__init__()
        self.img_size = img_size
        self.in_channel = in_channel
        self.Timemd_dim = Timemd_dim
        self.text_hidden_dim = text_hidden_dim
        
        # CLIP 텍스트 임베딩 처리용 LSTM
        self.text_lstm = CustomLSTM(input_sz=512, hidden_sz=text_hidden_dim)
        
        # 이미지 전처리
        self.l0 = nn.Sequential(
            nn.Linear(img_size*img_size*in_channel, img_size*img_size*in_channel),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # cross-attention proj & fusion
        self.time_proj = nn.Linear(Timemd_dim, img_size*img_size*in_channel)
        self.text_proj = nn.Linear(text_hidden_dim, img_size*img_size*in_channel)
        self.cross_attn = nn.MultiheadAttention(img_size*img_size*in_channel, num_heads=4, batch_first=True)
        self.out_proj = nn.Linear(2 * img_size*img_size*in_channel, img_size*img_size*in_channel)

        # 기존 결합 후 인코더
        self.l1 = nn.Sequential(
            nn.Linear(img_size*img_size*in_channel, img_size*img_size*in_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(img_size*img_size*in_channel, img_size*img_size*in_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(img_size*img_size*in_channel, img_size*img_size*in_channel + in_channel)
        )
    
    def forward(self, x, t_emb, text_emb):
    
        # Args:
        #     x: noisy image (B, C, H, W)
        #     t_emb: timestep embedding (B, Timemd_dim)
        #     text_emb: CLIP text embedding (B, 512)
        # Returns:
        #     img: encoded image
        #     out_scale: scaling factor

        b, _, _, _ = x.shape
        
        # 텍스트 임베딩을 LSTM으로 처리

        # LSTM 빼보기
        #self.text_lstm.init_hidden(text_emb)
        #text_hidden = self.text_lstm(text_emb)  # (B, text_hidden_dim)
        #text_emb = (batch size, 512) 512는 Clip model에서 지정한 텍스트 임베딩 차원
        #text_hidden = (batch size, text_hidden_dim)
        #LSTM forward pass를 수행  
        #여러 게이트 연산을 수행하여 정보의 일부를 강조 or 불필요한 부분을 무시

        
        # 이미지 flatten 및 전처리
        x_flat = x.view(b, -1).contiguous()
        x0 = self.l0(x_flat)   # (B, feat_dim)
        # x = (batch size,채널 수, 세로 픽셀 수, 가로 픽셀 수)
        # x_flat = (batch size,채널 수*세로 픽셀 수*가로 픽셀 수)
        # x0 = (batch size,채널 수*세로 픽셀 수*가로 픽셀 수)
        # l0에서는 nn.linear와 LeakyReLU를 통과시킴 (input vector끼리 서로 곱해지고 더해진다.)


        # Timestep, text projection해서 차원 맞춤
        t_proj = self.time_proj(t_emb)         # (B, feat_dim)
        txt_proj = self.text_proj(text_emb) # (B, feat_dim)
        # Timemd_dim,text_hidden_dim를 채널 수*세로 픽셀 수*가로 픽셀 수로 맞춤

        # reshape for MultiheadAttention: (B, 1, feat_dim)
        x0q = x0.unsqueeze(1)     # (B, 1, feat_dim)
        tkv = (t_proj + txt_proj).unsqueeze(1) # (B, 1, feat_dim) (또는 별도 / 여러개면 cat 또는 stack)
        # x0 = (batch size, 채널 수*세로 픽셀 수*가로 픽셀 수)
        # x0q = (batch size, 1, 채널 수*세로 픽셀 수*가로 픽셀 수)
        # tkv = (batch size, 1, 채널 수*세로 픽셀 수*가로 픽셀 수)

        # cross-attention (query: img, key/value: t_proj+txt_proj)
        attn_output, _ = self.cross_attn(x0q, tkv, tkv)   # (B, 1, feat_dim)
        attn_fused = torch.cat([x0q, attn_output], dim=-1).squeeze(1)  # (B, 2*feat_dim)
        fused = self.out_proj(attn_fused) # (B, feat_dim)
        # attn_output = (batch size, 1, 채널 수*세로 픽셀 수*가로 픽셀 수)   -> attention feature 얻기
        # attn_fused = (batch size, 2*채널 수*세로 픽셀 수*가로 픽셀 수)     -> 원본 이미지 feature와 cross-attention을 통해 얻은 attention feature 하나로 붙이기
        # fused = (batch size, 채널 수*세로 픽셀 수*가로 픽셀 수) -> 두 정보 섞어서 한개로 만들기


        # 기존 인코더 적용
        out = self.l1(fused)
        out_img = out[:, :-self.in_channel]
        # out = (batch size, 채널 수*세로 픽셀 수*가로 픽셀 수 + 채널 수)
        # 여러 개의 linear와 LeakyReLU 반복
        # out_img = (batch size, 채널 수*세로 픽셀 수*가로 픽셀 수)


        if self.in_channel == 1:
            out_scale = out[:, -self.in_channel:].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            out_scale = out[:, -self.in_channel:].unsqueeze(-1).unsqueeze(-1)
        # 이 값은 안 씀

        img = out_img.view(b, self.in_channel, self.img_size, self.img_size).contiguous()
        # img =  (batch size, 채널 수, 세로 픽셀 수, 가로 픽셀 수)
        return img, out_scale
