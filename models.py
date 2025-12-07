import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from diffusers.utils import BaseOutput
from modules import FreeSpaceProp, FreeSpaceProp_Multich, MaskBlockPhase, Digital_Encoder, Digital_Encoder_ClsEmd, Digital_Encoder_TimEmd, Digital_Encoder_TimClsEmd, Digital_Encoder_TextTimEmd

@dataclass
class DiffD2nnOutput(BaseOutput):

    sample: torch.FloatTensor
    scale: torch.FloatTensor
    
class Iterative_Optical_Generative_Model(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, 
                 img_size, 
                 in_channel,
                 #num_classes, --> 이제 class label은 필요없음 
                 dim_expand_ratio,
                 c, 
                 num_masks,
                 wlength_vc,
                 ridx_air, ridx_mask, attenu_factor,
                 total_x_num, total_y_num,
                 mask_x_num, mask_y_num, mask_init_method, mask_base_thick,
                 dx, dy,
                 object_mask_dist, mask_mask_dist, mask_sensor_dist,
                 obj_x_num, obj_y_num,
                 train_batch_size,
                 time_embedding_type = "positional",
                 num_train_timesteps = None):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channel
        self.total_x_num = total_x_num
        self.total_y_num = total_y_num
        self.mask_x_num = mask_x_num
        self.mask_y_num = mask_y_num
        self.obj_x_num = obj_x_num
        self.obj_y_num = obj_y_num

        if in_channel == 1:
            freq = c / wlength_vc
        else:
            freq = [c / wl for wl in wlength_vc] 

        time_embed_dim = img_size * 4

        # Digital Encoder 
        self.DE = Digital_Encoder_TextTimEmd(
            img_size=img_size, 
            in_channel=in_channel, 
            Timemd_dim=time_embed_dim,
            text_hidden_dim=512
        )

        # Diffractive Decoder blocks
        self.DD = nn.ModuleList()
        # NOTE: default here is multicolor generation. For single wavelength, please change the model accordingly
        self.DD.append(FreeSpaceProp_Multich(wlength_vc=wlength_vc, 
                                             ridx_air=ridx_air,
                                             total_x_num=total_x_num, total_y_num=total_y_num,
                                             dx=dx, dy=dy,
                                             prop_z=object_mask_dist)) # distance takein here
        self.DD.append(MaskBlockPhase(in_channel=in_channel, mask_x_num=mask_x_num, mask_y_num=mask_y_num, 
                                        mask_base_thick=mask_base_thick, mask_init_method=mask_init_method,
                                        total_x_num=total_x_num, total_y_num=total_y_num, ridx_mask=ridx_mask,
                                        freq=freq, c=c, attenu_factor=attenu_factor))
        
        for _ in range(num_masks - 1):
            self.DD.append(FreeSpaceProp_Multich(wlength_vc=wlength_vc, 
                                                 ridx_air=ridx_air,
                                                 total_x_num=total_x_num, total_y_num=total_y_num,
                                                 dx=dx, dy=dy,
                                                 prop_z=mask_mask_dist)) 
            self.DD.append(MaskBlockPhase(in_channel=in_channel, mask_x_num=mask_x_num, mask_y_num=mask_y_num, 
                                            mask_base_thick=mask_base_thick, mask_init_method=mask_init_method,
                                            total_x_num=total_x_num, total_y_num=total_y_num, ridx_mask=ridx_mask,
                                            freq=freq, c=c, attenu_factor=attenu_factor))
            
        self.DD.append(FreeSpaceProp_Multich(wlength_vc=wlength_vc, 
                                             ridx_air=ridx_air,
                                             total_x_num=total_x_num, total_y_num=total_y_num,
                                             dx=dx, dy=dy,
                                             prop_z=mask_sensor_dist))
        
        # time embedding
        if time_embedding_type == 'fourier':
            self.time_proj = GaussianFourierProjection(embedding_size=time_embed_dim, scale=16)
            timestep_input_dim = 2 * time_embed_dim
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(time_embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "learned":
            self.time_proj = nn.Embedding(num_train_timesteps, time_embed_dim)
            timestep_input_dim = time_embed_dim

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

    def forward(self, x, timestep=0, text_emb=None, return_dict=True, return_intermediate = False):
        """
        Args:
            x: noisy image
            timestep: diffusion timestep
            text_emb: CLIP text embedding (B, 512) - 새로 추가!
            return_dict: whether to return dict
        """
        # Timestep 처리 
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)
        timesteps = timesteps * torch.ones(x.shape[0], dtype=timesteps.dtype, device=timesteps.device)
        # timesteps = (batch size,) 1차원 텐서로
        
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(self.dtype)
        t_emb = self.time_embedding(t_emb)
        # positional 사용, timestep을 positional embedding 함
        # (batch size, time_embed_dim = 4*img_size)
        
        # *** 수정: text_emb를 Digital Encoder에 전달 ***
        x_encoded, scale = self.DE(x, t_emb, text_emb)

        intermediates = [] if return_intermediate else None
        # x_encoded = (batch size, 채널 수, 세로 픽셀 수, 가로 픽셀 수)

        
        # 광학 전파 
        img_cplx = self.img_preprocess(x_encoded)

        if return_intermediate:
            intermediates.append(self.center_crop(img_cplx.detach().clone(), [self.obj_y_num, self.obj_x_num]))
        # img_cplx = (batch size, 채널 수, 세로 픽셀 수, 가로 픽셀 수)
        # 기존과 shape은 동일. 다만 img를 phase에 투영한 후, SLM 크기에 맞게 리샘플링한 값

        
        for blocks in self.DD:
            img_cplx = blocks(img_cplx)
            if return_intermediate:
                intermediates.append(self.center_crop(img_cplx.detach().clone(), [self.obj_y_num, self.obj_x_num]))
        # 각 block를 통과해도 shape은 변하지 않음.
        
        img_cplx = self.center_crop(img_cplx, [self.obj_y_num, self.obj_x_num])
        img_cplx = torch.abs(img_cplx)
        # 실제 사용할 부분 (obj_y_num, obj_x_num) 즉 센서 사이즈만큼만 crop함
        # 그후 abs를 씌워 amplitude만 추출
        
        output = F.avg_pool2d(img_cplx, kernel_size=self.obj_y_num//self.img_size, 
                              stride=self.obj_y_num//self.img_size, padding=0)
        output = torch.square(output)
        # Result image를 평균값 필터를 씌운후 Amplitude를 Intensity로 변환 (I = |A|^2)
        
        if not return_dict:
            if return_intermediate:
                return output, scale, intermediates
            return (output, scale)
        else:
            if return_intermediate:
                return DiffD2nnOutput(sample=output, scale=scale), intermediates
            return DiffD2nnOutput(sample=output, scale=scale)

    def img_preprocess(self, x):
        alpha = 1.0
        x = (x * alpha * np.pi + alpha * np.pi).clamp(0.0, 2 * alpha * torch.tensor(np.pi).to(x.device))
        x_cplx = torch.complex(torch.cos(x), torch.sin(x))
        img_input = self.resize_phase_complex(x_cplx)
        return img_input
    
    def resize_phase_complex(self, x):
        pad_x = (self.total_x_num // 2 - self.obj_x_num // 2)
        pad_y = (self.total_y_num // 2 - self.obj_y_num // 2)
        output_real = F.interpolate(x.real, size=[self.obj_y_num, self.obj_x_num], mode='nearest')
        output_imag = F.interpolate(x.imag, size=[self.obj_y_num, self.obj_x_num], mode='nearest')
        output_real = F.pad(output_real, (pad_y, pad_y, pad_x, pad_x))
        output_imag = F.pad(output_imag, (pad_y, pad_y, pad_x, pad_x))
        
        return torch.complex(output_real, output_imag)
     
    def center_crop(self, x: torch.tensor, size: list):
        output = x[..., self.total_y_num // 2 - size[0] // 2 : self.total_y_num // 2 + size[0] // 2,
                        self.total_x_num // 2 - size[1] // 2 : self.total_x_num // 2 + size[1] // 2]
        output = output.contiguous()
        return output
    