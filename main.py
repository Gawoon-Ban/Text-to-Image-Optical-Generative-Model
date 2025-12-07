import os
import json
import math
import torch
import torch.nn.functional as F
import argparse
from PIL import Image
from safetensors.torch import load_file
from tqdm.auto import tqdm
from torchvision import transforms
from datasets import load_from_disk
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from accelerate import Accelerator, notebook_launcher
from functools import partial

from utils import _extract_into_tensor, kl_divergence_loss
from models import Iterative_Optical_Generative_Model
from pipeline_costum import DDPMPipeline_Costum, DDPMPipeline_Costum_ClsEmb, DDPMPipeline_TextCond  

import numpy as np
import clip
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Main configuration for optical generative models")
    parser.add_argument("--task", type=str, default="diffusion_digital", help="training task", 
                        choices=["diffusion_digital", "snapshot_optical", "multicolor_optical", "iterative_optical"])
    parser.add_argument("--num_gpu", type=int, default=1, help="Number of GPU to use")
    parser.add_argument("--data_path", type=str, default="data", help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default="./logs/exp", help="Where to save logs and checkpoints")
    parser.add_argument("--sample_size", type=int, default=32)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--out_channels", type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--train_batch_size", type=int, default=200)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate_digital", type=float, default=1e-4)
    parser.add_argument("--learning_rate_optical", type=float, default=5e-3)
    parser.add_argument("--prediction_type", type=str, default="epsilon", choices=["epsilon", "sample"])
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default='linear')
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16"])
    parser.add_argument("--seed", type=int, default=96)
    parser.add_argument("--save_image_epochs", type=int, default=10)
    parser.add_argument("--save_model_epochs", type=int, default=50)
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=0, help="For the dataset with class indices")

    ## for diffusion_digital task
    parser.add_argument("--time_embedding_type_d", type=str, default="positional")
    parser.add_argument(
        "--down_block_types",
        type=str,
        nargs="+",
        default=["DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"]
    )
    parser.add_argument(
        "--up_block_types",
        type=str,
        nargs="+",
        default=["AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"]
    )
    parser.add_argument(
        "--block_out_channels",
        type=int,
        nargs="+",
        default=[224, 448, 672, 896]
    )
    parser.add_argument("--layers_per_block", type=int, default=2)
    parser.add_argument("--prediction_type_d", type=str, default='epsilon')

    ## for optical generation task
    parser.add_argument("--c", type=float, default=299792458.0)
    parser.add_argument("--ridx_air", type=float, default=1.0)
    
    parser.add_argument("--object_layer_dist", type=float, default=5e-2)
    parser.add_argument("--layer_layer_dist", type=float, default=1e-2)
    parser.add_argument("--layer_sensor_dist", type=float, default=5e-2)

    parser.add_argument("--num_layer", type=int, default=1)
    parser.add_argument("--total_num", type=int, default=800)
    parser.add_argument("--obj_num", type=int, default=320)
    parser.add_argument("--layer_neuron_num", type=int, default=400)
    parser.add_argument("--dxdy", type=float, default=8e-6)
    parser.add_argument("--layer_init_method", type=str, default='zero')

    parser.add_argument("--amp_modulation", type=bool, default=False)

    ## for snapshot_optical task
    parser.add_argument("--teacher_ckpt_snst", type=str, default="./teacher_snapshot")

    parser.add_argument("--inference_acc_snst", type=bool, default=True)
    parser.add_argument("--acc_ratio_snst", type=int, default=20)
    
    parser.add_argument("--wavelength_snst", type=float, default=520e-9)
    parser.add_argument("--ridx_layer_snst", type=float, default=1.0) # needs update for physical layer
    parser.add_argument("--attenu_factor_snst", type=float, default=0.0) # needs update for physical layer

    parser.add_argument("--apply_scale_snst", type=bool, default=True)
    parser.add_argument("--scale_type_snst", type=str, default="neural_pred", choices=["static_mean", "neural_pred"])
    parser.add_argument("--noise_perturb_snst", type=float, default=1e-4)
    parser.add_argument("--eval_kl_snst", type=bool, default=False)
    parser.add_argument("--kl_ratio_snst", type=float, default=1e-4)

    ## for multicolor_optical task
    parser.add_argument("--teacher_ckpt_mtcl", type=str, default="./teacher_multicolor")

    parser.add_argument("--inference_acc_mtcl", type=bool, default=True)
    parser.add_argument("--acc_ratio_mtcl", type=int, default=20)
    
    parser.add_argument("--wavelength_mtcl", type=float, nargs='+', default=[450e-9, 520e-9, 638e-9], help="List of wavelengths in meters.")
    parser.add_argument("--ridx_layer_mtcl", type=float, nargs='+', default=[1.0, 1.0, 1.0]) # needs update for physical layer
    parser.add_argument("--attenu_factor_mtcl", type=float, nargs='+', default=[0.0, 0.0, 0.0]) # needs update for physical layer

    parser.add_argument("--apply_scale_mtcl", type=bool, default=False)
    parser.add_argument("--scale_type_mtcl", type=str, default="static_mean", choices=["static_mean", "neural_pred"])
    parser.add_argument("--noise_perturb_mtcl", type=float, default=1e-4)
    parser.add_argument("--eval_kl_mtcl", type=bool, default=False)
    parser.add_argument("--kl_ratio_mtcl", type=float, default=1e-4)

    # for iterative_optical task
    parser.add_argument("--time_embedding_type_o", type=str, default="positional")

    parser.add_argument("--wavelength_itrt", type=float, nargs='+', default=[450e-9, 520e-9, 638e-9], help="List of wavelengths in meters.")
    parser.add_argument("--ridx_layer_itrt", type=float, nargs='+', default=[1.0, 1.0, 1.0]) # needs update for physical layer
    parser.add_argument("--attenu_factor_itrt", type=float, nargs='+', default=[0.0, 0.0, 0.0]) # needs update for physical layer

    parser.add_argument("--prediction_type_o", type=str, default='sample', choices=["epsilon", "sample"])
    parser.add_argument("--beta_start_itrt", type=float, default=0.001)
    parser.add_argument("--beta_end_itrt", type=float, default=0.010)

    args = parser.parse_args()
    return args

def main(args):
    if args.output_dir is None:
        os.makedirs(args.output_dir, exist_ok=True)

    # load the dataset
    dataset = load_from_disk(args.data_path)['train']
    preprocess = transforms.Compose([
        transforms.Resize(size=(args.sample_size, args.sample_size), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    def transform(examples, wz_cls=False):
        images = [preprocess(Image.open(os.path.join("./data/flower/jpg", img_name)).convert('RGB')) for img_name in examples["img"]]
        captions = [txts[0] for txts in examples["text"]]
        return {"images": images, "texts": captions}
    
    wz_cls = False 
    transform_fn = partial(transform, wz_cls=wz_cls)
    dataset.set_transform(transform_fn)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    train_iterative(args, train_dataloader, wz_cls=wz_cls)        


def train_iterative(args, dataloader, wz_cls):
    model = Iterative_Optical_Generative_Model(
        img_size=args.sample_size,
        in_channel=args.in_channels,
        #num_classes=args.num_classes,
        dim_expand_ratio=128,
        c=args.c, num_masks=args.num_layer,
        wlength_vc=args.wavelength_mtcl,
        ridx_air=args.ridx_air, ridx_mask=args.ridx_layer_mtcl,
        attenu_factor=args.attenu_factor_mtcl,
        total_x_num=args.total_num, total_y_num=args.total_num,
        mask_x_num=args.layer_neuron_num, mask_y_num=args.layer_neuron_num,
        mask_init_method=args.layer_init_method, mask_base_thick=1.0e-3,
        dx=args.dxdy, dy=args.dxdy, 
        object_mask_dist=args.object_layer_dist, 
        mask_mask_dist=args.layer_layer_dist,
        mask_sensor_dist=args.layer_sensor_dist, 
        obj_x_num=args.obj_num, obj_y_num=args.obj_num,
        train_batch_size= args.train_batch_size,
        time_embedding_type=args.time_embedding_type_o,
        num_train_timesteps=args.ddpm_num_steps
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, 
                                    beta_schedule=args.ddpm_beta_schedule,
                                    prediction_type=args.prediction_type_o,
                                    beta_start=args.beta_start_itrt,
                                    beta_end=args.beta_end_itrt)
    
    optimizer = torch.optim.AdamW([{'params': model.DD.parameters(), 'lr': args.learning_rate_optical},
                                   {'params': model.DE.parameters(), 'lr': args.learning_rate_digital}], betas=(0.5, 0.999))
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(dataloader) * args.num_epochs)
    )
    clip_model, _ = clip.load("ViT-B/32", device='cuda')
    clip_model.eval()  # CLIP은 frozen

    visualize_epoch = 25  # 10 epoch마다    
    visualize_save_dir = "./visualizations"

    def evaluate(args, epoch, pipeline, text_emb_batch, text_list_batch=None):
        images = pipeline(
            batch_size=args.eval_batch_size,
            text_emb=text_emb_batch,  # (B, emb_dim)
            generator=torch.manual_seed(args.seed),
            num_inference_steps=args.ddpm_num_steps
        ).images
        num_images = len(images)
        cols = 1
        rows = num_images
        image_grid = make_image_grid(images, rows=rows, cols=cols)
        test_dir = os.path.join(args.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{epoch:04d}.png")

    def train_loop(args, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, wz_cls):
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(args.output_dir, "logs"),
        )
        if accelerator.is_main_process:
            accelerator.init_trackers("train_iterative")

        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

        global_step = 0

        ## training
        for epoch in range(args.num_epochs):
            epoch_losses = []
            running_loss = 0
            total_steps = 0
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                clean_images = batch["images"]
                texts = batch["texts"]

                with torch.no_grad():
                    text_tokens = clip.tokenize(texts, truncate=True).to(clean_images.device)
                    text_emb = clip_model.encode_text(text_tokens).float()
                
                noise = torch.randn_like(clean_images)
                batch_size = clean_images.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                         (batch_size,), device=clean_images.device, dtype=torch.int64)
                
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                
                with accelerator.accumulate(model):
                    if (epoch + 1) % visualize_epoch == 0 and step == 0:
                        noise_pred,scale,intermediates = model(noisy_images, timesteps, text_emb=text_emb, return_dict=False, return_intermediate=True)
                        save_optical_fields(intermediates=intermediates, epoch=epoch + 1, save_dir=visualize_save_dir,)
                    else:
                        noise_pred = model(noisy_images, timesteps, text_emb=text_emb, return_dict=False)[0]
                    
                    # Loss 계산 
                    if args.prediction_type_o == "epsilon":
                        loss = F.mse_loss(noise_pred, noise)
                    elif args.prediction_type_o == "sample":
                        alpha_t = _extract_into_tensor(
                            noise_scheduler.alphas_cumprod, timesteps, 
                            (clean_images.shape[0], 1, 1, 1)
                        )
                        snr_weights = alpha_t / (1 - alpha_t)
                        loss = snr_weights * F.mse_loss(noise_pred.float(), 
                                                       clean_images.float(), reduction="none")
                        loss = loss.mean()

                    ######################################
                    running_loss += loss.item()
                    total_steps += 1
                    ######################################

                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), 
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

                ########################################################################
                avg_epoch_loss = running_loss / total_steps
                epoch_losses.append(avg_epoch_loss)
                ############################################################

            if accelerator.is_main_process:
                model_unet = accelerator.unwrap_model(model)
                model_unet.config["sample_size"] = args.sample_size
                pipeline = DDPMPipeline_TextCond(unet=model_unet, scheduler=noise_scheduler)

                # 디버깅 용도로 매 에퐄마다로
                if (epoch + 1) % args.save_image_epochs == 0 or epoch == args.num_epochs - 1:
                    evaluate(args, epoch, pipeline, text_emb[:args.eval_batch_size], text_emb[:args.eval_batch_size])
                
                if (epoch + 1) % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                    #pipeline.save_pretrained(args.output_dir)
                    torch.save(model.state_dict(), "my_trained_model.pth")
                    torch.save(pipeline, "my_trained_pipeline.pth")


            plt.figure()
            plt.plot(range(1, len(epoch_losses)+1), epoch_losses, marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Average Loss')
            plt.title('Training Loss per Epoch')
            plt.grid(True)
            plt.savefig(f"{args.output_dir}/epoch_loss.png")
            plt.close()
            

    # start the train loop
    train_loop(args, model, noise_scheduler, optimizer, dataloader, lr_scheduler, wz_cls)
    # start the train loop with notebook launcher (only when you using notebook to launch the training)
    # config = (args, model, noise_scheduler, optimizer, dataloader, lr_scheduler)
    # notebook_launcher(train_loop, config, num_processes=args.num_gpu)

def save_optical_fields(intermediates, epoch, save_dir, channel_names=("R","G","B")):
    os.makedirs(save_dir, exist_ok=True)
    for step_idx, img_cplx in enumerate(intermediates):
        field = img_cplx[0].detach().cpu()  # (C,H,W), complex
        amp = torch.abs(field).numpy()
        inten = (amp ** 2)
        phase = torch.angle(field).numpy()

        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        for c in range(min(3, field.shape[0])):
            name = channel_names[c] if c < len(channel_names) else f"C{c}"
            axes[0, c].imshow(amp[c], cmap="gray");  axes[0, c].set_title(f"{name} amplitude"); axes[0, c].axis("off")
            axes[1, c].imshow(inten[c], cmap="gray"); axes[1, c].set_title(f"{name} intensity"); axes[1, c].axis("off")
            axes[2, c].imshow(phase[c], cmap="hsv", vmin=-np.pi, vmax=np.pi); axes[2, c].set_title(f"{name} phase"); axes[2, c].axis("off")

        for r in range(3):
            for c in range(3):
                if c >= field.shape[0]:
                    axes[r, c].axis("off")

        fig.suptitle(f"Epoch {epoch}, Step {step_idx}")
        plt.tight_layout()
        out = os.path.join(save_dir, f"epoch_{epoch:03d}_step_{step_idx:02d}.png")
        plt.savefig(out, dpi=200)
        plt.close(fig)


if __name__ == "__main__":#
    args = parse_args()
    main(args)