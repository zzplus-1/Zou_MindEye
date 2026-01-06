#!/usr/bin/env python3
"""
三维脑图像重建脚本
清理自Jupyter Notebook, 移除所有交互式调试代码
"""

import argparse
import os
import sys
import gc
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
import webdataset as wds

import utils
from models import Clipper, BrainNetwork, BrainDiffusionPrior, VersatileDiffusionPriorNetwork, Voxel2StableDiffusionModel
from diffusers import VersatileDiffusionDualGuidedPipeline, UniPCMultistepScheduler
from diffusers.models import DualTransformer2DModel

def main():
    # 设置随机种子
    seed = 42
    utils.seed_everything(seed=seed)
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="三维脑图像重建模型配置")
    
    parser.add_argument(
        "--model_name", type=str, default="testing",
        help="训练模型名称",
    )
    parser.add_argument(
        "--autoencoder_name", type=str, default="None",
        help="自编码器模型名称",
    )
    parser.add_argument(
        "--data_path", type=str, default="/fsx/proj-medarc/fmri/natural-scenes-dataset",
        help="NSD数据存储路径",
    )
    parser.add_argument(
        "--subj", type=int, default=1, choices=[1, 2, 5, 7],
        help="受试者编号",
    )
    parser.add_argument(
        "--img2img_strength", type=float, default=0.85,
        help="图像到图像转换强度 (1=无img2img; 0=输出低级图像本身)",
    )
    parser.add_argument(
        "--recons_per_sample", type=int, default=1,
        help="每个样本输出的重建数量，用于自动选择最佳结果",
    )
    parser.add_argument(
        "--vd_cache_dir", type=str, 
        default='/fsx/proj-medarc/fmri/cache/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7',
        help="Versatile Diffusion模型缓存路径",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=20,
        help="推理步骤数",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="批处理大小",
    )
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device)
    
    # 根据受试者编号确定体素数量
    voxel_counts = {
        1: 15724,
        2: 14278,
        3: 15226,
        4: 13153,
        5: 13039,
        6: 17907,
        7: 12682,
        8: 14386
    }
    num_voxels = voxel_counts.get(args.subj, 15724)
    print(f"受试者: {args.subj}, 体素数量: {num_voxels}")
    
    # 数据加载
    val_url = f"{args.data_path}/webdataset_avg_split/test/test_subj0{args.subj}_" + "{0..1}.tar"
    meta_url = f"{args.data_path}/webdataset_avg_split/metadata_subj0{args.subj}.json"
    num_val = 982
    voxels_key = 'nsdgeneral.npy'
    
    val_data = wds.WebDataset(val_url, resampled=False)\
        .decode("torch")\
        .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
        .to_tuple("voxels", "images", "coco")\
        .batched(args.batch_size, partial=False)
    
    val_dl = torch.utils.data.DataLoader(val_data, batch_size=None, shuffle=False)
    
    # 加载自编码器（如果需要）
    if args.autoencoder_name != "None":
        outdir = f'../train_logs/{args.autoencoder_name}'
        ckpt_path = os.path.join(outdir, 'epoch120.pth')
        
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            state_dict = checkpoint['model_state_dict']
            
            voxel2sd = Voxel2StableDiffusionModel(in_dim=num_voxels)
            voxel2sd.load_state_dict(state_dict, strict=False)
            voxel2sd = voxel2sd.to(device).half()
            voxel2sd.eval()
            voxel2sd.to(device)
            print("已加载低级模型!")
        else:
            print("未找到低级模型有效路径；不使用img2img!")
            args.img2img_strength = 1
    else:
        voxel2sd = None
        args.img2img_strength = 1
    
    # 加载Versatile Diffusion管道
    print('创建Versatile Diffusion重建管道...')
    # try:
    vd_pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained("/root/fMRI-reconstruction-NSD-main/cache/versatile-diffusion", local_files_only=True, torch_dtype=torch.float16).to(device)
    # except:
    #     print(f"下载Versatile Diffusion到 {args.vd_cache_dir}")
    #     vd_pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained(
    #         "shi-labs/versatile-diffusion",
    #         cache_dir=args.vd_cache_dir
    #     ).to(device).to(torch.float16)
    
    vd_pipe.image_unet.eval()
    vd_pipe.vae.eval()
    vd_pipe.image_unet.requires_grad_(False)
    vd_pipe.vae.requires_grad_(False)
    
    vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained(args.vd_cache_dir, subfolder="scheduler")
    
    # 设置双引导权重
    text_image_ratio = 0.0  # 0表示仅使用图像
    for name, module in vd_pipe.image_unet.named_modules():
        if isinstance(module, DualTransformer2DModel):
            module.mix_ratio = text_image_ratio
            for i, cond_type in enumerate(("text", "image")):
                if cond_type == "text":
                    module.condition_lengths[i] = 77
                    module.transformer_index_for_condition[i] = 1
                else:
                    module.condition_lengths[i] = 257
                    module.transformer_index_for_condition[i] = 0
    
    unet = vd_pipe.image_unet
    vae = vd_pipe.vae
    noise_scheduler = vd_pipe.scheduler
    
    # 加载扩散先验模型
    print('加载扩散先验模型...')
    img_variations = False
    
    out_dim = 257 * 768
    clip_extractor = Clipper("ViT-L/14", hidden_state=True, norm_embs=True, device=device)
    voxel2clip_kwargs = dict(in_dim=num_voxels, out_dim=out_dim)
    voxel2clip = BrainNetwork(**voxel2clip_kwargs)
    voxel2clip.requires_grad_(False)
    voxel2clip.eval()
    
    out_dim = 768
    depth = 6
    dim_head = 64
    heads = 12
    timesteps = 100
    
    prior_network = VersatileDiffusionPriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        learned_query_mode="pos_emb"
    )
    
    diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
        voxel2clip=voxel2clip,
    )
    
    outdir = f'../train_logs/{args.model_name}'
    ckpt_path = os.path.join(outdir, 'last.pth')
    
    print(f"检查点路径: {ckpt_path}")
    # checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint['model_state_dict']
    print(f"训练轮数: {checkpoint['epoch']}")
    
    diffusion_prior.load_state_dict(state_dict, strict=False)
    diffusion_prior = diffusion_prior.to(device)

    gc.collect()
    torch.cuda.empty_cache()

    diffusion_prior.eval().to(device)
    diffusion_priors = [diffusion_prior]
    
    # 开始重建
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    retrieve = False
    plotting = False
    saving = True
    verbose = False
    imsize = 512
    
    guidance_scale = 3.5 if not img_variations else 7.5
    ind_include = np.arange(num_val)
    # all_brain_recons = None
    all_brain_recons = []
    # all_images = None
    all_images = []
    
    # 设置img2img模式
    if args.img2img_strength == 1:
        img2img = False
        only_lowlevel = False
    elif args.img2img_strength == 0:
        img2img = True
        only_lowlevel = True
    else:
        img2img = True
        only_lowlevel = False
    
    for val_i, (voxel, img, coco) in enumerate(tqdm(val_dl, total=len(ind_include))):
        if val_i < np.min(ind_include):
            continue
            
        voxel = torch.mean(voxel, axis=1).to(device)
        print(f"voxel dtype: {voxel.dtype}")
        
        with torch.no_grad():
            # img2img处理
            if img2img and voxel2sd is not None:
                ae_preds = voxel2sd(voxel)
                blurry_recons = vd_pipe.vae.decode(ae_preds.to(device).half() / 0.18215).sample / 2 + 0.5
            else:
                blurry_recons = None
            
            # 低级图像直接作为重建结果
            if only_lowlevel:
                brain_recons = blurry_recons
            else:
                # 完整重建过程
                grid, brain_recons, laion_best_picks, recon_img = utils.reconstruction(
                    img, voxel,
                    clip_extractor, unet, vae, noise_scheduler,
                    voxel2clip_cls=None,
                    diffusion_priors=diffusion_priors,
                    text_token=None,
                    img_lowlevel=blurry_recons,
                    num_inference_steps=args.num_inference_steps,
                    n_samples_save=args.batch_size,
                    recons_per_sample=args.recons_per_sample,
                    guidance_scale=guidance_scale,
                    img2img_strength=args.img2img_strength,
                    timesteps_prior=100,
                    seed=seed,
                    retrieve=retrieve,
                    plotting=plotting,
                    img_variations=img_variations,
                    verbose=verbose,
                )
                
                brain_recons = brain_recons[:, laion_best_picks.astype(np.int8)]
            
            # 收集所有重建结果
            # if all_brain_recons is None:
            #     all_brain_recons = brain_recons
            #     all_images = img
            # else:
            #     all_brain_recons = torch.vstack((all_brain_recons, brain_recons.detach().cpu()))
            #     all_images = torch.vstack((all_images, img))
            all_brain_recons.append(brain_recons.detach().cpu())
            all_images.append(img.detach().cpu())
        
        if val_i >= np.max(ind_include):
            break
    
    # 保存结果
    all_brain_recons = torch.cat(all_brain_recons, dim=0)
    print(all_brain_recons.shape)
    all_images = torch.cat(all_images, dim=0)
    # all_brain_recons = all_brain_recons.view(-1, 3, imsize, imsize)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    if saving:
        torch.save(all_images, 'all_images.pt')
        recon_filename = f'{args.model_name}_recons_img2img{args.img2img_strength}_{args.recons_per_sample}_{args.subj}samples.pt'
        torch.save(all_brain_recons, recon_filename)
        print(f'重建结果保存至: {recon_filename}')
    
    print("重建完成!")

if __name__ == "__main__":
    main()