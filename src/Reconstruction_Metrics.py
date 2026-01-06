#!/usr/bin/env python3
"""
评估fMRI图像重建模型的脚本。
计算并输出PixCorr、SSIM及多种神经网络特征的2-way identification分数。
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from datetime import datetime
import argparse
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import (
    alexnet, 
    AlexNet_Weights, 
    inception_v3, 
    Inception_V3_Weights,
    efficientnet_b1,
    EfficientNet_B1_Weights
)
import clip

# 初始化设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 导入项目工具函数 (假设utils.py在同一目录)
import utils

# 设置随机种子
seed = 42
utils.seed_everything(seed=seed)


# ==================== 配置参数 ====================
def parse_args():
    parser = argparse.ArgumentParser(description="fMRI图像重建模型评估配置")
    parser.add_argument(
        "--recon_path", 
        type=str,
        required=True,
        help="重建/检索输出的文件路径 (e.g., 'prior_257_final_subj01_bimixco_softclip_byol_brain_recons_full_img2img0.85_16samples.pt')"
    )
    parser.add_argument(
        "--all_images_path", 
        type=str, 
        default="all_images.pt",
        help="真实图像数据的文件路径 (默认: 'all_images.pt')"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="输出结果CSV文件的路径 (默认: 使用recon_path基础名，仅更改扩展名)"
    )
    parser.add_argument(
        "--imsize",
        type=int,
        default=256,
        help="用于显示的图像大小 (默认: 256)"
    )
    parser.add_argument(
        "--sample_indices",
        type=str,
        default="112,119,101,44,159,22,173,174,175,189,981,243,249,255,265",
        help="用于可视化的样本索引，以逗号分隔 (默认: '112,119,101,44,159,22,173,174,175,189,981,243,249,255,265')"
    )
    return parser.parse_args()


# ==================== 核心评估函数 ====================
@torch.no_grad()
def two_way_identification(all_brain_recons, all_images, model, preprocess, feature_layer=None, return_avg=True):
    """
    计算2-way identification准确率。
    
    参数:
        all_brain_recons: 重建图像张量
        all_images: 真实图像张量
        model: 用于提取特征的模型
        preprocess: 图像预处理函数
        feature_layer: 要提取的特征层名称
        return_avg: 是否返回平均准确率
    
    返回:
        如果return_avg为True，返回平均准确率；否则返回成功计数和总比较数
    """
    preds = model(torch.stack([preprocess(recon) for recon in all_brain_recons], dim=0).to(device))
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
    
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    r = np.corrcoef(reals, preds)
    r = r[:len(all_images), len(all_images):]
    congruents = np.diag(r)
    success = r < congruents
    success_cnt = np.sum(success, 0)

    if return_avg:
        perf = np.mean(success_cnt) / (len(all_images) - 1)
        return perf
    else:
        return success_cnt, len(all_images) - 1


def visualize_reconstructions(all_images, all_brain_recons, indices, imsize=256, nrow=10):
    """
    可视化重建图像与真实图像的对比。
    
    参数:
        all_images: 真实图像张量
        all_brain_recons: 重建图像张量
        indices: 要显示的样本索引
        imsize: 显示图像大小
        nrow: 网格显示的行数
    """
    # 调整图像大小
    resize_transform = transforms.Resize((imsize, imsize))
    all_images_resized = resize_transform(all_images)
    all_brain_recons_resized = resize_transform(all_brain_recons)
    
    # 创建交错排列的图像网格 (真实, 重建, 真实, 重建, ...)
    all_interleaved = torch.zeros(len(indices) * 2, 3, imsize, imsize)
    
    for i, idx in enumerate(indices):
        all_interleaved[2*i] = all_images_resized[idx]
        all_interleaved[2*i + 1] = all_brain_recons_resized[idx]
    
    # 创建并显示网格
    plt.rcParams["savefig.bbox"] = 'tight'
    grid = make_grid(all_interleaved, nrow=nrow, padding=2)
    
    # 将张量转换为图像并显示
    grid_img = transforms.ToPILImage()(grid)
    
    fig, ax = plt.subplots(figsize=(20, 16))
    ax.imshow(np.asarray(grid_img))
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()
    
    return fig


# ==================== 主函数 ====================
def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置输出CSV文件路径
    if args.output_csv is None:
        args.output_csv = os.path.splitext(args.recon_path)[0] + '.csv'
    
    # 解析样本索引
    sample_indices = list(map(int, args.sample_indices.split(',')))
    
    print(f"加载重建图像: {args.recon_path}")
    print(f"加载真实图像: {args.all_images_path}")
    print(f"样本索引: {sample_indices}")
    
    # 加载数据
    all_brain_recons = torch.load(args.recon_path)
    all_images = torch.load(args.all_images_path)
    
    print(f"真实图像形状: {all_images.shape}")
    print(f"重建图像形状: {all_brain_recons.shape}")
    
    # 移动数据到设备并归一化
    all_images = all_images.to(device)
    all_brain_recons = all_brain_recons.to(device).to(all_images.dtype).clamp(0, 1)
    
    # ==================== 可视化重建结果 ====================
    print("\n" + "="*60)
    print("生成重建结果可视化...")
    fig = visualize_reconstructions(
        all_images, 
        all_brain_recons, 
        sample_indices, 
        imsize=args.imsize
    )
    
    # ==================== 计算PixCorr ====================
    print("\n" + "="*60)
    print("计算PixCorr...")
    
    preprocess_pixcorr = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
    ])
    
    # 展平图像
    all_images_flattened = preprocess_pixcorr(all_images).reshape(len(all_images), -1).cpu()
    all_brain_recons_flattened = preprocess_pixcorr(all_brain_recons).view(len(all_brain_recons), -1).cpu()
    
    print(f"展平后真实图像形状: {all_images_flattened.shape}")
    print(f"展平后重建图像形状: {all_brain_recons_flattened.shape}")
    
    corrsum = 0
    for i in tqdm(range(982), desc="计算像素相关性"):
        corrsum += np.corrcoef(all_images_flattened[i], all_brain_recons_flattened[i])[0][1]
    
    pixcorr = corrsum / 982
    print(f"PixCorr: {pixcorr:.6f}")
    
    # ==================== 计算SSIM ====================
    print("\n" + "="*60)
    print("计算SSIM...")
    
    preprocess_ssim = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
    ])
    
    # 转换为灰度图像
    img_gray = rgb2gray(preprocess_ssim(all_images).permute((0, 2, 3, 1)).cpu())
    recon_gray = rgb2gray(preprocess_ssim(all_brain_recons).permute((0, 2, 3, 1)).cpu())
    print("图像已转换为灰度，正在计算SSIM...")
    
    ssim_scores = []
    for im, rec in tqdm(zip(img_gray, recon_gray), total=len(all_images), desc="计算SSIM"):
        ssim_scores.append(ssim(
            rec, im, 
            channel_axis=None if rec.ndim == 2 else -1,
            gaussian_weights=True, 
            sigma=1.5, 
            use_sample_covariance=False, 
            data_range=1.0
        ))
    
    ssim_value = np.mean(ssim_scores)
    print(f"SSIM: {ssim_value:.6f}")
    
    # ==================== 计算AlexNet特征 ====================
    print("\n" + "="*60)
    print("计算AlexNet特征...")
    
    alex_weights = AlexNet_Weights.IMAGENET1K_V1
    alex_model = create_feature_extractor(
        alexnet(weights=alex_weights), 
        return_nodes=['features.4', 'features.11']
    ).to(device)
    alex_model.eval().requires_grad_(False)
    
    preprocess_alex = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # AlexNet第二层
    print("\n--- AlexNet(2) ---")
    alexnet2 = two_way_identification(
        all_brain_recons.to(device).float(), 
        all_images, 
        alex_model, 
        preprocess_alex, 
        'features.4'
    )
    print(f"2-way Percent Correct: {alexnet2:.4f}")
    
    # AlexNet第五层
    print("\n--- AlexNet(5) ---")
    alexnet5 = two_way_identification(
        all_brain_recons.to(device).float(), 
        all_images, 
        alex_model, 
        preprocess_alex, 
        'features.11'
    )
    print(f"2-way Percent Correct: {alexnet5:.4f}")
    
    # ==================== 计算InceptionV3特征 ====================
    print("\n" + "="*60)
    print("计算InceptionV3特征...")
    
    inception_weights = Inception_V3_Weights.DEFAULT
    inception_model = create_feature_extractor(
        inception_v3(weights=inception_weights), 
        return_nodes=['avgpool']
    ).to(device)
    inception_model.eval().requires_grad_(False)
    
    preprocess_inception = transforms.Compose([
        transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    inception = two_way_identification(
        all_brain_recons, 
        all_images,
        inception_model, 
        preprocess_inception, 
        'avgpool'
    )
    print(f"2-way Percent Correct: {inception:.4f}")
    
    # ==================== 计算CLIP特征 ====================
    print("\n" + "="*60)
    print("计算CLIP特征...")
    
    clip_model, _ = clip.load("ViT-L/14", device=device)
    clip_model.eval().requires_grad_(False)
    
    preprocess_clip = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])
    
    clip_score = two_way_identification(
        all_brain_recons, 
        all_images,
        clip_model.encode_image, 
        preprocess_clip, 
        None
    )
    print(f"2-way Percent Correct: {clip_score:.4f}")
    
    # ==================== 计算EfficientNet特征 ====================
    print("\n" + "="*60)
    print("计算EfficientNet特征...")
    
    eff_weights = EfficientNet_B1_Weights.DEFAULT
    eff_model = create_feature_extractor(
        efficientnet_b1(weights=eff_weights), 
        return_nodes=['avgpool']
    ).to(device)
    eff_model.eval().requires_grad_(False)
    
    preprocess_eff = transforms.Compose([
        transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    gt_eff = eff_model(preprocess_eff(all_images))['avgpool']
    gt_eff = gt_eff.reshape(len(gt_eff), -1).cpu().numpy()
    
    fake_eff = eff_model(preprocess_eff(all_brain_recons))['avgpool']
    fake_eff = fake_eff.reshape(len(fake_eff), -1).cpu().numpy()
    
    effnet = np.array([sp.spatial.distance.correlation(gt_eff[i], fake_eff[i]) 
                       for i in range(len(gt_eff))]).mean()
    print(f"EfficientNet距离: {effnet:.6f}")
    
    # ==================== 计算SwAV特征 ====================
    print("\n" + "="*60)
    print("计算SwAV特征...")
    
    # 注意: 需要网络连接以下载SwAV模型
    swav_model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    swav_model = create_feature_extractor(
        swav_model, 
        return_nodes=['avgpool']
    ).to(device)
    swav_model.eval().requires_grad_(False)
    
    preprocess_swav = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    gt_swav = swav_model(preprocess_swav(all_images))['avgpool']
    gt_swav = gt_swav.reshape(len(gt_swav), -1).cpu().numpy()
    
    fake_swav = swav_model(preprocess_swav(all_brain_recons))['avgpool']
    fake_swav = fake_swav.reshape(len(fake_swav), -1).cpu().numpy()
    
    swav = np.array([sp.spatial.distance.correlation(gt_swav[i], fake_swav[i]) 
                     for i in range(len(gt_swav))]).mean()
    print(f"SwAV距离: {swav:.6f}")
    
    # ==================== 汇总结果 ====================
    print("\n" + "="*60)
    print("评估结果汇总:")
    print("="*60)
    
    # 创建结果数据框
    data = {
        "Metric": ["PixCorr", "SSIM", "AlexNet(2)", "AlexNet(5)", "InceptionV3", "CLIP", "EffNet-B", "SwAV"],
        "Value": [pixcorr, ssim_value, alexnet2, alexnet5, inception, clip_score, effnet, swav],
    }
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    # 保存结果到CSV
    df.to_csv(args.output_csv, sep='\t', index=False)
    print(f"\n结果已保存到: {args.output_csv}")
    
    print("\n评估完成!")


if __name__ == "__main__":
    main()