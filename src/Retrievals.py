#!/usr/bin/env python3
"""
NSD fMRI图像检索与评估脚本
清理自Jupyter Notebook，移除所有交互式调试代码
支持LAION-5B检索和图像/脑信号双向检索评估
"""

import argparse
import os
import sys
import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import webdataset as wds
import PIL

import utils
from models import (
    Clipper, 
    BrainNetwork, 
    BrainDiffusionPrior, 
    BrainDiffusionPriorOld, 
    Voxel2StableDiffusionModel, 
    VersatileDiffusionPriorNetwork
)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="NSD fMRI图像检索模型配置")
    
    parser.add_argument(
        "--model_name", 
        type=str,
        required=True,
        help="257x768模型名称，用于除LAION-5B检索外的所有任务",
    )
    parser.add_argument(
        "--model_name2", 
        type=str,
        required=True,
        help="1x768模型名称，专门用于LAION-5B检索",
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="/fsx/proj-medarc/fmri/natural-scenes-dataset",
        help="NSD数据存储路径 (参见README)",
    )
    parser.add_argument(
        "--subj",
        type=int, 
        default=1, 
        choices=[1, 2, 5, 7],
        help="受试者编号",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="retrieval",
        choices=["retrieval", "laion", "both"],
        help="运行模式: retrieval(检索评估), laion(LAION检索), both(两者)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=300,
        help="检索评估的批处理大小",
    )
    parser.add_argument(
        "--num_retrieved",
        type=int,
        default=16,
        help="LAION检索中返回的图像数量",
    )
    
    return parser.parse_args()


def setup_environment(seed=42):
    """设置计算环境和随机种子"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    utils.seed_everything(seed=seed)
    
    return device


def get_voxel_count(subject_id):
    """根据受试者ID获取体素数量"""
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
    
    num_voxels = voxel_counts.get(subject_id, 15724)
    print(f"受试者: {subject_id}, 体素数量: {num_voxels}")
    
    return num_voxels


def create_data_loader(data_path, subject_id, batch_size=1, resampled=False, num_loops=1):
    """创建WebDataset数据加载器"""
    val_url = f"{data_path}/webdataset_avg_split/test/test_subj0{subject_id}_" + "{0..1}.tar"
    meta_url = f"{data_path}/webdataset_avg_split/metadata_subj0{subject_id}.json"
    voxels_key = 'nsdgeneral.npy'
    
    val_data = wds.WebDataset(val_url, resampled=resampled)\
        .decode("torch")\
        .rename(
            images="jpg;png", 
            voxels=voxels_key, 
            trial="trial.npy", 
            coco="coco73k.npy", 
            reps="num_uniques.npy"
        )\
        .to_tuple("voxels", "images", "coco")\
        .batched(batch_size, partial=False)
    
    if resampled:
        val_data = val_data.with_epoch(num_loops)
    
    return torch.utils.data.DataLoader(val_data, batch_size=None, shuffle=False)


def load_diffusion_prior_model(model_name, num_voxels, device, out_dim=768, use_cls_model=False):
    """加载扩散先验模型"""
    if use_cls_model:
        # 加载CLS模型 (1x768)
        out_dim = 768
        voxel2clip_kwargs = dict(in_dim=num_voxels, out_dim=out_dim)
        voxel2clip_cls = BrainNetwork(**voxel2clip_kwargs)
        voxel2clip_cls.requires_grad_(False)
        voxel2clip_cls.eval()
        
        diffusion_prior = BrainDiffusionPriorOld.from_pretrained(
            dict(),  # DiffusionPriorNetwork参数
            dict(
                condition_on_text_encodings=False,
                timesteps=1000,
                voxel2clip=voxel2clip_cls,
            ),
            voxel2clip_path=None,
        )
    else:
        # 加载标准模型 (257x768)
        clip_extractor = Clipper("ViT-L/14", hidden_state=True, norm_embs=True, device=device)
        voxel2clip_kwargs = dict(in_dim=num_voxels, out_dim=257*768)
        voxel2clip = BrainNetwork(**voxel2clip_kwargs)
        voxel2clip.requires_grad_(False)
        voxel2clip.eval()
        
        prior_network = VersatileDiffusionPriorNetwork(
            dim=out_dim,
            depth=6,
            dim_head=64,
            heads=12,
            causal=False,
            learned_query_mode="pos_emb"
        ).to(device)
        
        diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=out_dim,
            condition_on_text_encodings=False,
            timesteps=100,
            cond_drop_prob=0.2,
            image_embed_scale=None,
            voxel2clip=voxel2clip,
        ).to(device)
    
    # 加载检查点
    outdir = f'../train_logs/{model_name}'
    ckpt_path = os.path.join(outdir, 'last.pth')
    
    print(f"加载检查点: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    print(f"训练轮数: {checkpoint['epoch']}")
    
    diffusion_prior.load_state_dict(state_dict, strict=False)
    diffusion_prior.eval().to(device)
    
    return diffusion_prior


def perform_laion_retrieval(args, device, num_voxels, num_val=982):
    """执行LAION-5B检索"""
    print("\n" + "="*60)
    print("开始LAION-5B检索")
    print("="*60)
    
    # 创建数据加载器
    val_dl = create_data_loader(args.data_path, args.subj, batch_size=1, resampled=False)
    
    # 加载模型
    diffusion_prior = load_diffusion_prior_model(args.model_name, num_voxels, device)
    diffusion_prior_cls = load_diffusion_prior_model(args.model_name2, num_voxels, device, use_cls_model=True)
    
    clip_extractor = Clipper("ViT-L/14", hidden_state=True, norm_embs=True, device=device)
    
    # 执行检索
    retrieve = True
    plotting = False
    saving = True
    verbose = False
    imsize = 512
    
    all_brain_recons = []
    all_images = []
    ind_include = np.arange(num_val)
    
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for val_i, (voxel, img, coco) in enumerate(tqdm(val_dl, total=len(ind_include))):
        if val_i < np.min(ind_include):
            continue
            
        voxel = torch.mean(voxel, axis=1).to(device)
        
        with torch.no_grad():
            grid, brain_recons, laion_best_picks, recon_img = utils.reconstruction(
                img, voxel,
                clip_extractor,
                voxel2clip_cls=diffusion_prior_cls.voxel2clip,
                diffusion_priors=[diffusion_prior],
                text_token=None,
                n_samples_save=1,
                recons_per_sample=0,
                seed=42,
                retrieve=retrieve,
                plotting=plotting,
                verbose=verbose,
                num_retrieved=args.num_retrieved,
            )
            
            brain_recons = brain_recons[laion_best_picks.astype(np.int8)]
            all_brain_recons.append(brain_recons)
            all_images.append(img)
            
        if val_i >= np.max(ind_include):
            break
    
    # 合并结果
    all_brain_recons = torch.cat(all_brain_recons, dim=0)
    all_images = torch.cat(all_images, dim=0)
    all_brain_recons = all_brain_recons.view(-1, 3, imsize, imsize)
    
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"重建图像形状: {all_brain_recons.shape}")
    
    # 保存结果
    if saving:
        torch.save(all_images, 'all_images.pt')
        output_file = f'{args.model_name}_laion_retrievals_top{args.num_retrieved}.pt'
        torch.save(all_brain_recons, output_file)
        print(f'检索结果保存至: {output_file}')
    
    print("LAION-5B检索完成!")
    
    return all_brain_recons, all_images


def perform_retrieval_evaluation(args, device, num_voxels, batch_size=300, val_loops=30):
    """执行图像/脑信号双向检索评估"""
    print("\n" + "="*60)
    print("开始图像/脑信号双向检索评估")
    print("="*60)
    
    # 创建数据加载器
    val_dl = create_data_loader(
        args.data_path, args.subj, 
        batch_size=batch_size, 
        resampled=True, 
        num_loops=val_loops
    )
    
    # 加载模型
    diffusion_prior = load_diffusion_prior_model(args.model_name, num_voxels, device)
    clip_extractor = Clipper("ViT-L/14", hidden_state=True, norm_embs=True, device=device)
    
    # 执行检索评估
    percent_correct_fwds, percent_correct_bwds = [], []
    
    for val_i, (voxel, img, coco) in enumerate(tqdm(val_dl, total=val_loops)):
        with torch.no_grad():
            voxel = torch.mean(voxel, axis=1).to(device)
            
            # 提取CLIP图像嵌入
            emb = clip_extractor.embed_image(img.to(device)).float()
            
            # 提取CLIP脑信号嵌入
            _, emb_ = diffusion_prior.voxel2clip(voxel.float())
            
            # 展平并归一化
            emb = emb.reshape(len(emb), -1)
            emb_ = emb_.reshape(len(emb_), -1)
            
            emb = nn.functional.normalize(emb, dim=-1)
            emb_ = nn.functional.normalize(emb_, dim=-1)
            
            labels = torch.arange(len(emb)).to(device)
            
            # 计算双向相似度
            bwd_sim = utils.batchwise_cosine_similarity(emb, emb_)  # 图像 -> 脑信号
            fwd_sim = utils.batchwise_cosine_similarity(emb_, emb)  # 脑信号 -> 图像
            
            # 计算准确率
            percent_correct_fwds.append(utils.topk(fwd_sim, labels, k=1).item())
            percent_correct_bwds.append(utils.topk(bwd_sim, labels, k=1).item())
            
            if val_i == 0:
                print(f"第一次循环结果: 前向 {percent_correct_fwds[-1]:.4f}, 反向 {percent_correct_bwds[-1]:.4f}")
    
    # 计算统计结果
    percent_correct_fwd = np.mean(percent_correct_fwds)
    fwd_sd = np.std(percent_correct_fwds) / np.sqrt(len(percent_correct_fwds))
    fwd_ci = stats.norm.interval(0.95, loc=percent_correct_fwd, scale=fwd_sd)
    
    percent_correct_bwd = np.mean(percent_correct_bwds)
    bwd_sd = np.std(percent_correct_bwds) / np.sqrt(len(percent_correct_bwds))
    bwd_ci = stats.norm.interval(0.95, loc=percent_correct_bwd, scale=bwd_sd)
    
    print("\n" + "="*60)
    print("检索评估结果")
    print("="*60)
    print(f"前向检索准确率 (脑信号->图像): {percent_correct_fwd:.4f} 95% CI: [{fwd_ci[0]:.4f}, {fwd_ci[1]:.4f}]")
    print(f"反向检索准确率 (图像->脑信号): {percent_correct_bwd:.4f} 95% CI: [{bwd_ci[0]:.4f}, {bwd_ci[1]:.4f}]")
    
    # 注: 各受试者的典型结果:
    # SUBJ 1: 前向 0.9718, 反向 0.9468
    # SUBJ 2: 前向 0.9710, 反向 0.9386
    # SUBJ 5: 前向 0.9067, 反向 0.8573
    # SUBJ 7: 前向 0.8941, 反向 0.8582
    fwd_sim_numpy = fwd_sim.cpu().numpy()
    bwd_sim_numpy = bwd_sim.cpu().numpy()
    return percent_correct_fwd, percent_correct_bwd, fwd_sim_numpy, bwd_sim_numpy


# def visualize_retrieval_results(img, fwd_sim, bwd_sim, num_examples=4):
#     """可视化检索结果"""
#     # 前向检索可视化 (脑信号->图像)
#     print("\n前向检索可视化: 给定脑信号嵌入，找到正确的图像嵌入")
#     fig, ax = plt.subplots(nrows=num_examples, ncols=6, figsize=(11, 3*num_examples))
    
#     for trial in range(num_examples):
#         ax[trial, 0].imshow(utils.torch_to_Image(img[trial]))
#         ax[trial, 0].set_title("原始图像")
#         ax[trial, 0].axis("off")
        
#         for attempt in range(5):
#             which = np.flip(np.argsort(fwd_sim[trial]))[attempt]
#             ax[trial, attempt+1].imshow(utils.torch_to_Image(img[which]))
#             ax[trial, attempt+1].set_title(f"Top {attempt+1}")
#             ax[trial, attempt+1].axis("off")
    
#     fig.tight_layout()
#     plt.show()
    
#     # 反向检索可视化 (图像->脑信号)
#     print("\n反向检索可视化: 给定图像嵌入，找到正确的脑信号嵌入")
#     fig, ax = plt.subplots(nrows=num_examples, ncols=6, figsize=(11, 3*num_examples))
    
#     for trial in range(num_examples):
#         ax[trial, 0].imshow(utils.torch_to_Image(img[trial]))
#         ax[trial, 0].set_title("原始图像")
#         ax[trial, 0].axis("off")
        
#         for attempt in range(5):
#             which = np.flip(np.argsort(bwd_sim[trial]))[attempt]
#             ax[trial, attempt+1].imshow(utils.torch_to_Image(img[which]))
#             ax[trial, attempt+1].set_title(f"Top {attempt+1}")
#             ax[trial, attempt+1].axis("off")
    
#     fig.tight_layout()
#     plt.show()


def visualize_retrieval_results_save(
    img,
    fwd_sim,
    bwd_sim,
    num_examples=4,
    save_dir="../retrieval_vis/"
):
    os.makedirs(save_dir, exist_ok=True)

    # ===============================
    # 前向检索：脑信号 → 图像
    # ===============================
    print("\n前向检索可视化: 给定脑信号嵌入，找到正确的图像嵌入")

    fig, ax = plt.subplots(
        nrows=num_examples,
        ncols=6,
        figsize=(11, 3 * num_examples)
    )

    for trial in range(num_examples):
        ax[trial, 0].imshow(utils.torch_to_Image(img[trial]))
        ax[trial, 0].set_title("GT")
        ax[trial, 0].axis("off")

        for attempt in range(5):
            which = np.flip(np.argsort(fwd_sim[trial]))[attempt]
            ax[trial, attempt + 1].imshow(
                utils.torch_to_Image(img[which])
            )
            ax[trial, attempt + 1].set_title(f"Top-{attempt+1}")
            ax[trial, attempt + 1].axis("off")

    fig.tight_layout()
    fwd_path = os.path.join(save_dir, "retrieval_forward.png")
    fig.savefig(fwd_path, dpi=200)
    plt.close(fig)

    print(f"Saved forward retrieval visualization to {fwd_path}")

    # ===============================
    # 反向检索：图像 → 脑信号
    # ===============================
    print("\n反向检索可视化: 给定图像嵌入，找到正确的脑信号嵌入")

    fig, ax = plt.subplots(
        nrows=num_examples,
        ncols=6,
        figsize=(11, 3 * num_examples)
    )

    for trial in range(num_examples):
        ax[trial, 0].imshow(utils.torch_to_Image(img[trial]))
        ax[trial, 0].set_title("GT")
        ax[trial, 0].axis("off")

        for attempt in range(5):
            which = np.flip(np.argsort(bwd_sim[trial]))[attempt]
            ax[trial, attempt + 1].imshow(
                utils.torch_to_Image(img[which])
            )
            ax[trial, attempt + 1].set_title(f"Top-{attempt+1}")
            ax[trial, attempt + 1].axis("off")

    fig.tight_layout()
    bwd_path = os.path.join(save_dir, "retrieval_backward.png")
    fig.savefig(bwd_path, dpi=200)
    plt.close(fig)

    print(f"Saved backward retrieval visualization to {bwd_path}")



# def visualize_zebra_examples(args, data_path, subject_id, num_voxels):
#     """可视化斑马示例 (完整数据集检索)"""
#     print("\n" + "="*60)
#     print("斑马示例: 使用完整数据集(982样本)进行检索")
#     print("="*60)
    
#     # 切换到CPU以避免内存溢出
#     device_cpu = torch.device('cpu')
    
#     # 创建数据加载器 (加载所有982个样本)
#     val_dl = create_data_loader(data_path, subject_id, batch_size=982, resampled=False)
    
#     # 加载模型到CPU
#     clip_extractor = Clipper("ViT-L/14", hidden_state=True, norm_embs=True, device=device_cpu)
#     diffusion_prior = load_diffusion_prior_model(args.model_name, num_voxels, device_cpu)
    
#     # 处理数据
#     for val_i, (voxel, img_input, coco) in enumerate(tqdm(val_dl, total=1)):
#         with torch.no_grad():
#             voxel = torch.mean(voxel, axis=1).to(device_cpu)
            
#             # 提取嵌入
#             emb = clip_extractor.embed_image(img_input.to(device_cpu)).float()
#             _, emb_ = diffusion_prior.voxel2clip(voxel.float())
            
#             # 展平并归一化
#             emb = emb.reshape(len(emb), -1)
#             emb_ = emb_.reshape(len(emb_), -1)
            
#             emb = nn.functional.normalize(emb, dim=-1)
#             emb_ = nn.functional.normalize(emb_, dim=-1)
            
#             labels = torch.arange(len(emb)).to(device_cpu)
            
#             # 计算相似度
#             bwd_sim = utils.batchwise_cosine_similarity(emb, emb_)
#             fwd_sim = utils.batchwise_cosine_similarity(emb_, emb)
            
#             # 计算准确率
#             percent_correct_fwd = utils.topk(fwd_sim, labels, k=1)
#             percent_correct_bwd = utils.topk(bwd_sim, labels, k=1)
            
#             print(f"完整数据集检索准确率: 前向 {percent_correct_fwd.item():.4f}, 反向 {percent_correct_bwd.item():.4f}")
            
#             fwd_sim_np = fwd_sim.numpy()
            
#             # 斑马示例可视化
#             zebra_indices = [891, 892, 893, 863, 833, 652, 516, 512, 498, 451, 331, 192, 129, 66]
#             print(f"斑马图像数量: {len(zebra_indices)}")
            
#             fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(11, 5))
            
#             for trial, t in enumerate(zebra_indices[:2]):
#                 ax[trial, 0].imshow(utils.torch_to_Image(img_input[t]))
#                 ax[trial, 0].set_title("原始图像")
#                 ax[trial, 0].axis("off")
                
#                 for attempt in range(5):
#                     which = np.flip(np.argsort(fwd_sim_np[t]))[attempt]
#                     ax[trial, attempt+1].imshow(utils.torch_to_Image(img_input[which]))
#                     ax[trial, attempt+1].set_title(f"Top {attempt+1}")
#                     ax[trial, attempt+1].axis("off")
            
#             fig.tight_layout()
#             plt.show()
            
#             break  # 只处理第一批数据

def visualize_zebra_examples_save(
    args,
    data_path,
    subject_id,
    num_voxels,
    save_dir="../zebra_vis/"
):
    """可视化斑马示例 (完整数据集检索) - 保存为 PNG"""
    print("\n" + "=" * 60)
    print("斑马示例: 使用完整数据集(982样本)进行检索")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)

    # 切换到CPU（这是论文代码的关键点）
    device_cpu = torch.device("cpu")

    # 加载所有 982 张
    val_dl = create_data_loader(
        data_path,
        subject_id,
        batch_size=982,
        resampled=False
    )

    # 模型加载到 CPU
    clip_extractor = Clipper(
        "ViT-L/14",
        hidden_state=True,
        norm_embs=True,
        device=device_cpu
    )
    diffusion_prior = load_diffusion_prior_model(
        args.model_name,
        num_voxels,
        device_cpu
    )

    for val_i, (voxel, img_input, coco) in enumerate(tqdm(val_dl, total=1)):
        with torch.no_grad():
            voxel = torch.mean(voxel, axis=1).to(device_cpu)

            # CLIP image embedding
            emb = clip_extractor.embed_image(
                img_input.to(device_cpu)
            ).float()

            # brain → CLIP embedding
            _, emb_ = diffusion_prior.voxel2clip(voxel.float())

            # flatten + normalize
            emb = nn.functional.normalize(
                emb.reshape(len(emb), -1), dim=-1
            )
            emb_ = nn.functional.normalize(
                emb_.reshape(len(emb_), -1), dim=-1
            )

            labels = torch.arange(len(emb)).to(device_cpu)

            # similarity
            bwd_sim = utils.batchwise_cosine_similarity(emb, emb_)
            fwd_sim = utils.batchwise_cosine_similarity(emb_, emb)

            # accuracy（只是打印，和论文一致）
            percent_correct_fwd = utils.topk(fwd_sim, labels, k=1)
            percent_correct_bwd = utils.topk(bwd_sim, labels, k=1)
            print(
                f"完整数据集检索准确率: "
                f"前向 {percent_correct_fwd.item():.4f}, "
                f"反向 {percent_correct_bwd.item():.4f}"
            )

            fwd_sim_np = fwd_sim.numpy()

            # zebra indices（论文里手选的）
            zebra_indices = [
                891, 892, 893, 863, 833, 652, 516,
                512, 498, 451, 331, 192, 129, 66
            ]
            print(f"斑马图像数量: {len(zebra_indices)}")

            # ===== 可视化：只画 2 行（和论文一致）=====
            fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(11, 5))

            for trial, t in enumerate(zebra_indices[:2]):
                ax[trial, 0].imshow(
                    utils.torch_to_Image(img_input[t])
                )
                ax[trial, 0].set_title("GT")
                ax[trial, 0].axis("off")

                for attempt in range(5):
                    which = np.flip(
                        np.argsort(fwd_sim_np[t])
                    )[attempt]
                    ax[trial, attempt + 1].imshow(
                        utils.torch_to_Image(img_input[which])
                    )
                    ax[trial, attempt + 1].set_title(
                        f"Top-{attempt+1}"
                    )
                    ax[trial, attempt + 1].axis("off")

            fig.tight_layout()

            save_path = os.path.join(
                save_dir,
                f"zebra_retrieval_subj{subject_id}.png"
            )
            fig.savefig(save_path, dpi=200)
            plt.close(fig)

            print(f"Saved zebra retrieval visualization to {save_path}")
            break  # 只处理一次（982 张一起）

def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 设置环境
    device = setup_environment()
    
    # 获取体素数量
    num_voxels = get_voxel_count(args.subj)
    
    # 根据模式执行相应任务
    if args.mode == "laion" or args.mode == "both":
        perform_laion_retrieval(args, device, num_voxels)
    
    if args.mode == "retrieval" or args.mode == "both":
        # 执行检索评估
        percent_correct_fwd, percent_correct_bwd, fwd_sim, bwd_sim = perform_retrieval_evaluation(
            args, device, num_voxels, batch_size=args.batch_size
        )
        
        # 可视化结果 (需要获取一些样本图像)
        val_dl = create_data_loader(args.data_path, args.subj, batch_size=args.batch_size, resampled=True)
        for val_i, (voxel, img, coco) in enumerate(val_dl):
            if val_i == 0:
                visualize_retrieval_results_save(img, fwd_sim, bwd_sim)
                break
        
        # 可视化斑马示例
        visualize_zebra_examples_save(args, args.data_path, args.subj, num_voxels)
    
    print("\n所有任务完成!")


if __name__ == "__main__":
    main()