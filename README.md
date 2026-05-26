# Simplified Stable Diffusion on MNIST

基于 Latent Diffusion 架构的简化版 Stable Diffusion，在 MNIST 数据集上训练。

## 项目结构

```
├── models/
│   ├── vae.py           # VAE: 28x28 图像 -> 4x4 latent
│   ├── unet.py          # U-Net: latent space 噪声预测
│   └── diffusion.py     # DDPM: 前向加噪 + 反向采样
├── train.py             # 训练脚本
├── sample.py            # 采样脚本
├── configs/
│   └── default.yaml     # 默认配置
├── notebooks/
│   ├── 01_vae_exploration.ipynb
│   ├── 02_diffusion_training.ipynb
│   └── 03_sampling.ipynb
└── utils/
    └── visualization.py # 可视化工具
```

## 阿里云 DSW 环境配置

### 1. 创建 DSW 实例

1. 登录 [阿里云 PAI-DSW](https://dsw-dev.data.aliyun.com/)
2. 创建实例，选择 **GPU** 实例（推荐 V100 或 A10）
3. 选择镜像：`Python 3.10 + CUDA 11.8 + PyTorch 2.0`

### 2. 上传项目

```bash
# 方式1: 在 DSW 终端中 clone
git clone <your-repo-url>
cd diffusion-model-practice

# 方式2: 通过 JupyterLab 上传整个文件夹
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 训练模型

```bash
# 阶段1: 训练 VAE (~5 分钟)
python train.py --stage vae --epochs-vae 30

# 阶段2: 训练扩散模型 (~20 分钟)
python train.py --stage diffusion --epochs-diff 100

# 或者一次性训练全部
python train.py --stage all
```

### 5. 生成图像

```bash
# 生成 16 张图片
python sample.py --num_samples 16

# 查看去噪过程
python sample.py --show_process
```

### 6. 使用 Notebook 交互学习

在 JupyterLab 中打开 `notebooks/` 目录下的 notebook 文件。

## 模型架构

### VAE (Variational Autoencoder)
- **Encoder**: 28x28 → 14x14 → 7x7 → 4x4, 输出 latent (64维)
- **Decoder**: latent → 4x4 → 7x7 → 14x14 → 28x28
- **损失**: 重建损失 (MSE) + KL 散度

### U-Net
- 3-level encoder-decoder with skip connections
- 时间步嵌入: Sinusoidal → MLP
- 操作在 4x4 latent space

### DDPM (Denoising Diffusion Probabilistic Model)
- 线性噪声调度: β_t ∈ [1e-4, 0.02]
- T = 1000 步
- 训练目标: 预测噪声 ε

## 训练时间预估 (单卡 V100)

| 阶段 | 时间 | 说明 |
|------|------|------|
| VAE | ~5 min | 30 epochs |
| Diffusion | ~20 min | 100 epochs |
| 采样 | ~1 min | 16 张图 |

## 参考文献

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., 2022)](https://arxiv.org/abs/2112.10752)
