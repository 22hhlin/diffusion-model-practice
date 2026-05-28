"""
EmoSet LoRA 微调流水线

三步流程：
    1. collect — 把嵌套文件夹中的图片收集到统一目录，按 000000.jpg, 000001.jpg 编号
    2. caption — 用 BLIP2 模型自动为每张图片生成文字描述
    3. train   — 用生成的描述和图片训练 LoRA

数据流：
    原始图片（嵌套文件夹）→ 扁平目录 + 编号 → metadata.jsonl（图片路径+描述）→ LoRA 训练

Usage:
    # 跑完整流程（收集 + 题注 + 训练）
    python run_emoset_pipeline.py --image_dir /path/to/origin_image

    # 分步执行
    python run_emoset_pipeline.py --image_dir /path/to/origin_image --step collect
    python run_emoset_pipeline.py --image_dir /path/to/origin_image --step caption
    python run_emoset_pipeline.py --image_dir /path/to/origin_image --step train
"""
import argparse
import os
import json
import shutil
import torch
from PIL import Image
from tqdm import tqdm

# 支持的图片格式
VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


# ============================================================
# 步骤 1：收集图片
# ============================================================
def collect_images(image_dir, output_dir):
    """
    递归扫描 image_dir 下所有子文件夹，把图片收集到 output_dir，
    并按 000000.jpg, 000001.jpg, ... 顺序编号。

    为什么需要这步？
      原始数据集（如 EmoSet）按情感类别分子文件夹存放：
        origin_image/joy/001.jpg
        origin_image/joy/002.jpg
        origin_image/sadness/001.jpg
        ...
      训练时需要一个扁平目录 + 统一编号，方便 DataLoader 读取。
    """
    if not os.path.isdir(image_dir):
        print(f"Error: {image_dir} not found")
        return 0

    os.makedirs(output_dir, exist_ok=True)

    # 如果已经收集过（目录里有图片），跳过避免重复
    existing = [f for f in os.listdir(output_dir) if os.path.splitext(f)[1].lower() in VALID_EXT]
    if existing:
        print(f"Already collected {len(existing)} images in {output_dir}, skipping...")
        return len(existing)

    # os.walk 递归遍历所有子目录
    # root=当前目录, dirs=子目录列表, files=文件列表
    all_images = []
    for root, dirs, files in os.walk(image_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in VALID_EXT:
                all_images.append(os.path.join(root, f))

    # 排序保证编号顺序一致（每次运行结果相同）
    all_images.sort()
    print(f"Found {len(all_images)} images in {image_dir}")

    # 复制并重命名为 000000.jpg, 000001.jpg, ...
    for i, src_path in enumerate(tqdm(all_images, desc="Collecting")):
        ext = os.path.splitext(src_path)[1].lower()
        dst_path = os.path.join(output_dir, f'{i:06d}{ext}')
        shutil.copy2(src_path, dst_path)

    print(f"Collected {len(all_images)} images to {output_dir}")
    return len(all_images)


# ============================================================
# 步骤 2：BLIP2 自动题注
# ============================================================
def generate_captions(image_dir, output_meta, max_images=None):
    """
    用 BLIP2 模型为每张图片生成文字描述。

    BLIP2 是一个多模态大语言模型，能"看图说话"：
      输入：一张图片
      输出：一段文字描述，如 "a sunset over the ocean with orange clouds"

    为什么用 BLIP2 而不是人工标注？
      15532 张图片人工标注成本太高，BLIP2 可以自动、快速、质量尚可地完成。

    输出格式：metadata.jsonl，每行一个 JSON：
      {"file_name": "/abs/path/000000.jpg", "text": "a sunset over the ocean"}
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sd_inference'))
    from utils import get_model_path
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    # 扫描目录中的图片
    all_files = os.listdir(image_dir)
    print(f"Total files in {image_dir}: {len(all_files)}")
    images = sorted([f for f in all_files
                     if os.path.splitext(f)[1].lower() in VALID_EXT])
    print(f"Valid image files: {len(images)}")

    if not images:
        print(f"Error: No images found in {image_dir}")
        return

    # 可选：只处理前 max_images 张（调试用）
    if max_images:
        images = images[:max_images]
        print(f"Limited to {max_images} images")

    print(f"Captioning {len(images)} images...")

    # 加载 BLIP2 模型
    # BLIP2 架构：ViT（看图）→ Q-Former（桥梁）→ OPT-2.7B（生成文字）
    # 优先用本地缓存的模型，避免重复下载
    local_blip2 = '/mnt/workspace/models/modelscope/models/Salesforce/blip2-opt-2.7b'
    if os.path.isdir(local_blip2):
        blip2_path = local_blip2
        print(f"Using local BLIP2: {blip2_path}")
    else:
        blip2_path = get_model_path('Salesforce/blip2-opt-2.7b')
    print(f"Loading BLIP2 from: {blip2_path}")

    # processor：图片预处理（resize、normalize）+ 文字 tokenize
    # model：BLIP2 生成模型，float16 半精度节省显存
    processor = Blip2Processor.from_pretrained(blip2_path)
    model = Blip2ForConditionalGeneration.from_pretrained(
        blip2_path, torch_dtype=torch.float16
    ).to('cuda')
    print("BLIP2 loaded successfully!")

    metadata = []
    batch_size = 4  # 批量处理，比逐张快

    # 逐批处理图片
    print("Generating captions with BLIP2...")
    for i in tqdm(range(0, len(images), batch_size), desc="BLIP2"):
        batch_files = images[i:i+batch_size]

        # 加载并预处理一批图片
        batch_imgs = []
        for f in batch_files:
            img = Image.open(os.path.join(image_dir, f)).convert('RGB')
            batch_imgs.append(img.resize((512, 512)))  # BLIP2 要求固定尺寸

        # processor 把图片转为模型输入 tensor
        inputs = processor(images=batch_imgs, return_tensors="pt").to('cuda')

        # generate() 自回归生成文字，max_new_tokens=50 限制最多 50 个词
        ids = model.generate(**inputs, max_new_tokens=50)

        # 把 token ID 解码回文字
        captions = processor.batch_decode(ids, skip_special_tokens=True)

        # 保存每张图片的路径和描述
        for f, cap in zip(batch_files, captions):
            metadata.append({
                'file_name': os.path.abspath(os.path.join(image_dir, f)),
                'text': cap.strip(),
            })

        # 释放显存，避免 OOM
        del inputs
        torch.cuda.empty_cache()

    # 写入 metadata.jsonl（JSON Lines 格式，每行一个 JSON）
    with open(output_meta, 'w') as f:
        for m in metadata:
            f.write(json.dumps(m) + '\n')

    print(f"\nSaved {len(metadata)} captions to {output_meta}")

    # 打印前 5 条样例，方便人工检查质量
    print("\nSample captions:")
    for m in metadata[:5]:
        print(f"  {os.path.basename(m['file_name'])}: {m['text']}")

    return metadata


# ============================================================
# 步骤 3：LoRA 微调
# ============================================================
def train_lora(data_root, epochs=20, rank=16, lr=1e-4, batch_size=4, resolution=256):
    """
    调用 train_lora.py 执行 LoRA 微调。

    参数说明：
      data_root   — 包含 metadata.jsonl 和 images/ 的目录
      epochs      — 训练轮数（所有图片看多少遍）
      rank        — LoRA 秩（低秩矩阵维度，越大表达力越强）
      lr          — 学习率（每步参数更新幅度）
      batch_size  — 批量大小（每步处理多少张图）
      resolution  — 训练分辨率（图片缩放到多大）
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    # 检查 metadata.jsonl 是否存在（需要先跑完 caption 步骤）
    meta_path = os.path.join(data_root, 'metadata.jsonl')
    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} not found. Run caption step first.")
        return

    print(f"\nStarting LoRA training...")
    print(f"  Data: {meta_path}")
    print(f"  Epochs: {epochs}, Rank: {rank}, LR: {lr}, Batch: {batch_size}, Resolution: {resolution}")

    # 导入 train_lora.py 的 main 函数
    # 通过修改 sys.argv 传参，模拟命令行调用
    from train_lora import main as train_main

    sys.argv = [
        'train_lora.py',
        '--data_dir', data_root,
        '--epochs', str(epochs),
        '--rank', str(rank),
        '--lr', str(lr),
        '--batch_size', str(batch_size),
        '--resolution', str(resolution),
    ]
    train_main()


# ============================================================
# 主函数：组装三步流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='EmoSet LoRA Pipeline')
    parser.add_argument('--image_dir', type=str,
                        default='/mnt/workspace/data/EmoEdit_origin_image/origin_image',
                        help='原始图片目录（可以是嵌套文件夹）')
    parser.add_argument('--data_root', type=str, default=None,
                        help='工作目录，存放处理后的数据（默认：image_dir 的上级目录）')
    parser.add_argument('--step', type=str, default='all',
                        choices=['collect', 'caption', 'train', 'all'],
                        help='执行哪个步骤：collect/caption/train/all')
    parser.add_argument('--max_images', type=int, default=None,
                        help='题注最多处理多少张图（默认：全部）')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--resolution', type=int, default=256)
    args = parser.parse_args()

    # data_root 默认为 image_dir 的上级目录
    # 例如 image_dir=/data/EmoEdit/origin_image → data_root=/data/EmoEdit
    if args.data_root is None:
        args.data_root = os.path.dirname(args.image_dir.rstrip('/'))

    # 收集后的图片放在 data_root/images/ 下
    flat_dir = os.path.join(args.data_root, 'images')

    # 执行各步骤
    if args.step in ('collect', 'all'):
        collect_images(args.image_dir, flat_dir)

    if args.step in ('caption', 'all'):
        meta_path = os.path.join(args.data_root, 'metadata.jsonl')
        generate_captions(flat_dir, meta_path, args.max_images)

    if args.step in ('train', 'all'):
        train_lora(args.data_root, args.epochs, args.rank, args.lr,
                   args.batch_size, args.resolution)

    print("\n=== Pipeline Complete ===")
    print(f"Generate images: python inference_lora.py --lora_path checkpoints/lora/final --prompt 'a photo'")


if __name__ == '__main__':
    main()
