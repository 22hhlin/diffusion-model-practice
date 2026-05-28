"""
下载数据集 + BLIP2 自动题注

功能：
    1. 从 HuggingFace 下载图片数据集
    2. 用 BLIP2 为每张图片自动生成文字描述
    3. 输出 metadata.jsonl 供 LoRA 训练使用

BLIP2 题注原理：
    BLIP2 = ViT（视觉编码器）+ Q-Former（桥梁）+ OPT-2.7B（语言模型）
    流程：
      图片 → ViT 提取视觉特征 → Q-Former 转为"查询向量" → OPT 生成文字
    Q-Former 的作用：
      把 ViT 的高维视觉特征压缩为固定数量的向量（32 个），
      作为 OPT 的"前缀"，引导语言模型生成描述

Usage:
    # 下载 200 张图片并题注
    python download_and_caption.py --num_images 200

    # 用指定数据集
    python download_and_caption.py --dataset lambdalabs/pokemon-blip-captions --num_images 100

    # 只对已有图片题注（跳过下载）
    python download_and_caption.py --image_dir ./my_images --skip_download
"""
import argparse
import os
import json
import torch
from PIL import Image
from tqdm import tqdm


# ============================================================
# 步骤 1：下载图片
# ============================================================
def download_images(dataset_id, split, output_dir, num_images, hf_token=None):
    """
    从 HuggingFace datasets 下载图片数据集。

    HuggingFace datasets 是一个数据集仓库，支持流式加载（streaming=True），
    不需要把整个数据集下载到内存，逐条读取即可。

    Args:
        dataset_id: 数据集 ID，如 'lambdalabs/pokemon-blip-captions'
        split: 数据集分片，如 'train', 'test'
        output_dir: 图片保存目录
        num_images: 下载多少张
        hf_token: HuggingFace 访问令牌（私有数据集需要）
    """
    from datasets import load_dataset

    print(f"Loading: {dataset_id}")

    # streaming=True：流式加载，逐条读取，不占内存
    # 适合大数据集（几万张以上）
    ds = load_dataset(dataset_id, split=split, streaming=True, token=hf_token)

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for item in tqdm(ds, desc="Downloading"):
        if count >= num_images:
            break

        # 不同数据集的图片字段名不同，尝试常见 key
        image = None
        for key in ['image', 'img', 'pixel_values', 'bytes']:
            if key in item:
                image = item[key]
                break

        if image is None:
            continue

        # 处理不同格式的图片数据
        if isinstance(image, dict) and 'bytes' in image:
            # 有些数据集把图片存为 bytes 格式
            from io import BytesIO
            image = Image.open(BytesIO(image['bytes']))
        if not isinstance(image, Image.Image):
            continue

        # 保存为 512×512 的 PNG
        img_path = os.path.join(output_dir, f'{count:05d}.png')
        image.convert('RGB').resize((512, 512)).save(img_path)
        count += 1

    print(f"Downloaded {count} images to {output_dir}")
    return count


# ============================================================
# 步骤 2：BLIP2 自动题注
# ============================================================
def generate_captions(image_dir, output_meta, batch_size=4):
    """
    用 BLIP2 为目录中的所有图片生成文字描述。

    BLIP2 推理流程：
      1. 图片预处理：resize → normalize → tensor
      2. ViT 编码：图片 → 视觉特征 [1, 257, 1408]
         （257 = 1 CLS token + 256 patch token，每个 patch 是 16×16 像素）
      3. Q-Former 压缩：[1, 257, 1408] → [1, 32, 768]
         （32 个可学习的查询向量，通过交叉注意力提取关键信息）
      4. OPT 生成：以 Q-Former 输出为前缀，自回归生成文字
         "a" → "a sunset" → "a sunset over" → "a sunset over the ocean"

    Args:
        image_dir: 图片目录
        output_meta: 输出 metadata.jsonl 路径
        batch_size: 批量处理大小
    """
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    # processor = 图片预处理（ViT 的 normalize）+ 文字 tokenize
    # model = BLIP2 完整模型（ViT + Q-Former + OPT-2.7B）
    print("Loading BLIP2...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16   # 半精度加载，约 5GB 显存
    ).to('cuda')

    # 扫描目录中的图片文件
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = sorted([f for f in os.listdir(image_dir)
                     if os.path.splitext(f)[1].lower() in valid_ext])

    print(f"Captioning {len(images)} images...")
    metadata = []

    # 逐批处理（batch 比逐张快，因为 GPU 可以并行）
    for i in tqdm(range(0, len(images), batch_size), desc="BLIP2"):
        batch_files = images[i:i+batch_size]

        # 加载一批图片，转为 RGB（排除 RGBA 灰度等格式）
        batch_imgs = [Image.open(os.path.join(image_dir, f)).convert('RGB')
                      for f in batch_files]

        # processor 把图片转为 ViT 的输入 tensor
        # 包含：resize 到 224×224 → normalize（ImageNet 均值/标准差）→ 转 tensor
        inputs = processor(images=batch_imgs, return_tensors="pt").to('cuda', torch.float16)

        # generate() 自回归生成文字
        # max_new_tokens=50：最多生成 50 个词（BLIP2 默认用 OPT 的 tokenizer）
        ids = model.generate(**inputs, max_new_tokens=50)

        # 把 token ID 解码回文字
        # skip_special_tokens=True：去掉 <s>, </s>, <pad> 等特殊标记
        captions = processor.batch_decode(ids, skip_special_tokens=True)

        # 保存每张图片的绝对路径和生成的描述
        for f, cap in zip(batch_files, captions):
            metadata.append({
                'file_name': os.path.abspath(os.path.join(image_dir, f)),
                'text': cap.strip(),
            })

        # 释放显存（inputs 在 GPU 上，不释放会累积）
        del inputs
        torch.cuda.empty_cache()

    # 写入 metadata.jsonl
    # JSON Lines 格式：每行一个独立的 JSON 对象
    # 和普通 JSON 数组的区别：可以逐行追加，不需要读取整个文件
    with open(output_meta, 'w') as f:
        for m in metadata:
            f.write(json.dumps(m) + '\n')

    print(f"Saved {len(metadata)} captions to {output_meta}")
    return metadata


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='lambdalabs/pokemon-blip-captions',
                        help='HuggingFace 数据集 ID')
    parser.add_argument('--split', type=str, default='train',
                        help='数据集分片（train/test/validation）')
    parser.add_argument('--image_dir', type=str, default='./downloaded_images',
                        help='图片保存/读取目录')
    parser.add_argument('--num_images', type=int, default=200,
                        help='下载图片数量')
    parser.add_argument('--skip_download', action='store_true',
                        help='跳过下载，只对已有图片题注')
    parser.add_argument('--metadata', type=str, default='metadata.jsonl',
                        help='输出 metadata 文件名')
    args = parser.parse_args()

    # 下载图片（除非 --skip_download）
    if not args.skip_download:
        download_images(args.dataset, args.split, args.image_dir, args.num_images)

    # 生成题注
    # metadata 保存在 image_dir 的上级目录
    meta_path = os.path.join(os.path.dirname(os.path.abspath(args.image_dir)), args.metadata)
    generate_captions(args.image_dir, meta_path)

    print(f"\nDone! Train with:")
    print(f"  python train_lora.py --data_dir {os.path.dirname(os.path.abspath(args.image_dir))}")


if __name__ == '__main__':
    main()
