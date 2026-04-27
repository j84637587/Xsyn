# Taming Generative Synthetic Data for X-ray Prohibited Item Detection

[[Arxiv](https://arxiv.org/abs/2511.15299)] [[HuggingFace](https://huggingface.co/papers/2511.15299)]

<figure style="display:block; text-align:center; margin:0 auto;">
  <img src="figures/analysis.jpg"
       alt="Analysis of existing X-ray image synthesis methods"
       style="width:90%; max-width:600px; margin:0 auto; display:block;">
</figure>

We propose **Xsyn**, a simple and effective one-stage synthesis pipeline for X-ray security imaging. To the best of our knowledge, Xsyn is the first to achieve high-quality X-ray security image synthesis without incurring additional labor-intensive foreground preparation.

---

## Table of Contents

- [Taming Generative Synthetic Data for X-ray Prohibited Item Detection](#taming-generative-synthetic-data-for-x-ray-prohibited-item-detection)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
    - [Environment](#environment)
  - [Data Preparation](#data-preparation)
  - [Download Checkpoints](#download-checkpoints)
  - [Inference](#inference)
    - [Basic (no SAM)](#basic-no-sam)
    - [With SAM + CAR annotation refinement](#with-sam--car-annotation-refinement)
    - [Key arguments](#key-arguments)
  - [Output Structure](#output-structure)
  - [Training](#training)
  - [Acknowledgment](#acknowledgment)

---

## Requirements

### Environment

**Option A — conda env file (recommended)**

```bash
conda env create -f environment_win.yml   # Windows + CUDA 12.1
conda activate xsyn
```

**Option B — manual install**

```bash
conda create -n xsyn python=3.10 -c conda-forge -y
conda activate xsyn
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install setuptools diffusers==0.29.2 transformers==4.46.3 omegaconf==2.1.1 \
    numpy==1.24.4 Pillow==9.5.0 opencv-python==4.10.0.84 scipy==1.10.1 \
    scikit-image==0.21.0 accelerate==0.31.0 huggingface-hub==0.23.4 einops==0.8.0 \
    peft==0.13.2 segment-anything pycocotools safetensors tokenizers tqdm \
    matplotlib pandas scikit-learn timm pytorch-lightning torchmetrics gradio \
    torchviz kornia==0.6.0 albumentations==1.4.15 bezier natsort
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/CompVis/taming-transformers.git
```

---

## Data Preparation

Supported datasets: [PIDray](https://github.com/lutao2021/PIDray), [OPIXray](https://github.com/DIG-Beihang/OPIXray), [HiXray](https://github.com/HiXray-author/HiXray).

Place the dataset under `data/` following this structure:

```
data/
└── pidray/
    ├── train/          # training images
    ├── test/           # test images
    └── annotations/
        ├── train.json
        ├── test.json
        ├── test_easy.json
        ├── test_hard.json
        └── test_hidden.json
```

Convert COCO-format annotations to `.pt` files required by the dataloader:

```bash
python prepare_pidray.py --ann_file data/pidray/annotations/train.json       --output data/pidray/pidray_train.pt
python prepare_pidray.py --ann_file data/pidray/annotations/test.json         --output data/pidray/pidray_test.pt
python prepare_pidray.py --ann_file data/pidray/annotations/test_easy.json    --output data/pidray/pidray_test_easy.pt
python prepare_pidray.py --ann_file data/pidray/annotations/test_hard.json    --output data/pidray/pidray_test_hard.pt
python prepare_pidray.py --ann_file data/pidray/annotations/test_hidden.json  --output data/pidray/pidray_test_hidden.pt
```

---

## Download Checkpoints

All models are based on [GLIGEN](https://github.com/gligen/GLIGEN).

| Dataset | Mode                     | Download                                       |
| ------- | ------------------------ | ---------------------------------------------- |
| PIDray  | text-grounded inpainting | [HF Hub](https://huggingface.co/Pillow-1/Xsyn) |
| OPIXray | text-grounded inpainting | [HF Hub](https://huggingface.co/Pillow-1/Xsyn) |
| HiXray  | text-grounded inpainting | [HF Hub](https://huggingface.co/Pillow-1/Xsyn) |

```bash
huggingface-cli download Pillow-1/Xsyn pidray_xsyn.pth --local-dir checkpoints
```

Place checkpoint files under `checkpoints/`.

---

## Inference

### Basic (no SAM)

|生成的圖完全一樣，差別只在 annotation txt 裡的 bbox 準不準。
|如果只是要看生成效果，Basic 就夠了。如果要拿去訓練偵測模型、需要精準 bbox，才需要 SAM + CAR。

| | Basic |	With SAM + CAR |
| 生成圖 | 一樣 | 一樣 |
| bbox 來源 | 直接用原始標註縮放 | 用 CA map + SAM 精修過的 bbox |
| 需要 SAM checkpoint | 不需要 | 需要 sam_vit_h_4b8939.pth |
| 速度 | 快 | 慢（每張多跑 SAM） |
| 標註品質 | 可能偏移 | 更貼近實際合成位置 |

```bash
python gligen_inference.py \
    --ckpt_path checkpoints/pidray_xsyn.pth \
    --gligen_caption_pt data/pidray/pidray_train.pt \
    --image_path data/pidray/train \
    --output_path output/images/ \
    --annotation_path output/annotations/ \
    --use_sam False \
    --refine_anno False \
    --gen_method 1 \
    --batch_size 1
```

### With SAM + CAR annotation refinement

```bash
python gligen_inference.py \
    --ckpt_path checkpoints/pidray_xsyn.pth \
    --gligen_caption_pt data/pidray/pidray_train.pt \
    --image_path data/pidray/train \
    --output_path output/images/ \
    --annotation_path output/annotations/ \
    --sam_weight checkpoints/sam_vit_h_4b8939.pth \
    --use_sam True \
    --refine_anno True \
    --gen_method 1 \
    --batch_size 1
```

### Key arguments

| Argument              | Description                                                   |
| --------------------- | ------------------------------------------------------------- |
| `--ckpt_path`         | Path to the Xsyn model checkpoint                             |
| `--gligen_caption_pt` | Preprocessed `.pt` datafile (from `prepare_pidray.py`)        |
| `--image_path`        | Directory of source images to inpaint                         |
| `--output_path`       | Directory to save generated images                            |
| `--annotation_path`   | Directory to save bounding box annotations (TXT)              |
| `--vis_path`          | Directory to save GT-comparison visualizations                |
| `--ca_vis_path`       | Directory to save cross-attention map visualizations          |
| `--gen_method`        | `1` = Xsyn-M (fixed position), `3` = Xsyn-A (reference-based) |
| `--refine_anno`       | `True` to enable **CAR** (Cross-Attention Refinement)         |
| `--latent_redist`     | `True` to enable **BOM** (Background Occlusion Modeling)      |
| `--use_sam`           | `True` to use SAM for mask generation (required by BOM)       |
| `--sam_weight`        | Path to SAM checkpoint (`sam_vit_h_4b8939.pth`)               |

---

## Output Structure

```
output/
├── images/
│   └── ori/                        # Generated synthetic X-ray images (PNG)
│       └── {image_id}_{prompt}.png
└── annotations/
    ├── {id}_annotation.txt         # Raw GLIGEN bounding box output
    └── {id}_annotation_refine.txt  # CAR-refined bounding box (if --refine_anno True)
```

Additional directories created when `--refine_anno True`:

```
output/
├── visualization/                  # Side-by-side comparison with ground truth
└── visualization_ca/               # Cross-attention map overlays
```

---

## Training

Please follow the text-grounded inpainting training instructions in [GLIGEN](https://github.com/gligen/GLIGEN).

After inference, use `downstream_test.sh` to evaluate synthetic data performance. The downstream detection environment is [mmdetection](https://github.com/open-mmlab/mmdetection).

---

## Acknowledgment

This work is implemented based on [GLIGEN](https://github.com/gligen/GLIGEN). We greatly appreciate their valuable contributions to the community.
