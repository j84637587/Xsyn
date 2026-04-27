# Taming Generative Synthetic Data for X-ray Prohibited Item Detection

[[Arxiv](https://arxiv.org/abs/2511.15299)] [[HF](https://huggingface.co/papers/2511.15299)]

<figure style="display:block; text-align:center; margin:0 auto;">
  <img src="figures/analysis.jpg"
       alt="Analysis of existing X-ray image synthesis methods"
       style="width:90%; max-width:600px; margin:0 auto; display:block;">
</figure>

- We propose Xsyn, a simple and effective one-stage
 synthesis pipeline in the X-ray security domain. To the
 best of our knowledge, Xsyn is the first to achieve high
quality X-ray security image synthesis without incurring
 additional labor-intensive foreground preparation.

## Requirements
- Enviroment: We provide [enviroment.yml](environment.yml) to setup enviroment.
- Data: [PIDray](https://github.com/lutao2021/PIDray), [OPIXray](https://github.com/DIG-Beihang/OPIXray) and [HiXray](https://github.com/HiXray-author/HiXray/blob/main/README.md). 



```
python prepare_pidray.py --ann_file DATA/pidray/annotations/train.json --output   DATA/pidray/pidray_train.pt          
python prepare_pidray.py --ann_file DATA/pidray/annotations/test.json --output   DATA/pidray/pidray_test.pt  
python prepare_pidray.py --ann_file DATA/pidray/annotations/test_easy.json --output   DATA/pidray/pidray_test_easy.pt
python prepare_pidray.py --ann_file DATA/pidray/annotations/test_hard.json --output   DATA/pidray/pidray_test_hard.pt
python prepare_pidray.py --ann_file DATA/pidray/annotations/test_hidden.json --output   DATA/pidray/pidray_test_hidden.pt
```

```
conda env create -f environment.yml
```

```
conda create -n xsyn python=3.10 -c conda-forge -y
conda activate xsyn
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install setuptools diffusers==0.29.2 transformers==4.46.3 omegaconf==2.1.1 numpy==1.24.4 Pillow==9.5.0 opencv-python==4.10.0.84 scipy==1.10.1 scikit-image==0.21.0 accelerate==0.31.0 huggingface-hub==0.23.4 einops==0.8.0 peft==0.13.2 clip-anytorch segment-anything pycocotools safetensors tokenizers tqdm matplotlib pandas scikit-learn timm pytorch-lightning torchmetrics gradio 
pip install torchviz kornia==0.6.0 albumentations==1.4.15 bezier natsort
pip install git+https://github.com/openai/CLIP.git
```

## Download Xsyn models

We will provide checkpoints for different datasets. All models here are based on GLIGEN.
| Dataset | Mode                     | Download                                       |
| ------- | ------------------------ | ---------------------------------------------- |
| PIDray  | text-grounded inpainting | [HF Hub](https://huggingface.co/Pillow-1/Xsyn) |
| OPIXray | text-grounded inpainting | [HF Hub](https://huggingface.co/Pillow-1/Xsyn) |
| HiXray  | text-grounded inpainting | [HF Hub](https://huggingface.co/Pillow-1/Xsyn) |

```
huggingface-cli download Pillow-1/Xsyn pidray_xsyn.pth --local-dir Xray/checkpoints
```

## Training
Please follow the instruction of text-grounded inpainting training in [GLIGEN](https://github.com/gligen/GLIGEN).

## Inference
We provide one script to generate x-ray security images and construct their annotations. First download models and put them in `--ckpt_path`. Then run
```bash
python gligen_inference.py

python gligen_inference.py --ckpt_path checkpoints/pidray_xsyn.pth  --gligen_caption_pt data/pidray/pidray_train.pt --image_path data/pidray/train --output_path output/images/ --annotation_path output/annotations/ --use_sam False --refine_anno False --gen_method 1 --batch_size 1

python gligen_inference.py --ckpt_path checkpoints/pidray_xsyn.pth --gligen_caption_pt data/pidray/pidray_train.pt --image_path data/pidray/train --output_path output/images/ --annotation_path output/annotations/ --sam_weight checkpoints/sam_vit_h_4b8939.pth --use_sam True --refine_anno True --gen_method 1 --batch_size 1
```

Details of some important args:

- `--output_path`: the path to save your generated x-ray security images
- `--annotation_path`: the path to save the refined annotation(stored in txt format)
- `--vis_path`: the path to save visualization compared with gt
- `--ca_vis_path`: the path to save cross-attention maps
- `--image_path`: the path to load images you want to inpaint
- `--ckpt_path`: the generation model checkpoint path
- `--gligen_caption_pt`: the file to prepare your training/test data in [GLIGEN](https://github.com/gligen/GLIGEN) format
- `--gen_method`: set to 1 for Xsyn-M and 3 for Xsyn-A
- `--refine_anno`: set to True for `CAR`
- `--latent_redist`: set to True for `BOM`

After inference, we use `downstream_test.sh` to test the performance of our sythetic data. Our downstream detection environment is [mmdetection](https://github.com/open-mmlab/mmdetection).

## 🙏 Acknowledgment
This work is implemented based on [GLIGEN](https://github.com/gligen/GLIGEN). We greatly appreciate their valuable contributions to the community.