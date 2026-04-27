import argparse
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.util import *
from ldm.models.diffusion.plms import PLMSSampler
import os 
from transformers import CLIPProcessor, CLIPModel
from copy import deepcopy
import torch 
from torch.utils.data import Dataset, DataLoader
from ldm.util import instantiate_from_config
from trainer import read_official_ckpt, batch_to_device
from inpaint_mask_func import draw_masks_from_boxes
import numpy as np
import clip 
from scipy.io import loadmat
from functools import partial
import torchvision.transforms.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from distributed_util import init_distributed_mode, dist_cleanup
from diffusers.pipelines.stable_diffusion.clip_image_project_model import CLIPImageProjection
from utils import *
import json
import cv2
import pdb
import random
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from annotation_refine import Caption_coco
from transformers import CLIPTokenizer
from config import RunConfig
import random
from pycocotools.coco import COCO
from collections import Counter


def get_anno_from_img_name(file_name, coco):

    # 获取所有图片的ID
    img_ids = coco.getImgIds()

    # 遍历所有图片ID，找到文件名匹配的图片ID
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        if img_info['file_name'].startswith(file_name):
            # 获取该图片ID的所有标注
            annIds = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(annIds)
            return anns

def set_alpha_scale(model, alpha_scale):
    from ldm.modules.attention import GatedCrossAttentionDense, GatedSelfAttentionDense
    for module in model.modules():
        if type(module) == GatedCrossAttentionDense or type(module) == GatedSelfAttentionDense:
            module.scale = alpha_scale


def alpha_generator(length, type=None):
    """
    length is total timestpes needed for sampling. 
    type should be a list containing three values which sum should be 1
    
    It means the percentage of three stages: 
    alpha=1 stage 
    linear deacy stage 
    alpha=0 stage. 
    
    For example if length=100, type=[0.8,0.1,0.1]
    then the first 800 stpes, alpha will be 1, and then linearly decay to 0 in the next 100 steps,
    and the last 100 stpes are 0.    
    """
    if type == None:
        type = [1,0,0]

    assert len(type)==3 
    assert type[0] + type[1] + type[2] == 1
    
    stage0_length = int(type[0]*length)
    stage1_length = int(type[1]*length)
    stage2_length = length - stage0_length - stage1_length
    
    if stage1_length != 0: 
        decay_alphas = np.arange(start=0, stop=1, step=1/stage1_length)[::-1]
        decay_alphas = list(decay_alphas)
    else:
        decay_alphas = []
        
    
    alphas = [1]*stage0_length + decay_alphas + [0]*stage2_length
    
    assert len(alphas) == length
    
    return alphas



def load_ckpt(ckpt_path, device):

    saved_ckpt = torch.load(ckpt_path, map_location='cpu')
    config = saved_ckpt["config_dict"]["_content"]

    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    # donot need to load official_ckpt for self.model here, since we will load from our ckpt
    model.load_state_dict( saved_ckpt['model'] )
    autoencoder.load_state_dict( saved_ckpt["autoencoder"]  )
    text_encoder.load_state_dict( saved_ckpt["text_encoder"]  )
    diffusion.load_state_dict( saved_ckpt["diffusion"]  )

    return model, autoencoder, text_encoder, diffusion, config




def project(x, projection_matrix):
    """
    x (Batch*768) should be the penultimate feature of CLIP (before projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.  
    this function will return the CLIP feature (without normalziation)
    """
    return x@torch.transpose(projection_matrix, 0, 1)


def get_clip_feature(model, processor, input, image_project, device, is_image=False):
    which_layer_text = 'before'
    which_layer_image = 'after_reproject'

    if is_image:
        if input == None:
            return None
        #image = Image.open(input).convert("RGB")
        inputs = processor(images=[input],  return_tensors="pt", padding=True)
        inputs['pixel_values'] = inputs['pixel_values'].to(device) # we use our own preprocessing without center_crop 
        inputs['input_ids'] = torch.tensor([[0,1,2,3]]).to(device)  # placeholder
        outputs = model(**inputs)
        feature = outputs.image_embeds 
        if which_layer_image == 'after_reproject':
            #feature = project( feature, torch.load('projection_matrix').to(device).T ).squeeze(0)
            feature = image_project(feature).squeeze(0)
            feature = ( feature / feature.norm() )  * 28.7 
            feature = feature.unsqueeze(0)
    else:
        if input == None:
            return None
        inputs = processor(text=input,  return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['pixel_values'] = torch.ones(1,3,224,224).to(device) # placeholder 
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        outputs = model(**inputs)
        if which_layer_text == 'before':
            feature = outputs.text_model_output.pooler_output
    return feature


def complete_mask(has_mask, max_objs):
    mask = torch.ones(1,max_objs)
    if has_mask == None:
        return mask 

    if type(has_mask) == int or type(has_mask) == float:
        return mask * has_mask
    else:
        for idx, value in enumerate(has_mask):
            mask[0,idx] = value
        return mask



@torch.no_grad()
def prepare_batch(meta, device, batch=1, max_objs=30):
    phrases, images = meta.get("phrases"), meta.get("images")
    phrases = [None]*len(images) if phrases==None else phrases 
    images = [None]*len(phrases) if images==None else images 

    version = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(version).to(device)
    processor = CLIPProcessor.from_pretrained(version)
    image_project = CLIPImageProjection().to(device)

    boxes = torch.zeros(max_objs, 4)
    masks = torch.zeros(max_objs)
    text_masks = torch.zeros(max_objs)
    image_masks = torch.zeros(max_objs)
    text_embeddings = torch.zeros(max_objs, 768)
    image_embeddings = torch.zeros(max_objs, 768)
    
    text_features = []
    image_features = []
    for phrase, image in zip(phrases,images):
        text_features.append(  get_clip_feature(model, processor, phrase, image_project, device, is_image=False) )
        image_features.append( get_clip_feature(model, processor, image, image_project, device, is_image=True) )

    for idx, (box, text_feature, image_feature) in enumerate(zip( meta['locations'], text_features, image_features)):
        boxes[idx] = torch.tensor(box)
        masks[idx] = 1
        if text_feature is not None:
            text_embeddings[idx] = text_feature
            text_masks[idx] = 1 
        if image_feature is not None:
            image_embeddings[idx] = image_feature
            image_masks[idx] = 1 
    '''
    out = {
        "boxes" : boxes.unsqueeze(0).repeat(batch,1,1),
        "masks" : masks.unsqueeze(0).repeat(batch,1),
        "text_masks" : text_masks.unsqueeze(0).repeat(batch,1)*complete_mask( meta.get("text_mask"), max_objs ),
        "image_masks" : image_masks.unsqueeze(0).repeat(batch,1)*complete_mask( meta.get("image_mask"), max_objs ),
        "text_embeddings"  : text_embeddings.unsqueeze(0).repeat(batch,1,1),
        "image_embeddings" : image_embeddings.unsqueeze(0).repeat(batch,1,1)
    }
    '''
    out = {
        "boxes" : boxes.unsqueeze(0).repeat(batch,1,1),
        "masks" : masks.unsqueeze(0).repeat(batch,1),
        "text_embeddings"  : text_embeddings.unsqueeze(0).repeat(batch,1,1)
    }
    return batch_to_device(out, device) 

@torch.no_grad()
def prepare_batch_depth(meta, device, batch=1):
    
    pil_to_tensor = transforms.PILToTensor()

    depth = Image.open(meta['depth']).convert("RGB")
    depth = crop_and_resize(depth)
    depth = ( pil_to_tensor(depth).float()/255 - 0.5 ) / 0.5

    out = {
        "depth" : depth.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 


def crop_and_resize(image):
    crop_size = min(image.size)
    image = TF.center_crop(image, crop_size)
    image = image.resize( (512, 512) )
    return image

def pil_to_tensor(pil_image):
    arr = np.array(pil_image)
    
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2,0,1])

    return torch.tensor(arr)

def colorEncode(labelmap, colors):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)

    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    return labelmap_rgb

def get_bbox_from_mask_img(mask_image):
    # 找到图像中的轮廓
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #assert len(contours) == 1, f'{mask_name} has more than 1 contours!'
    # 遍历所有轮廓，找出面积最大的轮廓
    max_area = -1
    max_contour = None
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
            
    if max_contour is not None:
        # 获取外接矩形的坐标和尺寸
        x, y, w, h = cv2.boundingRect(max_contour)
        return [x, y, w, h]
    else:
        return None

def iou(box1, box2):
	'''
	2D IoU计算
	box表示形式：[x1,y1,x2,y2] 分别是两对角点的坐标
	'''
	in_w = min(box1[2],box2[2]) - max(box1[0],box2[0])
	in_h = min(box1[3],box2[3]) - max(box1[1],box2[1])

	inter = 0 if in_w < 0 or in_h < 0 else in_w * in_h
	union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1])  - inter
	iou = inter / (union + 1e-16)
	return iou

def isOk(this_box, boxes, sam_box_iou_thre):
    ok = True
    for box in boxes:
        ok &= (iou(this_box, box) < sam_box_iou_thre)
    return ok

def select_box(all_boxes, boxes, all_masks, sam_box_iou_thre, min_area_thre=500, max_area_thre=30000):
    '''
    pidray: 500, 30000
    opiray: 1050, 60000
    hixray: 1400, 290000
    '''
    candidate = []
    masks = []
    for single_box, single_mask in zip(all_boxes, all_masks):
        x, y, w, h = single_box
        area = w*h
        if area < min_area_thre or area > max_area_thre:
            continue
        this_box = [x, y, x + w, y + h]
        if isOk(this_box, boxes, sam_box_iou_thre):
            candidate.append(this_box)
            masks.append(single_mask)
    
    if len(candidate) == 0:
        return None, None
    else:
        return candidate, masks
        

def select_category(selected_box, category_group_area_boundary):
    x1, y1, x2, y2 = selected_box
    area = (x2 - x1) * (y2 - y1)
    for i in range(len(category_group_area_boundary)):
        if area < category_group_area_boundary[i]:
            return i 
    return len(category_group_area_boundary) 

@torch.no_grad()
def run(config):
    
    ###分布式设置###
    init_distributed_mode(args)
    device = torch.device(args.device)
    #args.device = device
    torch.cuda.set_device(args.gpu)

    # - - - - - prepare models - - - - - # 
    model, autoencoder, text_encoder, diffusion, config = load_ckpt(args.ckpt_path, device)
    model.device = device

    tokenizer = CLIPTokenizer.from_pretrained(getattr(args, 'tokenizer_path', 'openai/clip-vit-large-patch14'))
    
    if args.use_sam:
        sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_weight)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        mask_generator = SamAutomaticMaskGenerator(sam)
    
    #pdb.set_trace()

    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input
    
    
    grounding_downsampler_input = None
    if "grounding_downsampler_input" in config:
        grounding_downsampler_input = instantiate_from_config(config['grounding_downsampler_input'])


    # - - - - - update config from args - - - - - #
    from omegaconf import OmegaConf as _OC, DictConfig
    if isinstance(config, DictConfig):
        config = _OC.to_container(config, resolve=True)
    config.update(vars(args))
    config = OmegaConf.create(config)


    # - - - - - prepare batch - - - - - #

    # pidray
    
    categories = [
      'Baton', 'Bullet', 'Gun', 'Hammer', 'Powerbank', 'Wrench',
      'HandCuffs', 'Knife', 'Lighter', 'Pliers', 'Scissors', 'Sprayer'
    ]
    group1_categories = ['Lighter', 'Bullet']
    group3_categories = ['Hammer']
    group2_categories = [ 'Knife', 'Gun', 'Powerbank', 'Wrench',
      'HandCuffs', 'Baton', 'Pliers', 'Scissors', 'Sprayer']
    
    categories2id = {
      "Baton": 1,
      "Pliers": 2,
      "Hammer": 3,
      "Powerbank": 4,
      "Scissors": 5,
      "Wrench": 6,
      "Gun": 7,
      "Bullet": 8,
      "Sprayer": 9,
      "HandCuffs": 10,
      "Knife": 11,
      "Lighter": 12
    }
    
    
    # opiray
    '''
    categories = ["Utility_Knife", "Multi-tool_Knife", "Folding_Knife", "Straight_Knife", "Scissor"]
    categories2id = {
        "Utility_Knife":1,
        "Multi-tool_Knife":2,
        "Folding_Knife":3,
        "Straight_Knife":4,
        "Scissor":5
    }
    group1_categories = ["Multi-tool_Knife", "Folding_Knife"]
    group3_categories = ["Scissor"]
    group2_categories = [ "Straight_Knife", "Utility_Knife"]
    '''

    # hixray
    '''
    categories = ['Portable_Charger_1', "Portable_Charger_2", "Water", "Laptop", "Mobile_Phone", "Tablet", "Cosmetic", "Nonmetallic_Lighter"]
    categories2id = {
      "Portable_Charger_1": 1,
      "Portable_Charger_2": 2,
      "Water": 3,
      "Laptop": 4,
      "Mobile_Phone": 5,
      "Tablet": 6,
      "Cosmetic": 7,
      "Nonmetallic_Lighter": 8
    }
    group1_categories = ['Portable_Charger_1', "Portable_Charger_2", "Water", "Mobile_Phone", "Cosmetic", "Nonmetallic_Lighter"]
    group2_categories = ["Tablet"]
    group3_categories = ["Laptop"]
    '''

    group_categories = [group1_categories, group2_categories, group3_categories]

    if args.scratch_generate:
        dataset = Caption(args)
    else:
        dataset = Caption_coco(args)
    print(f'len(dataset) = {len(dataset)}')
    if getattr(args, 'distributed', False):
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    num_workers = 0 if os.name == 'nt' else 8
    data_loader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=num_workers, collate_fn = dataset.collate_fn, drop_last = False)
    print(f'len(data_loader)={len(data_loader)}')
    
    specific_category_filenames = {}
    fti_filenames = os.listdir(args.ref_path) if args.ref_path else []

    for category in categories:
      specific_category_fti(category, fti_filenames, specific_category_filenames)
    
    os.makedirs(args.output_path, exist_ok=True)
    
    os.makedirs(args.annotation_path, exist_ok=True)
    
    os.makedirs(args.vis_path, exist_ok=True)
    
    os.makedirs(args.ca_vis_path, exist_ok=True)

    
    if args.gen2_hidden:
        coco = COCO(args.gen2_hidden_anno)
    
    if args.gen3_hidden:
        coco = COCO(args.gen3_hidden_anno)

    f = open(args.annotation_path + f'{args.gpu}_' + 'annotation.txt', 'a+')
    
    if args.refine_anno:
        f_list = []
        sampling_info = {}
        '''
        if args.refine_strategies == 'topk_sampling':
            for topk in args.topks:
                topk_annotation_path = os.path.join(args.annotation_path, str(topk))
                os.makedirs(topk_annotation_path, exist_ok=True)
                f_list.append(open(os.path.join(topk_annotation_path, f'{args.gpu}_' + 'annotation.txt'), 'a+'))
        elif args.refine_strategies == 'h_sampling':
            for range_num in args.range_nums:
                range_annotation_path = os.path.join(args.annotation_path, str(range_num))
                os.makedirs(range_annotation_path, exist_ok=True)
                f_list.append(open(os.path.join(range_annotation_path, f'{args.gpu}_' + 'annotation.txt'), 'a+'))
        '''
        if args.ablation_opt is not None:
            for ablation in args.ablation_opt:
                ablation_anno_path = os.path.join(args.annotation_path, ablation)
                os.makedirs(ablation_anno_path, exist_ok=True)
                f_list.append(open(os.path.join(ablation_anno_path, f'{args.gpu}_' + 'annotation.txt'), 'a+'))
            
            for hidden_idx in range(len(args.alpha)):
                os.makedirs(os.path.join(args.output_path, str(args.alpha[hidden_idx])), exist_ok=True)

        
    
    with open(args.annotation_path + f'{args.gpu}_' + 'annotation_refine.txt', 'a+') as refine_f:
      #left_num = 2
      for internel_iter, gligen_caption_prompts in enumerate(data_loader):
        for gligen_caption_prompt in gligen_caption_prompts:
          
          if args.scratch_generate:
            image = Image.open(os.path.join(args.image_path, gligen_caption_prompt["file_path"])) #inpaint_image
            image_size = 512
            input_image = transform_image(image, image_size)
            original_image_size = image.size
            background_filename = gligen_caption_prompt["file_path"]
            
            boxes = [anno['bbox'] for anno in gligen_caption_prompt["annos"]]
            #gligen_masks = [Image.open(anno['mask']) for anno in gligen_caption_prompt["annos"]]
            gligen_phrases = [anno['caption'] for anno in gligen_caption_prompt["annos"]]
            gligen_categories = [phrase.split(' ')[-1] for phrase in gligen_phrases] #pidray
            
            #vis_img_paths = []
            #vis_img_paths.append(os.path.join(args.image_path, background_filename)) #原图
            #vis_img_paths.append(os.path.join(args.depth_path, background_filename)) #原深度图
            prompt = gligen_caption_prompt['captions']

            # if len(args.alpha) > 1:
            #     vis_img_paths = [[] for _ in  range(len(args.alpha))]
            #     for vis_img_path in vis_img_paths:
            #         vis_img_path.append(os.path.join(args.image_path, background_filename)) #原图
            # else:
            vis_img_paths = []
            vis_img_paths.append(os.path.join(args.image_path, background_filename)) #原图
            
            #pdb.set_trace()

            
            #--------------------------------- 生成方式一： 固定位置+原类别 -----------------------------------#

            if args.gen_method == 1:
            
                anno_boxes = []
                vis_boxes = []
                categories, phrases = [], []
                hidden_boxes = []
                
                accept = False
                for box, category, phrase in zip(boxes, gligen_categories, gligen_phrases):
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    #valid, (new_x1, new_y1, new_x2, new_y2) = recalculate_box_and_verify_if_valid(x1, y1, w, h, image_size, original_image_size,0.005) # hixray
                    #valid, (new_x1, new_y1, new_x2, new_y2) = recalculate_box_and_verify_if_valid(x1, y1, w, h, image_size, original_image_size,0.004) # opixray
                    valid, (new_x1, new_y1, new_x2, new_y2) = recalculate_box_and_verify_if_valid(x1, y1, w, h, image_size, original_image_size) #pidray
                    if valid:
                        accept = True
                        vis_boxes.append([new_x1, new_y1, new_x2, new_y2])
                        hidden_boxes.append([new_x1 / image_size, new_y1 / image_size, new_x2 / image_size, new_y2 / image_size])
                        new_w, new_h = new_x2 - new_x1, new_y2 - new_y1
                        anno_boxes.append([new_x1, new_y1, new_w, new_h])
                        categories.append(category)
                        phrases.append(phrase)
                
                if not accept:
                    continue
                
                gligen_boxes = [[x / image_size for x in box] for box in vis_boxes]
                
                caption = make_a_sentence(categories)
                
                filename_prefix = os.path.join(args.output_path, f"{background_filename}_{caption}")
                filename = f"{background_filename}_{caption}"
                filename += '.png'
                
                current_filenames = os.listdir(args.output_path)
                if filename in current_filenames:
                    continue
            
        
            #------------------------------------------------------------------------------#
            
            #--------------------------------- 生成方式二：固定位置+随机类别 -----------------------------------#

            elif args.gen_method == 2:
            
                anno_boxes = []
                vis_boxes = []
                categories, phrases = [], []
                accept = False
                hidden_categories = []
                for box, category in zip(boxes, gligen_categories):
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    valid, (new_x1, new_y1, new_x2, new_y2) = recalculate_box_and_verify_if_valid(x1, y1, w, h, image_size, original_image_size)
                    if valid:
                        accept = True
                        vis_boxes.append([new_x1, new_y1, new_x2, new_y2])
                        new_w, new_h = new_x2 - new_x1, new_y2 - new_y1
                        anno_boxes.append([new_x1, new_y1, new_w, new_h])
                        if not args.gen2_hidden:
                            hidden_categories.append(category)
                            if category in group1_categories:
                                category = np.random.choice(group1_categories)
                            elif category in group2_categories:
                                #p = [0.5, 0.5, 0,0,0,0,0,0,0]
                                category = np.random.choice(group2_categories, p=None)
                            else:
                                category = np.random.choice(group3_categories)
                            categories.append(category)
                            phrases.append(f'one {category}')
                
                if not accept:
                    continue

                if args.gen2_hidden:
                    gen2_annotation = get_anno_from_img_name(background_filename, coco)
                    anno_categories_id = [anno['category_id'] for anno in gen2_annotation]
                    categories = [coco.loadCats(cat_id)[0]['name'] for cat_id in anno_categories_id]
                    phrases = [f'one {category}' for category in categories]

                
                gligen_boxes = [[x / image_size for x in box] for box in vis_boxes]
                
                caption = make_a_sentence(categories)
                
                filename_prefix = os.path.join(args.output_path, f"{background_filename}_{caption}")
                filename = f"{background_filename}_{caption}"
                filename += '.png'
                
                current_filenames = os.listdir(args.output_path)
                if filename in current_filenames:
                    continue
            
            #------------------------------------------------------------------------------#
            
            #--------------------------------- 生成方式三：随机位置(SAM) + 随机类别 -----------------------------------#

            elif args.gen_method == 3:

                if not args.gen3_hidden:
            
                    masks = mask_generator.generate(np.array(image))
                    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=False) #从小到大排序 
                    masks = sorted_masks[:-2] # filter background & baggage
                    all_boxes = [mask['bbox'] for mask in masks]
                    all_masks = [mask['segmentation'] for mask in masks]
                    selected_boxes, selected_masks = select_box(all_boxes, boxes, all_masks, args.sam_box_iou_thre, min_area_thre=1050, max_area_thre=60000) #opixray
                    #selected_boxes, selected_masks = select_box(all_boxes, boxes, all_masks, args.sam_box_iou_thre) # pidray
                    if selected_boxes is None:
                        continue
                    
                    candidate_idx = list(range(len(selected_boxes)))
                    valid = False

                    while len(candidate_idx) > 0:
                        #idx = random.randint(0, len(selected_boxes) - 1)
                        idx = random.choice(candidate_idx)
                        selected_box = selected_boxes[idx]
                        selected_mask = selected_masks[idx]
                        x1, y1, x2, y2 = selected_box
                        w = x2 - x1
                        h = y2 - y1
                        valid, (new_x1, new_y1, new_x2, new_y2) = recalculate_box_and_verify_if_valid(x1, y1, w, h, image_size, original_image_size, min_box_size=0.004) # opixray
                        #valid, (new_x1, new_y1, new_x2, new_y2) = recalculate_box_and_verify_if_valid(x1, y1, w, h, image_size, original_image_size, min_box_size=0.001) # pidray
                        valid_selected_box = [new_x1, new_y1, new_x2, new_y2]
                        candidate_idx.remove(idx)
                        if valid or len(candidate_idx) == 0:
                            break

                    if not valid:
                        continue

                    category_group = select_category(valid_selected_box, args.category_group_area_boudary)
                    category = np.random.choice(group_categories[category_group], p=None)
                    
                    boxes.append(selected_box)
                    gligen_categories.append(category)
                    gligen_phrases.append(f'one {category}')
                    
                    anno_boxes = []
                    vis_boxes = []
                    categories = []
                    hidden_boxes = []
                    
                    accept = False
                    for box, category in zip(boxes, gligen_categories):
                        x1, y1, x2, y2 = box
                        w = x2 - x1
                        h = y2 - y1
                        valid, (new_x1, new_y1, new_x2, new_y2) = recalculate_box_and_verify_if_valid(x1, y1, w, h, image_size, original_image_size, min_box_size=0.004) # opixray
                        #valid, (new_x1, new_y1, new_x2, new_y2) = recalculate_box_and_verify_if_valid(x1, y1, w, h, image_size, original_image_size,  min_box_size=0.001) # pidray
                        if valid:
                            accept = True
                            vis_boxes.append([new_x1, new_y1, new_x2, new_y2])
                            hidden_boxes.append([new_x1 / image_size, new_y1 / image_size, new_x2 / image_size, new_y2 / image_size])
                            new_w, new_h = new_x2 - new_x1, new_y2 - new_y1
                            anno_boxes.append([new_x1, new_y1, new_w, new_h])
                            categories.append(category)
                    
                    if not accept:
                        continue

                else:
                    hidden_boxes = []
                    gen3_annotation = get_anno_from_img_name(background_filename, coco)
                    if gen3_annotation is None:
                        continue
                    anno_categories_id = [anno['category_id'] for anno in gen3_annotation]
                    anno_boxes = [anno['bbox'] for anno in gen3_annotation]
                    vis_boxes = []

                    for anno_box in anno_boxes:
                        x,y,w,h = anno_box
                        x1, y1, x2, y2 = x, y, x+w, y+h
                        vis_boxes.append([x1, y1, x2, y2])
                        if (x2 - x1) * (y2 - y1) / (image_size * image_size) > 0.01:
                            hidden_boxes.append([x1 / image_size, y1 / image_size, x2 / image_size, y2 / image_size])

                    boxes.append(vis_boxes[-1])
                    valid_selected_box = vis_boxes[-1]
                    x1, y1, x2, y2 = valid_selected_box
                    if (x2 - x1) * (y2 - y1) / (image_size * image_size) > 0.01:
                        hidden_boxes.append([x / image_size for x in valid_selected_box])
                    categories = [coco.loadCats(cat_id)[0]['name'] for cat_id in anno_categories_id]
                    category = categories[-1]
                    
                gligen_boxes = [[x / image_size for x in valid_selected_box]]
                
                phrases = [f'one {category}']
                caption = make_a_sentence([category])
                
                filename_prefix = os.path.join(args.output_path, f"{background_filename}_{caption}")
                filename = f"{background_filename}_{caption}"
                filename += '.png'
                
                current_filenames = os.listdir(args.output_path)
                if filename in current_filenames:
                    continue
            
            #------------------------------------------------------------------------------#
            
            #--------------------------------- 生成方式四：3+1 -----------------------------------#

            elif args.gen_method == 4:
                # anno_boxes = []
                # vis_boxes = []
                # categories, phrases = [], []
                hidden_boxes = []

                # for prob syn-1
                gen_vis_boxes = []
                gen_anno_boxes = []
                gen_categories, gen_phrases = [], []

                ori_vis_boxes = []
                ori_anno_boxes = []
                ori_categories = []
                '''
                masks = mask_generator.generate(np.array(image))
                sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=False) #从小到大排序 
                masks = sorted_masks[:-2] # filter background & baggage
                all_boxes = [mask['bbox'] for mask in masks]
                all_masks = [mask['segmentation'] for mask in masks]
                selected_boxes, selected_masks = select_box(all_boxes, boxes, all_masks, args.sam_box_iou_thre, min_area_thre=1050, max_area_thre=60000) #opixray
                #selected_boxes, selected_masks = select_box(all_boxes, boxes, all_masks, args.sam_box_iou_thre) # pidray
                #selected_boxes, selected_masks = select_box(all_boxes, boxes, all_masks, args.sam_box_iou_thre, min_area_thre=1400, max_area_thre=290000)#hixray
                
                valid = False
                if selected_boxes is not None:
                    candidate_idx = list(range(len(selected_boxes)))

                    while len(candidate_idx) > 0:
                        #idx = random.randint(0, len(selected_boxes) - 1)
                        idx = random.choice(candidate_idx)
                        selected_box = selected_boxes[idx]
                        selected_mask = selected_masks[idx]
                        x1, y1, x2, y2 = selected_box
                        w = x2 - x1
                        h = y2 - y1
                        #valid, (new_x1, new_y1, new_x2, new_y2) = recalculate_box_and_verify_if_valid(x1, y1, w, h, image_size, original_image_size, min_box_size=0.004) # opixray
                        #valid, (new_x1, new_y1, new_x2, new_y2) = recalculate_box_and_verify_if_valid(x1, y1, w, h, image_size, original_image_size, min_box_size=0.01) # pidray
                        valid, (new_x1, new_y1, new_x2, new_y2) = recalculate_box_and_verify_if_valid(x1, y1, w, h, image_size, original_image_size, 0.005) #hixray
                        valid_selected_box = [new_x1, new_y1, new_x2, new_y2]
                        candidate_idx.remove(idx)
                        if valid or len(candidate_idx) == 0:
                            break
                
                if valid:
                    category_group = select_category(valid_selected_box, args.category_group_area_boudary)
                    category = np.random.choice(group_categories[category_group], p=None)
                    gen_categories.append(category)
                    gen_phrases.append(f'one {category}')
                    new_x1, new_y1, new_x2, new_y2 = valid_selected_box
                    new_w, new_h = new_x2 - new_x1, new_y2 - new_y1
                    gen_anno_boxes.append([new_x1, new_y1, new_w, new_h])
                    #boxes.append(selected_box)
                    hidden_boxes.append([new_x1 / image_size, new_y1 / image_size, new_x2 / image_size, new_y2 / image_size])
                    gen_vis_boxes.append(valid_selected_box)
                
                else:
                    continue
                '''
                
                gen3_annotation = get_anno_from_img_name(background_filename, coco)
                if gen3_annotation is not None:

                    anno_categories_id = [anno['category_id'] for anno in gen3_annotation]
                    gen3_anno_boxes = [anno['bbox'] for anno in gen3_annotation]
                    gen3_vis_boxes = []
                    for anno_box in gen3_anno_boxes:
                        x,y,w,h = anno_box
                        x1, y1, x2, y2 = x, y, x+w, y+h
                        gen3_vis_boxes.append([x1, y1, x2, y2])

                    valid_selected_box = gen3_vis_boxes[-1]
                    gen3_categories = [coco.loadCats(cat_id)[0]['name'] for cat_id in anno_categories_id]
                    category = gen3_categories[-1]
                    
                    new_x1, new_y1, new_x2, new_y2 = valid_selected_box
                    new_w, new_h = new_x2 - new_x1, new_y2 - new_y1

                    hidden_boxes.append([new_x1 / image_size, new_y1 / image_size, new_x2 / image_size, new_y2 / image_size])

                    # anno_boxes.append([new_x1, new_y1, new_w, new_h])
                    # vis_boxes.append(valid_selected_box)
                    # categories.append(category)
                    # phrases.append(f'one {category}')

                    gen_anno_boxes.append([new_x1, new_y1, new_w, new_h])
                    gen_vis_boxes.append(valid_selected_box)
                    gen_categories.append(category)
                    gen_phrases.append(f'one {category}')
                
                else:
                    continue
                
        
                accept = False
                syn_1_prob = 0.3

                # 1. divide by group
                group_boxes, group_gligen_categories, group_gligen_phrases = [], [], []
                group1_boxes, group1_gligen_categories, group1_gligen_phrases = [], [], []
                group2_boxes, group2_gligen_categories, group2_gligen_phrases = [], [], []
                group3_boxes, group3_gligen_categories, group3_gligen_phrases = [], [], []

                for box, category, phrase in zip(boxes, gligen_categories, gligen_phrases):
                    if category in group1_categories:
                        group1_boxes.append(box)
                        group1_gligen_categories.append(category)
                        group1_gligen_phrases.append(phrase)
                    elif category in group2_categories:
                        group2_boxes.append(box)
                        group2_gligen_categories.append(category)
                        group2_gligen_phrases.append(phrase)
                    else:
                        group3_boxes.append(box)
                        group3_gligen_categories.append(category)
                        group3_gligen_phrases.append(phrase)
                
                group_boxes = group1_boxes + group2_boxes + group3_boxes
                # 2.shuffle by group
                if len(group1_gligen_categories) > 0:
                    idx = list(range(len(group1_gligen_categories)))
                    random.shuffle(idx)
                    group1_gligen_categories = [group1_gligen_categories[idx_] for idx_ in idx]
                    group1_gligen_phrases = [group1_gligen_phrases[idx_] for idx_ in idx]
                
                if len(group2_gligen_categories) > 0:
                    idx = list(range(len(group2_gligen_categories)))
                    random.shuffle(idx)
                    group2_gligen_categories = [group2_gligen_categories[idx_] for idx_ in idx]
                    group2_gligen_phrases = [group2_gligen_phrases[idx_] for idx_ in idx]

                if len(group3_gligen_categories) > 0:
                    idx = list(range(len(group3_gligen_categories)))
                    random.shuffle(idx)
                    group3_gligen_categories = [group3_gligen_categories[idx_] for idx_ in idx]
                    group3_gligen_phrases = [group3_gligen_phrases[idx_] for idx_ in idx]

                group_gligen_categories = group1_gligen_categories + group2_gligen_categories + group3_gligen_categories
                group_gligen_phrases = group1_gligen_phrases + group2_gligen_phrases + group3_gligen_phrases

                for box, category, phrase in zip(group_boxes, group_gligen_categories, group_gligen_phrases):
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    #valid, (new_x1, new_y1, new_x2, new_y2) = recalculate_box_and_verify_if_valid(x1, y1, w, h, image_size, original_image_size,0.004) # opixray
                    valid, (new_x1, new_y1, new_x2, new_y2) = recalculate_box_and_verify_if_valid(x1, y1, w, h, image_size, original_image_size, 0.001) #pidray
                    #valid, (new_x1, new_y1, new_x2, new_y2) = recalculate_box_and_verify_if_valid(x1, y1, w, h, image_size, original_image_size, 0.005) #hixray
                    if valid:
                        accept = True
                        hidden_boxes.append([new_x1 / image_size, new_y1 / image_size, new_x2 / image_size, new_y2 / image_size])
                        # vis_boxes.append([new_x1, new_y1, new_x2, new_y2])
                        # new_w, new_h = new_x2 - new_x1, new_y2 - new_y1
                        # anno_boxes.append([new_x1, new_y1, new_w, new_h])
                        # categories.append(category)
                        # phrases.append(phrase)
                        new_w, new_h = new_x2 - new_x1, new_y2 - new_y1
                        # prob = random.random()
                        # if prob < syn_1_prob and category in args.hard_categories:
                        #     gen_vis_boxes.append([new_x1, new_y1, new_x2, new_y2])
                        #     gen_anno_boxes.append([new_x1, new_y1, new_w, new_h])
                        #     gen_categories.append(category)
                        #     gen_phrases.append(phrase)
                        # else:
                        #     ori_vis_boxes.append([new_x1, new_y1, new_x2, new_y2])
                        #     ori_anno_boxes.append([new_x1, new_y1, new_w, new_h])
                        #     ori_categories.append(category)
                        gen_vis_boxes.append([new_x1, new_y1, new_x2, new_y2])
                        gen_anno_boxes.append([new_x1, new_y1, new_w, new_h])
                        gen_categories.append(category)
                        gen_phrases.append(phrase)
                
                anno_boxes = gen_anno_boxes + ori_anno_boxes
                
                if not accept:
                   boxes = [valid_selected_box]
                else:
                   boxes = [valid_selected_box] + boxes


                # gligen_boxes = [[x / image_size for x in box] for box in vis_boxes]
                # caption = make_a_sentence(categories)

                gligen_boxes = [[x / image_size for x in box] for box in gen_vis_boxes]
                caption = make_a_sentence(gen_categories)
                
                filename_prefix = os.path.join(args.output_path, f"{background_filename}_{caption}")
                filename = f"{background_filename}_{caption}"
                filename += '.png'
                
                current_filenames = os.listdir(args.output_path)
                if filename in current_filenames:
                    continue
            
            #------------------------------------------------------------------------------#
          if args.inpaint:
            meta = dict(
                input_image = input_image,
                prompt = caption,
                images = None,
                locations = gligen_boxes,
                phrases = phrases,
            )
          else:
            meta = dict(
                prompt = caption,
                images = None,
                locations = gligen_boxes,
                phrases = phrases,
            )

          batch = prepare_batch(meta, device=device)
          
          #--------------------------------------------------------------------------------#
          
          
          context = text_encoder.encode([meta["prompt"]]*config.batch_size)
          uc = text_encoder.encode( config.batch_size*[""] )
          if args.negative_prompt is not None:
              uc = text_encoder.encode( config.batch_size*[args.negative_prompt] )


          # - - - - - sampler - - - - - # 
          if config.no_plms:
              sampler = DDIMSampler(diffusion, model)#, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
              steps = 50 
          else:
              sampler = PLMSSampler(diffusion, model)#, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
              steps = 50 


          # - - - - - inpainting related - - - - - #
          inpainting_mask = z0 = None  # used for replacing known region in diffusion process
          inpainting_extra_input = None # used as model input 
          if "input_image" in meta:
              # inpaint mode 
              #assert config.inpaint_mode, 'input_image is given, the ckpt must be the inpaint model, are you using the correct ckpt?'
            
              inpainting_mask = draw_masks_from_boxes( batch['boxes'], model.image_size  ).to(device)
              #inpainting_mask = draw_masks_from_boxes( batch['boxes'], image_size // 8  ).to(device)
              '''
              if args.scratch_generate:
                Input_image = F.pil_to_tensor( meta["input_image"].convert("RGB").resize((512,512)) ) 
              else:
                Input_image = F.pil_to_tensor( meta["input_image"].convert("RGB"))
              Input_image = ( Input_image.float().unsqueeze(0).to(device) / 255 - 0.5 ) / 0.5
              '''
              Input_image = meta['input_image'].float().unsqueeze(0).to(device)
              z0 = autoencoder.encode( Input_image )
            
              masked_z = z0*inpainting_mask
              inpainting_extra_input = torch.cat([masked_z,inpainting_mask], dim=1)              
        

          # - - - - - input for gligen - - - - - #
          grounding_input = grounding_tokenizer_input.prepare(batch)
          grounding_extra_input = None
          if grounding_downsampler_input != None:
              grounding_extra_input = grounding_downsampler_input.prepare(batch)
          
          #if args.scratch_generate:
          starting_noise = torch.randn(config.batch_size, 4, 64, 64).to(device)
          #starting_noise = torch.randn(config.batch_size, 4, image_size // 8, image_size // 8).to(device)
          '''
          else:
            _t = torch.rand(z0.shape[0]).to(z0.device)
            t = (torch.pow(_t, 1) * 1000).long()
            t = torch.where(t!=1000, t, 999) # if 1000, then replace it with 999
            noise = torch.randn_like(z0)
            starting_noise = diffusion.q_sample(x_start=z0, t=t, noise=noise)
          '''
            
          input = dict(
                    x = starting_noise, 
                    timesteps = None, 
                    context = context, 
                    grounding_input = grounding_input,
                    inpainting_extra_input = inpainting_extra_input,
                    grounding_extra_input = grounding_extra_input,
            )

          # - - - - - start sampling - - - - - #
          shape = (config.batch_size, model.in_channels, model.image_size, model.image_size)
          #shape = (config.batch_size, model.in_channels, image_size // 8, image_size // 8)

          controller = AttentionStore()

          #token_indices, input_boxes = get_indices_boxes_to_alter(tokenizer, caption, vis_boxes)
          #args.token_indices = token_indices

          #args.boxdiff_config = RunConfig()
          #args.vis_boxes = input_boxes
          

          #do_hidden = True if args.latent_redist and (len(anno_boxes) == 1 or args.gen_method == 3) else False
          do_hidden = True if args.latent_redist else False
          
          hidden_info = None
          
          if do_hidden and not args.use_sam:
              do_hidden = False  # BOM requires SAM; skip if SAM disabled

          if do_hidden:
              hidden_info = {}
              hidden_info['hidden_bg_box'] = None
              #fg_area = anno_boxes[0][2] * anno_boxes[0][3]
              if args.occlusion_method == 3:
                  hidden_bg_box = None
                  masks = mask_generator.generate(np.array(image))
                  sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=False) #从小到大排序 
                  masks = sorted_masks[:-2] # filter background & baggage
                  all_boxes = [mask['bbox'] for mask in masks]
                  all_masks = [mask['segmentation'] for mask in masks]
                  #selected_boxes, selected_masks = select_box(all_boxes, boxes, all_masks, args.sam_box_iou_thre, min_area_thre=1050, max_area_thre=60000) # opixray
                  selected_boxes, selected_masks = select_box(all_boxes, boxes, all_masks, args.sam_box_iou_thre) # pidray
                  if selected_boxes is None:
                      do_hidden = False
                  else:
                    new_selected_boxes, new_selected_masks = [], []
                    for selected_box, selected_mask in zip(selected_boxes, selected_masks):
                        x1, y1, x2, y2 = selected_box
                        w = x2 - x1
                        h = y2 - y1
                        #valid, (new_x1, new_y1, new_x2, new_y2) = recalculate_box_and_verify_if_valid(x1, y1, w, h, image_size, original_image_size, min_box_size=0.004) #opiray
                        valid, (new_x1, new_y1, new_x2, new_y2) = recalculate_box_and_verify_if_valid(x1, y1, w, h, image_size, original_image_size, min_box_size=0.001) #pidray
                        #valid, (new_x1, new_y1, new_x2, new_y2) = recalculate_box_and_verify_if_valid(x1, y1, w, h, image_size, original_image_size, min_box_size=0.005) #hixray
                        if valid:
                            new_selected_boxes.append([new_x1 / image_size, new_y1 / image_size, new_x2 / image_size, new_y2 / image_size])
                            selected_mask = cv2.resize(selected_mask.astype(np.uint8), (64, 64)).astype(np.uint8)
                            selected_mask //= 255
                            selected_mask = torch.from_numpy(selected_mask)
                            new_selected_masks.append(selected_mask)
                    if len(new_selected_boxes) == 0 or len(hidden_boxes) == 0:
                        do_hidden = False
                    else:
                        hidden_info['hidden_bg_boxs'] = []
                        hidden_info['hidden_bg_masks'] = []
                        # full_same_bg_hidden
                        idx = random.randint(0, len(new_selected_boxes) - 1)
                        #for _ in range(len(anno_boxes)):
                        for _ in range(len(hidden_boxes)):
                            # full_diff_bg_hidden
                            #idx = random.randint(0, len(new_selected_boxes) - 1)
                            selected_box = new_selected_boxes[idx]
                            selected_mask = new_selected_masks[idx]
                            hidden_info['hidden_bg_boxs'].append(selected_box)
                            hidden_info['hidden_bg_masks'].append(selected_mask)
                        
                        hidden_info['hidden_fg_boxes'] = hidden_boxes
                        print('hidden3')
            
          
          if do_hidden and args.hidden_time == 'after':
            #samples_fake, samples_hidden = sampler.sample(S=steps, shape=shape, input=input,  uc=uc, guidance_scale=config.guidance_scale, mask=inpainting_mask, x0=z0, controller=controller, args=args, boxes=batch['boxes'], do_hidden=do_hidden, hidden_info=hidden_info)
            samples_fake, samples_hiddens = sampler.sample(S=steps, shape=shape, input=input,  uc=uc, guidance_scale=config.guidance_scale, mask=inpainting_mask, x0=z0, controller=controller, args=args, boxes=batch['boxes'], do_hidden=do_hidden, hidden_info=hidden_info, inpaint=args.inpaint)
          else:
            samples_fake = sampler.sample(S=steps, shape=shape, input=input,  uc=uc, guidance_scale=config.guidance_scale, mask=inpainting_mask, x0=z0, controller=controller, args=args, boxes=batch['boxes'], do_hidden=do_hidden, hidden_info=hidden_info, inpaint=args.inpaint)
          
          cross_attention_maps = None
          if args.refine_anno:
              cross_attention_maps = show_cross_attention(controller, from_where=('up', 'down'), select=0, prompts=[caption], tokenizer=tokenizer)

          #print('ca_yes')
          
          if args.do_decode:
            '''
            if do_hidden and args.hidden_time == 'after':
                samples_hidden = autoencoder.decode(samples_hidden)
                # - - - - - save - - - - - #
                sample_hidden = samples_hidden[0]
                sample_hidden = torch.clamp(sample_hidden, min=-1, max=1) * 0.5 + 0.5
                sample_hidden = sample_hidden.cpu().numpy().transpose(1,2,0) * 255 
                sample_hidden = Image.fromarray(sample_hidden.astype(np.uint8))
                sample_hidden.save(filename_prefix + '.png')
            '''
            
            # for different alpha
            if do_hidden and args.hidden_time == 'after':
                for hidden_idx in range(len(args.alpha)):
                    samples_hidden = samples_hiddens[hidden_idx]
                    samples_hidden = autoencoder.decode(samples_hidden)
                    # - - - - - save - - - - - #
                    sample_hidden = samples_hidden[0]
                    sample_hidden = torch.clamp(sample_hidden, min=-1, max=1) * 0.5 + 0.5
                    sample_hidden = sample_hidden.cpu().numpy().transpose(1,2,0) * 255 
                    sample_hidden = Image.fromarray(sample_hidden.astype(np.uint8))
                    sample_hidden.save(os.path.join(args.output_path + str(args.alpha[hidden_idx]), filename))
                    #vis_img_paths[hidden_idx].append(os.path.join(args.output_path + str(args.alpha[hidden_idx]), filename))
                    #vis_img_paths[hidden_idx].append(os.path.join(args.output_path + str(args.alpha[hidden_idx]), filename))
            
            # for ori
            samples_fake_decode = autoencoder.decode(samples_fake)
            # - - - - - save - - - - - #
            sample = samples_fake_decode[0]
            sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
            sample = sample.cpu().numpy().transpose(1,2,0) * 255 
            sample = Image.fromarray(sample.astype(np.uint8))
            os.makedirs(os.path.join(args.output_path, 'ori'), exist_ok=True)
            sample.save(os.path.join(args.output_path + 'ori', filename))
            # if do_hidden is False:
            #     for hidden_idx in range(len(args.alpha)):
            #         sample.save(os.path.join(args.output_path + str(args.alpha[hidden_idx]), filename))
            '''
            #if args.occlusion_method == 3:
            if (not do_hidden) or args.hidden_time == 'during':
                # if len(args.alpha) > 1:
                #     for hidden_idx in range(len(args.alpha)):
                #         os.makedirs(os.path.join(args.output_path, str(args.alpha[hidden_idx])), exist_ok=True)
                #         sample.save(os.path.join(args.output_path + str(args.alpha[hidden_idx]), filename))
                # else:
                    
                sample.save(filename_prefix + '.png')
            '''
          
          if args.scratch_generate:
            #for visualization
            vis_img_paths.append(filename_prefix + '.png')
            vis_img_paths.append(filename_prefix + '.png')
          
          
          ori_img = np.array(sample)
          refine_points = None

          if args.refine_anno:
            #sampling_info['topk'] = args.topks[0]
            #sampling_info['range_num'] = args.range_nums[0]
            #sampling_info['h_sampling_strategies'] = args.h_sampling_strategies
            # for ablation_idx in range(len(args.ablation_opt)):
            #     ablation = args.ablation_opt[ablation_idx]
            #     refine_f = f_list[ablation_idx]
            #     if ablation == 'only_box':
            #         args.refine_strategies = 'only_box'
            #     elif ablation == 'range4':
            #         args.refine_strategies = 'h_sampling'

            for ablation_idx in range(len(args.ablation_opt)):
                ablation = args.ablation_opt[ablation_idx]
                refine_f = f_list[ablation_idx]
                if ablation == 'only_box': # range_num = 0
                    refine_strategies = None
                else:
                    refine_strategies = args.refine_strategies
                    sampling_info['h_sampling_strategies'] = args.h_sampling_strategies
                    sampling_info['with_sam'] = True
                    if ablation == 'range1':
                        sampling_info['range_num'] = 1
                    elif ablation == 'range2':
                        sampling_info['range_num'] = 2
                    elif ablation == 'range3':
                        sampling_info['range_num'] = 3
                    elif ablation == 'range4':
                        sampling_info['range_num'] = 4
                    elif ablation == 'sam':
                        sampling_info['range_num'] = 4
                        sampling_info['with_sam'] = False
                    elif ablation == 'top16':
                        refine_strategies = 'topk_sampling'
                        sampling_info['topk'] = 16
                    elif ablation == 'mode':
                        sampling_info['h_sampling_strategies'] = 'mode_convex'

                if args.gen_method != 3:
                    refine_anno_boxes, refine_points, refine_masks = refine_anno(ori_img, anno_boxes, cross_attention_maps, predictor, sampling_info, args.refine_strategies)
                    #refine_anno_boxes, refine_points, refine_masks = refine_anno(ori_img, gen_anno_boxes, cross_attention_maps, predictor, sampling_info, args.refine_strategies)
                else:
                    refine_selected_boxes, refine_points, _ = refine_anno(ori_img, [anno_boxes[-1]], cross_attention_maps, predictor, sampling_info, refine_strategies)
                    refine_selected_box = refine_selected_boxes[0]
                    refine_anno_boxes = []
                    for box in anno_boxes[:-1]:
                        refine_anno_boxes.append(box)
                    refine_anno_boxes.append(refine_selected_box)
            
                vis_save_path = args.vis_path
                ca_vis_path = args.ca_vis_path

                refine_vis_boxes = []
                for refine_anno_box in refine_anno_boxes:
                    x, y, w, h = refine_anno_box
                    x1, y1, x2, y2 = x, y, x+w, y+h
                    refine_vis_boxes.append([x1, y1, x2, y2])
                # refine_gen_vis_boxes = []
                # for refine_gen_anno_box in refine_gen_anno_boxes:
                #     x, y, w, h = refine_gen_anno_box
                #     x1, y1, x2, y2 = x, y, x+w, y+h
                #     refine_gen_vis_boxes.append([x1, y1, x2, y2])
                
                vis_categories = categories if args.gen_method !=3 else [category]
                #vis_categories = gen_categories
                
                for ca_map, category in zip(cross_attention_maps, vis_categories):
                    ca_images = []
                    ca_images.append(sample)
                    ca_images.append(Image.fromarray(ca_map))
                    if args.scratch_generate:
                        save_filename = os.path.join(ca_vis_path, filename.split('.')[0] + '_' + category + '.png')
                        if os.path.exists(save_filename):
                            save_filename = save_filename[:-4] + '_' + str(np.random.randn()) + '.png'
                        save_visualization(ca_images, save_filename, refine_points)
                
                
                # os.makedirs(os.path.join(vis_save_path, ablation), exist_ok=True)
                # new_vis_save_path = os.path.join(vis_save_path, ablation)
                # save_visualization_and_correspondence(args, new_vis_save_path, vis_img_paths, boxes, vis_boxes, refine_vis_boxes, categories, refine_points=refine_points, correspondence_data=None)
                
                #save_visualization_and_correspondence(args, vis_save_path,  vis_img_paths, boxes, vis_boxes, refine_vis_boxes, categories, refine_points=refine_points, correspondence_data=None)
                
                #save_visualization_and_correspondence(args, vis_save_path,  vis_img_paths, boxes, gen_vis_boxes, refine_gen_vis_boxes, categories, refine_points=refine_points, correspondence_data=None)
                
                #categories = gen_categories + ori_categories
                #refine_anno_boxes = refine_gen_anno_boxes + ori_anno_boxes
                
                refine_f.write(filename)
                for (gligen_category, bbox) in zip(categories, refine_anno_boxes):
                    gligen_category_id = categories2id[gligen_category]
                    refine_f.write('\t' + str(gligen_category_id))
                    for coord in bbox:
                        refine_f.write('\t' + str(coord))
                refine_f.write('\n')
                #print(f'I am here')
        
          f.write(filename)
          for (gligen_category, bbox) in zip(categories, anno_boxes):
              gligen_category_id = categories2id[gligen_category]
              f.write('\t' + str(gligen_category_id))
              for coord in bbox:
                  f.write('\t' + str(coord))
          f.write('\n')
          print(f'I am here')

        
    # 保存对应关系文件
    # with open(os.path.join(args.correspondence_path, f'{args.gpu}_correspondence.json'), 'w') as w_f:
    #   json.dump(correspondence_data, w_f, indent=4)
    dist_cleanup()
    f.close()

    if args.refine_anno:
        if len(f_list) > 0:
            for f in f_list:
                f.close()


def find_median_pixel(image, mask):
    # 提取mask区域的像素值和坐标
    masked_pixels = image[mask > 0]
    masked_coords = np.argwhere(mask > 0)

    # nonzero_mask = image > 0
    # new_mask = mask * nonzero_mask
    # masked_pixels = image[new_mask > 0]
    # masked_coords = np.argwhere(new_mask > 0)

    
    # 找到中位数
    # median_pixel_value = np.median(masked_pixels)
    # 获取数组元素的索引，这些索引对应于排序后的数组
    indices = np.argsort(masked_pixels)
    length = len(masked_pixels)

    # 检查数组长度是否为奇数
    if len(masked_pixels) % 2 == 1:
        # 如果是奇数，中位数是中间索引对应的值
        median_index =  length // 2
        median_pixel_value = masked_pixels[indices[median_index]]
    else:
        # 如果是偶数，从中间两个索引对应的值中随机选择一个
        mid_index1 = length // 2 - 1
        mid_index2 = length // 2
        
        median_candidates = masked_pixels[indices[mid_index1:mid_index2+1]]
        median_pixel_value = random.choice(median_candidates)
    
    # 找到中位数值在原始二维图像中的坐标
    median_index = np.where(masked_pixels == median_pixel_value)[0][0]
    median_coords = masked_coords[median_index]
    
    return median_pixel_value, median_coords

def find_min_pixel_out_mask(image, mask):
    bg_points = []

    try:
        # 提取mask区域以外的像素值和坐标
        mask_out_pixels = image[mask == 0]
        mask_out_coords = np.argwhere(mask == 0)
        min_pixel_value = np.min(mask_out_pixels)

        # 找到最小数值在原始二维图像中的坐标
        min_index = np.where(mask_out_pixels == min_pixel_value)[0][0]
        min_coords = mask_out_coords[min_index]

        bg_points.append(min_coords)

        return bg_points
    
    except:
        return bg_points

def recursive_divide(image, mask, n=1, medians=None):
    if medians is None:
        medians = []
    
    # 基本情况：如果已经达到划分次数或者mask区域为空，则返回
    if n == 0 or mask.sum() == 0:
        return medians
    
    # 找到中位数像素值和坐标
    median_pixel_value, median_coords = find_median_pixel(image, mask)
    
    # 保存中位数对应的坐标
    medians.append(median_coords)
    
    # 创建两个子mask，一个小于等于中位数，一个大于中位数
    mask_low = mask.copy()
    mask_low[image > median_pixel_value] = 0
    
    mask_high = mask.copy()
    mask_high[image < median_pixel_value] = 0
    
    # 递归地在两个子mask中继续划分
    recursive_divide(image, mask_low, n-1, medians)
    recursive_divide(image, mask_high, n-1, medians)
    
    return medians

def find_modes(array_1d):
    # 使用Counter来统计每个元素的出现次数
    counts = Counter(array_1d)
    # 找到最大出现次数
    max_count = max(counts.values())
    # 找出所有出现次数等于最大次数的元素
    modes = [key for key, count in counts.items() if count == max_count]
    return modes

def find_count(array_1d, value):
    return sum(1 for x in array_1d if x == value)


def find_median(array_1d):
    # 找到中位数
    # median_pixel_value = np.median(masked_pixels)
    # 获取数组元素的索引，这些索引对应于排序后的数组
    indices = np.argsort(array_1d)
    length = len(array_1d)

    # 检查数组长度是否为奇数
    if len(array_1d) % 2 == 1:
        # 如果是奇数，中位数是中间索引对应的值
        median_index =  length // 2
        median_pixel_value = array_1d[indices[median_index]]
    else:
        # 如果是偶数，选择出现次数较多的那一个
        mid_index1 = length // 2 - 1
        mid_index2 = length // 2
        count1 = find_count(array_1d, array_1d[mid_index1])
        count2 = find_count(array_1d, array_1d[mid_index2])
        if count1 < count2:
            median_pixel_value = array_1d[mid_index2]
        else:
            median_pixel_value = array_1d[mid_index1]
    
    return [median_pixel_value]


def find_convex_pixel(image, mask, method='mode'):
    masked_img = image * mask
    non_zero_items = masked_img[masked_img != 0]
    if method == 'mode':
        points = find_modes(non_zero_items)
    elif method == 'median':
        points = find_median(non_zero_items)

    coords = []
    values = []
    for point in points:
        point_y, point_x = np.where(masked_img == point)
        for y, x in zip(point_y, point_x):
            coords.append((x, y))
            values.append(image[y, x])
    coords = np.array(coords)
    values = np.array(values)
    
    total_values = sum(values)
    weights = values / total_values
    convex_combination_coords = np.sum(weights[:, np.newaxis] * coords, axis=0)
    rounded_convex_combination_coords = np.round(convex_combination_coords).astype(int)

    return rounded_convex_combination_coords


def refine_anno(ori_img, anno_boxes, cross_attention_maps, predictor, sampling_info, refine_strategy):

    refine_boxes = []
    refine_points = []
    refine_masks = []

    for anno_box, ca_map in zip(anno_boxes, cross_attention_maps):

        points, labels, refine_point = [], [], []

        x, y, w, h = anno_box
        x, y, w, h = int(x), int(y), int(w), int(h)
        start_x, start_y = x, y
        x1, y1, x2, y2 = x, y, x+w, y+h
        input_bbox = np.array([x1, y1, x2, y2])

        if refine_strategy == 'topk_sampling':
            area = h * w
            topk = min(sampling_info['topk'], area)

            # 方式1：选择框内前topk大个点
            # 选择区域并展平
            region = ca_map[y:y+h, x:x+w]
            region_flattened = region.flatten()
            # 找出这个区域中最大的topk个值
            topk_indices = np.argpartition(region_flattened, -topk)[-topk:]  # 使用argpartition来找到最大的k个值的索引
            # 将展平后的索引转换回二维数组中的坐标
            topk_coords = np.unravel_index(topk_indices, region.shape)
            for y, x in zip(topk_coords[0], topk_coords[1]):
                points.append([x+start_x, y+start_y])
                labels.append(1)
                refine_point.append([x+start_x, y+start_y])
        
        elif refine_strategy == 'h_sampling':
        #else:
            range_num = sampling_info['range_num']
            with_sam = sampling_info['with_sam']
            h_sampling_strategies = sampling_info['h_sampling_strategies']
            ca_map_seg = cv2.merge([ca_map, ca_map, ca_map])
            predictor.set_image(ca_map_seg)
            ca_masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_bbox[None, :],
                multimask_output=False, 
            )

            if h_sampling_strategies == 'median':
                if with_sam:

                    # original segmented by SAM
                    ca_mask = ca_masks[0].astype(np.uint8) * 255

                    # no SAM (ablation on SAM)
                    #ca_mask = np.zeros((512, 512)).astype(np.uint8) * 255
                    
                    bg_points = find_min_pixel_out_mask(ca_map[y:y+h, x:x+w], ca_mask[y:y+h, x:x+w])
                    if len(bg_points) > 0:
                        for bg_point in bg_points:
                            bg_y, bg_x = bg_point
                            points.append([bg_x + start_x, bg_y + start_y])
                            labels.append(0)
                            refine_point.append([bg_x + start_x, bg_y + start_y])
                    
                    #ca_mask[y:y+h, x:x+w] = 255 # for ablation use
                    anchor_points = recursive_divide(ca_map[y:y+h, x:x+w], ca_mask[y:y+h, x:x+w], n=range_num)
                    for anchor in anchor_points:
                        anchor_y, anchor_x = anchor
                        points.append([anchor_x + start_x, anchor_y + start_y])
                        labels.append(1)
                        refine_point.append([anchor_x + start_x, anchor_y + start_y])
                else:
                    # original segmented by SAM
                    #ca_mask = ca_masks[0].astype(np.uint8) * 255

                    # no SAM (ablation on SAM)
                    ca_mask = np.zeros((512, 512)).astype(np.uint8) * 255
                    
                    bg_points = find_min_pixel_out_mask(ca_map[y:y+h, x:x+w], ca_mask[y:y+h, x:x+w])
                    if len(bg_points) > 0:
                        for bg_point in bg_points:
                            bg_y, bg_x = bg_point
                            points.append([bg_x + start_x, bg_y + start_y])
                            labels.append(0)
                            refine_point.append([bg_x + start_x, bg_y + start_y])
                    
                    ca_mask[y:y+h, x:x+w] = 255 # for ablation use
                    anchor_points = recursive_divide(ca_map[y:y+h, x:x+w], ca_mask[y:y+h, x:x+w], n=range_num)
                    for anchor in anchor_points:
                        anchor_y, anchor_x = anchor
                        points.append([anchor_x + start_x, anchor_y + start_y])
                        labels.append(1)
                        refine_point.append([anchor_x + start_x, anchor_y + start_y])
            
            elif h_sampling_strategies == 'mode_convex':
                ca_mask = ca_masks[0].astype(np.uint8) * 255
                mode_convex_point_x, mode_convex_point_y = find_convex_pixel(ca_map[y:y+h, x:x+w], ca_mask[y:y+h, x:x+w], method='mode')
                points.append([mode_convex_point_x + start_x, mode_convex_point_y + start_y])
                labels.append(1)
                refine_point.append([mode_convex_point_x + start_x, mode_convex_point_y + start_y])
            
            # elif h_sampling_strategies == 'median_convex':
            #     convex_point_x, convex_point_y = find_convex_pixel(ca_map[y:y+h, x:x+w], ca_mask[y:y+h, x:x+w], method='median')
            #     points.append([convex_point_x + start_x, convex_point_y + start_y])
            #     labels.append(1)
            #     refine_point.append([convex_point_x + start_x, convex_point_y + start_y])
            
        predictor.set_image(ori_img)
        points = np.array(points)
        labels = np.array(labels)
        masks, _, _ = predictor.predict(
            point_coords=points if len(points) != 0 else None,
            point_labels=labels if len(labels) != 0 else None,
            box=input_bbox[None, :],
            multimask_output=False, 
        )

        mask = masks[0].astype(np.uint8) * 255

        refine_bbox = get_bbox_from_mask_img(mask)
        if refine_bbox is None:
            refine_bbox = anno_box

        refine_boxes.append(refine_bbox)
        refine_points.append(refine_point)

        refine_masks.append(mask)
    
    return refine_boxes, refine_points, refine_masks

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #DDP 配置
    parser.add_argument('--world_size', type=int, default = 1, help = 'number of distributed processes')#开启的进程数
    parser.add_argument('--dist_url', default = 'env://', help='url used to set up distributed training')
    parser.add_argument('--device', default = 'cuda', help = 'device') #训练设备类型

    # hixray
    '''
    parser.add_argument('--output_path', default = '/home/ct/data/sjl/gligen_official_hixray_inpaint/generated_images/text_box_180000/gen1_3_0.5%_filter_best_hard/data/') #
    parser.add_argument('--annotation_path', default = '/home/ct/data/sjl/gligen_official_hixray_inpaint/generated_images/text_box_180000/gen1_3_0.5%_filter_best_hard/annotation/')
    #parser.add_argument('--annotation_file', default = '/home/ct/data/sjl/gligen_official_inpaint/generated_images/pidray/low_resolution/text_box_180000/gen1_scar/annotation/gen1_scar_anno.json')
    parser.add_argument('--vis_path', default = '/home/ct/data/sjl/gligen_official_hixray_inpaint/generated_images/text_box_180000/gen1_3_0.5%_filter_best_hard/visualization/')
    parser.add_argument('--ca_vis_path', default = '/home/ct/data/sjl/gligen_official_hixray_inpaint/generated_images/text_box_180000/gen1_3_0.5%_filter_best_hard/visualization_ca_mp_16/')
    '''
    # opixray
    '''
    parser.add_argument('--output_path', default = '/home/ct/data/sjl/gligen_official_inpaint_opiray/generated_images/text_box_180000/gen3_ablation/data/') #
    parser.add_argument('--annotation_path', default = '/home/ct/data/sjl/gligen_official_inpaint_opiray/generated_images/text_box_180000/gen3_ablation/annotation/')
    parser.add_argument('--vis_path', default = '/home/ct/data/sjl/gligen_official_inpaint_opiray/generated_images/text_box_180000/gen3_ablation/visualization/')
    parser.add_argument('--ca_vis_path', default = '/home/ct/data/sjl/gligen_official_inpaint_opiray/generated_images/text_box_180000/gen3_ablation/visualization_ca_mp_16/')
    '''

    '''
    parser.add_argument('--output_path', default = '/home/ct/data/sjl/gligen_official_opixray/generated_images/text_box_180000/gligen/data/') #
    parser.add_argument('--annotation_path', default = '/home/ct/data/sjl/gligen_official_opixray/generated_images/text_box_180000/gligen/annotation/')
    parser.add_argument('--vis_path', default = '/home/ct/data/sjl/gligen_official_opixray/generated_images/text_box_180000/gligen/visualization/')
    parser.add_argument('--ca_vis_path', default = '/home/ct/data/sjl/gligen_official_opixray/generated_images/text_box_180000/gligen/visualization_ca_mp_16/')
    '''
    # pidray
    
    parser.add_argument('--output_path', default = '/home/ct/data/sjl/gligen_official/generated_images/text_box_180000/gligen_gen1/data/') #
    parser.add_argument('--annotation_path', default = '/home/ct/data/sjl/gligen_official/generated_images/text_box_180000/gligen_gen1/annotation/')
    parser.add_argument('--vis_path', default = '/home/ct/data/sjl/gligen_official/generated_images/text_box_180000/gligen_gen1/visualization/')
    parser.add_argument('--ca_vis_path', default = '/home/ct/data/sjl/gligen_official/generated_images/text_box_180000/gligen_gen1/visualization_ca_mp_16/')

    #parser.add_argument('--image_path', default = '/home/ct/data/lwz/dataset/OPIXray/train/train_image')
    #parser.add_argument('--image_path', default = '/home/ct/data/lwz/dataset/HiXray/train/train_image')
    parser.add_argument('--image_path', default = '/home/ct/data/lwz/dataset/PIDray/pidray_prohibited/train_imgs')
    parser.add_argument('--ckpt_path', default='/home/ct/data/sjl/gligen_official/text_box/tag01/checkpoint_00180001.pth')
    #parser.add_argument('--ckpt_path', default='/home/ct/data/sjl/gligen_official_inpaint/text_box_180000/tag01/checkpoint_00050000.pth')
    #parser.add_argument('--ckpt_path', default='/home/ct/data/sjl/gligen_official_hixray_inpaint/text_box/tag00/checkpoint_00050000.pth')
    #parser.add_argument('--ckpt_path', default='/home/ct/data/sjl/gligen_official_opixray_inpaint_ckpt/checkpoint_00050000.pth')
    #parser.add_argument('--gligen_caption_pt', default='/home/ct/data/sjl/diffusion/pidray_low_resolution/gligen/pidray_train_gligen_seg_image_project.pt')
    parser.add_argument('--gligen_caption_pt', default='/home/ct/data/sjl/diffusion/pidray_low_resolution/gligen/pidray_train_mask.pt')
    #parser.add_argument('--gligen_caption_pt', default='/home/ct/data/sjl/diffusion/pidray_low_resolution/gligen/pidray_train_full.pt')
    #parser.add_argument('--gligen_caption_pt', default='/home/ct/data/sjl/diffusion/hixray/gligen/hixray_train.pt')
    #parser.add_argument('--gligen_caption_pt', default='/home/ct/data/sjl/diffusion/opiray/gligen/opiray_train.pt')
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead. WARNING: I did not test the code yet")
    parser.add_argument("--batch_size",  default=1, help="")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    parser.add_argument("--negative_prompt", type=str,  default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help="")
    parser.add_argument('--sam_weight', default = 'sam_vit_h_4b8939.pth')
    parser.add_argument('--ref_path', default = '', help='directory of reference foreground images (Xsyn-A only)')
    parser.add_argument('--sam_model_type', default = 'vit_h')
    parser.add_argument('--sam_box_iou_thre', default = 0.2)
    parser.add_argument('--use_sam', default = True)
    parser.add_argument('--category_group_area_boudary', default = [10000, 25000]) # pidray
    #parser.add_argument('--category_group_area_boudary', default = [10000, 15000]) # opiray
    #parser.add_argument('--category_group_area_boudary', default = [40000, 100000]) # hixray

    # Generation method(set to 1 for Xsyn-M and 3 for Xsyn-A)
    parser.add_argument('--gen_method', default=1, help='')

    # BOM
    parser.add_argument('--latent_redist', default=True, help='whether or not redistribute latents according to alpha')
    parser.add_argument('--alpha', default=[0.3], help='indicate the hidden propotion')
    parser.add_argument('--hidden_start_step', default=1, help='indicate which timestep to do latent redist')
    parser.add_argument('--hidden_time', default='after', help='chosen from [after | during]')
    parser.add_argument('--occlusion_method', default=3, help='1: self-occlusion; 2: inter-occlusion; 3: background-occlusion')
    parser.add_argument('--rand_fg_mask_hidden', default=False)
    parser.add_argument('--gen2_hidden', default=False)
    parser.add_argument('--gen2_hidden_anno', default='/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen1_1%_filter_full_scar_range1_rand_fg_hidden/annotation/anno_refine.json')
    parser.add_argument('--gen3_hidden', default=False)
    #parser.add_argument('--gen3_hidden_anno', default='/home/ct/data/sjl/gligen_official_inpaint/generated_images/text_box_180000/gen3_0.1%_filter_full_scar_best_hidden_newer/annotation/anno_refine.json') #pidray
    parser.add_argument('--gen3_hidden_anno', default='/home/ct/data/sjl/gligen_official_inpaint_opiray/generated_images/text_box_100000/gen3_0.4%_filter_best/annotation/anno_refine.json') #opixray
    #parser.add_argument('--gen3_hidden_anno', default='/home/ct/data/sjl/gligen_official_hixray_inpaint/generated_images/text_box_180000/gen3_0.5%_filter_best/annotation/anno_refine.json')
    
    # CAR
    parser.add_argument('--refine_anno', default=True, help='whether or not use car')
    parser.add_argument('--scratch_generate', default=True)
    parser.add_argument('--do_decode', default=True, help='whether or not decode the image')
    parser.add_argument('--refine_strategies', default='h_sampling', help='car strategies, chosen from [h_sampling | topk_sampling]')
    parser.add_argument('--h_sampling_strategies', default='median', help='h_sampling strategies, chosen from [mode_convex | median_convex | median | median_topk], meaningful only when refine_strategies is h_sampling')
    parser.add_argument('--range_nums', default=[1,2,3,4], help='indicates the range of anchor points in ca_map')
    parser.add_argument('--topks', default=[1], help='topk sample points in ca_map')
    parser.add_argument('--ablation_opt', default=['range4'], help='ablation options: only_box, range4')

    parser.add_argument('--boxdiff', default=False, help='whether or not use boxdiff')
    parser.add_argument('--attention_res', default=16, help='cross attention map resolution')
    parser.add_argument('--smooth_attentions', default=True, help='whether or not use Gaussian to smooth ca_map')
    parser.add_argument('--sigma', default=0.5, help=' Gaussian sigma')
    parser.add_argument('--kernel_size', default=3, help='Gaussian kernel_size')
    parser.add_argument('--normalize_eot', default=False, help='')

    parser.add_argument('--inpaint', default=True, help='')

    args = parser.parse_args()

    # Fix bool args: argparse stores string "False" which is truthy
    _bool_args = [
        'use_sam', 'latent_redist', 'rand_fg_mask_hidden', 'gen2_hidden', 'gen3_hidden',
        'refine_anno', 'scratch_generate', 'do_decode', 'boxdiff',
        'smooth_attentions', 'normalize_eot', 'inpaint',
    ]
    for _a in _bool_args:
        v = getattr(args, _a, None)
        if isinstance(v, str):
            setattr(args, _a, v.lower() not in ('false', '0', 'no'))

    # Fix int args: argparse stores them as strings when no type= is specified
    _int_args = ['gen_method', 'batch_size', 'hidden_start_step',
                 'occlusion_method', 'attention_res', 'kernel_size']
    for _a in _int_args:
        v = getattr(args, _a, None)
        if isinstance(v, str):
            setattr(args, _a, int(v))

    # Fix float args
    _float_args = ['guidance_scale', 'sigma', 'sam_box_iou_thre']
    for _a in _float_args:
        v = getattr(args, _a, None)
        if isinstance(v, str):
            setattr(args, _a, float(v))

    run(args)