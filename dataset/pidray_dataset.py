from .base_dataset import BaseDataset, recalculate_box_and_verify_if_valid  
import torch
import random
from PIL import Image, ImageOps
import os
import numpy as np
import albumentations as A
from transformers import (
    CLIPProcessor,
    CLIPVisionModelWithProjection
)
from diffusers.pipelines.stable_diffusion.clip_image_project_model import CLIPImageProjection
import cv2
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

def make_a_sentence(obj_names, clean=False):

    if clean:
        obj_names = [ name[:-6] if ("-other" in name) else name for name in obj_names]

    caption = ""
    tokens_positive = []
    for obj_name in obj_names:
        start_len = len(caption)
        caption += obj_name
        end_len = len(caption)
        caption += ", "
        tokens_positive.append(
            [[start_len, end_len]] # in real caption, positive tokens can be disjoint, thus using list of list
        )
    caption = caption[:-2] # remove last ", "

    return caption #, tokens_positive

def mask_for_random_drop_text_or_image_feature(masks, random_drop_embedding):
    """
    input masks tell how many valid grounding tokens for this image
    e.g., 1,1,1,1,0,0,0,0,0,0...

    If random_drop_embedding=both.  we will random drop either image or
    text feature for each token, 
    but we always make sure there is at least one feature used. 
    In other words, the following masks are not valid 
    (because for the second obj, no feature at all):
    image: 1,0,1,1,0,0,0,0,0
    text:  1,0,0,0,0,0,0,0,0

    if random_drop_embedding=image. we will random drop image feature 
    and always keep the text one.  

    """
    N = masks.shape[0]

    if random_drop_embedding=='both':
        temp_mask = torch.ones(2,N)
        for i in range(N):
            if random.uniform(0, 1) < 0.5: # else keep both features 
                idx = random.sample([0,1], 1)[0] # randomly choose to drop image or text feature 
                temp_mask[idx,i] = 0 
        image_masks = temp_mask[0]*masks
        text_masks = temp_mask[1]*masks
    
    if random_drop_embedding=='image':
        image_masks = masks*(torch.rand(N)>0.5)*1
        text_masks = masks

    return image_masks, text_masks

def draw_masks_from_boxes(boxes, image_size, randomize_fg_mask=False, random_add_bg_mask=False):
    "boxes should be the output from dataset, which is a batch of bounding boxes"

    image_masks = [] 
    for box in boxes: # This is batch dimension

        image_mask = torch.ones(image_size)
        for bx in box:
            x0,y0,x1,y1 = bx
            x0,y0,x1,y1 = int(x0), int(y0), int(x1), int(y1)
            image_mask[y0:y1,x0:x1] = 0  # box itself is mask for the obj

        image_masks.append(image_mask)
    return torch.stack(image_masks).unsqueeze(1)

def image_embbeding(img, processor, image_encoder, image_project, normalize_constant):
    with torch.no_grad():
        inputs = processor(images=[img], return_tensors='pt', input_data_format="channels_last")
        # except ZeroDivisionError:
        #     print(f'ZeroDivisionError: drop this image and continue next iteration...')
        #     drop = True
        # if drop:
        #     break
        inputs["pixel_values"] = inputs["pixel_values"].to(image_encoder.dtype)
        outputs = image_encoder(**inputs)
        feature = outputs.image_embeds
        feature = image_project(feature).squeeze(0)
        image_embeddings_before_projection = (feature / feature.norm()).cpu()
        image_embeddings_before_projection *= float(normalize_constant)
    
    return image_embeddings_before_projection

class PIDrayDataset(BaseDataset):
    def __init__(self, 
                pidray_path,
                image_path,
                depth_path=None,
                which_layer_text='before', 
                which_layer_image="after_reproject",
                prob_use_caption=0.5,
                random_drop_embedding='none',
                image_size=512, 
                min_box_size=0.01,
                max_boxes_per_data=8,
                max_images=None, # set as 30K used to eval
                random_crop = False,
                random_flip = True,
                ):
        super().__init__(random_crop, random_flip, image_size)
        self.pidray_path = pidray_path
        self.image_path = image_path
        self.depth_path = depth_path
        self.which_layer_text  = which_layer_text
        self.which_layer_image = which_layer_image
        self.prob_use_caption = prob_use_caption
        self.random_drop_embedding = random_drop_embedding
        self.min_box_size = min_box_size
        self.max_boxes_per_data = max_boxes_per_data
        self.max_images = max_images
        
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained("/home/ct/data/sjl/openaiclip_vit_large_patch14")
        self.processor = CLIPProcessor.from_pretrained("/home/ct/data/sjl/openaiclip_vit_large_patch14")
        self.image_project = CLIPImageProjection()
        self.normalize_constant = 28.7
        
        self.random_trans=A.Compose([
            A.Resize(height=224,width=224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20),
            A.Blur(p=0.3),
            A.ElasticTransform(p=0.3)
        ])

        assert which_layer_text in ['before','after']
        assert which_layer_image in ['after', 'after_renorm', 'after_reproject']
        assert random_drop_embedding in ['none', 'both', 'image']
        
        # preprocessed CLIP feature embedding length: 768  
        self.embedding_len = 768
        
        self.data_list = torch.load(pidray_path, map_location="cpu")


    def total_images(self):
        return len(self)

    def __getitem__(self, index):
        if self.max_boxes_per_data > 99:
            assert False, "Are you sure setting such large number of boxes per image?"

        data = self.data_list[index]

        out = {}

        # -------------------- id and image ------------------- # 
        image = Image.open(os.path.join(self.image_path, data["file_path"])).convert("RGB")
        #image = Image.open(os.path.join(self.depth_path, data["file_path"])).convert("RGB")
        image_tensor, trans_info = self.transform_image(image)
        out["image"] = image_tensor

        # -------------------- grounding token ------------------- # 
        annos = data['annos']
        
        areas = []
        all_boxes = []
        all_masks = []
        all_text_embeddings = []
        all_image_embeddings = []
        all_category_names = []

        text_embedding_name = 'text_embeddings_before_projection' 
        image_embedding_name = np.random.choice(['seg_image_embeddings_before_projection', 'crop_seg_image_embeddings_before_projection', 'original_image_embeddings_before_projection']) # crop_seg_image_embeddings_before_projection
        image_name = np.random.choice(['seg_img', 'crop_seg_img', 'original_image'])

        for anno in annos:
            #x, y, w, h = anno['bbox']
            x0, y0, x1, y1 = anno["bbox"]
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
            valid, (x0, y0, x1, y1) = recalculate_box_and_verify_if_valid(x, y, w, h, trans_info, self.image_size, self.min_box_size)

            if valid:
                areas.append(  (x1-x0)*(y1-y0)  )
                all_boxes.append( torch.tensor([x0,y0,x1,y1]) / self.image_size ) # scale to 0-1
                all_masks.append(1)
                all_text_embeddings.append(anno[text_embedding_name])
                
                # no augmentation
                #all_image_embeddings.append(anno[image_embedding_name])
                
                # with augmentation
                '''
                if image_name == 'original_image' or image_name == 'crop_seg_img':
                    ref_img = Image.open(anno[image_name]).convert("RGB").crop(anno["bbox"])
                else:
                    ref_img = Image.open(anno[image_name]).convert("RGB")
                ref_img = np.array(ref_img)
                #ref_img = anno[image_name]
                aug_ref_img = self.random_trans(image=ref_img)
                aug_ref_img = Image.fromarray(aug_ref_img["image"])
                all_image_embeddings.append(image_embbeding(aug_ref_img, self.processor, self.image_encoder, self.image_project, self.normalize_constant))
                '''
                all_category_names.append(anno["caption"].split(' ')[-1])

        # Sort according to area and choose the largest N objects   
        wanted_idxs = torch.tensor(areas).sort(descending=True)[1]
        wanted_idxs = wanted_idxs[0:self.max_boxes_per_data]
        category_names = [all_category_names[i] for i in wanted_idxs]

        boxes = torch.zeros(self.max_boxes_per_data, 4)
        masks = torch.zeros(self.max_boxes_per_data)
        text_embeddings =  torch.zeros(self.max_boxes_per_data, self.embedding_len)
        image_embeddings = torch.zeros(self.max_boxes_per_data, self.embedding_len)

        for i, idx in enumerate(wanted_idxs):
            boxes[i] = all_boxes[idx]
            masks[i] = all_masks[idx]
            text_embeddings[i] =  all_text_embeddings[idx]
            #image_embeddings[i] = all_image_embeddings[idx]


        if self.random_drop_embedding != 'none':
            image_masks, text_masks = mask_for_random_drop_text_or_image_feature(masks, self.random_drop_embedding)
        else:
            image_masks = masks
            text_masks = masks


        out["boxes"] = boxes
        out["masks"] = masks # indicating how many valid objects for this image-text data
        out["image_masks"] = image_masks # indicating how many objects still there after random dropping applied
        out["text_masks"] = text_masks # indicating how many objects still there after random dropping applied
        out["text_embeddings"] =  text_embeddings  
        out["image_embeddings"] = image_embeddings
        
        #inpainting_mask = draw_masks_from_boxes(out['boxes'], image_tensor.shape[1:])   
        #out['inpaint_image'] = out["image"] * inpainting_mask
        #out['inpaint_mask'] = inpainting_mask

        # -------------------- caption ------------------- # 
        if random.uniform(0, 1) > self.prob_use_caption:
            out["caption"] = make_a_sentence(category_names)
        else:
            out["caption"] = ""

        return out



    def __len__(self):
        return len(self.data_list)

class PIDrayDepthDataset():
    def __init__(self, pidray_path,
                image_path, depth_path, prob_use_caption=0.5, image_size=512,
                random_flip = False):
        self.prob_use_caption = prob_use_caption
        self.image_size = image_size
        self.pidray_path = pidray_path
        self.image_path = image_path
        self.depth_path = depth_path
        self.random_flip = random_flip
        
        self.pil_to_tensor = transforms.PILToTensor()
        
        self.data_list = torch.load(pidray_path, map_location="cpu")

    def __getitem__(self, index):

        data = self.data_list[index]

        out = {}

        image = Image.open(os.path.join(self.image_path, data["file_path"]))
        depth = Image.open(os.path.join(self.depth_path, data["file_path"]))
        
        # - - - - - center_crop, resize and random_flip - - - - - - #  
        assert  image.size == depth.size   

        crop_size = min(image.size)
        image = TF.center_crop(image, crop_size)
        image = image.resize( (self.image_size, self.image_size) )

        depth = TF.center_crop(depth, crop_size)
        depth = depth.resize( (self.image_size, self.image_size) )


        if self.random_flip and random.random()<0.5:
            image = ImageOps.mirror(image)
            depth = ImageOps.mirror(depth)
        
        out['image'] = ( self.pil_to_tensor(image).float()/255 - 0.5 ) / 0.5
        out['depth'] = ( self.pil_to_tensor(depth).float()/255 - 0.5 ) / 0.5
        out['mask'] = torch.tensor(1.0) 
        
        annos = data['annos']
        all_category_names = []
        for anno in annos:
            all_category_names.append(anno["caption"].split(' ')[-1])

        # -------------------- caption ------------------- # 
        if random.uniform(0, 1) > self.prob_use_caption:
            out["caption"] = make_a_sentence(all_category_names)
        else:
            out["caption"] = ""

        return out

    def total_images(self):
        return len(self)

    def __len__(self):
        return len(self.data_list)