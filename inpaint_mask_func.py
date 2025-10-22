import re
import cv2
import random
import importlib
import torch
from argparse import Namespace
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision
import copy
import bezier

def get_tensor(normalize=False, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def draw_masks_from_boxes(boxes, size, randomize_fg_mask=False, random_add_bg_mask=False, arbitrary_mask_percent=0.5):
    "boxes should be the output from dataset, which is a batch of bounding boxes"

    image_masks = [] 
    
    for box in boxes: # This is batch dimension
        image_mask = torch.ones(size,size)
        for bx in box:
            x0,y0,x1,y1 = bx*size
            x0,y0,x1,y1 = int(x0), int(y0), int(x1), int(y1)
            bbox = [x0,y0,x1,y1]
            '''
            mask_aug_prob = random.uniform(0, 1)
            if mask_aug_prob < arbitrary_mask_percent: #Paint-by-example augmentation
                W, H = size, size
                extended_bbox=copy.copy(bbox)
                left_freespace=bbox[0]-0
                right_freespace=W-bbox[2]
                up_freespace=bbox[1]-0
                down_freespace=H-bbox[3]
                extended_bbox[0]=bbox[0]-random.randint(0,int(0.4*left_freespace))
                extended_bbox[1]=bbox[1]-random.randint(0,int(0.4*up_freespace))
                extended_bbox[2]=bbox[2]+random.randint(0,int(0.4*right_freespace))
                extended_bbox[3]=bbox[3]+random.randint(0,int(0.4*down_freespace))
                
                mask_img = Image.new('L', (size, size), 255) 
                bbox_mask=copy.copy(bbox)
                extended_bbox_mask=copy.copy(extended_bbox)
                top_nodes = np.asfortranarray([
                                [bbox_mask[0],(bbox_mask[0]+bbox_mask[2])/2 , bbox_mask[2]],
                                [bbox_mask[1], extended_bbox_mask[1], bbox_mask[1]],
                            ])
                down_nodes = np.asfortranarray([
                        [bbox_mask[2],(bbox_mask[0]+bbox_mask[2])/2 , bbox_mask[0]],
                        [bbox_mask[3], extended_bbox_mask[3], bbox_mask[3]],
                    ])
                left_nodes = np.asfortranarray([
                        [bbox_mask[0],extended_bbox_mask[0] , bbox_mask[0]],
                        [bbox_mask[3], (bbox_mask[1]+bbox_mask[3])/2, bbox_mask[1]],
                    ])
                right_nodes = np.asfortranarray([
                        [bbox_mask[2],extended_bbox_mask[2] , bbox_mask[2]],
                        [bbox_mask[1], (bbox_mask[1]+bbox_mask[3])/2, bbox_mask[3]],
                    ])
                top_curve = bezier.Curve(top_nodes,degree=2)
                right_curve = bezier.Curve(right_nodes,degree=2)
                down_curve = bezier.Curve(down_nodes,degree=2)
                left_curve = bezier.Curve(left_nodes,degree=2)
                curve_list=[top_curve,right_curve,down_curve,left_curve]
                pt_list=[]
                random_width=5
                for curve in curve_list:
                    x_list=[]
                    y_list=[]
                    for i in range(1,19):
                        if (curve.evaluate(i*0.05)[0][0]) not in x_list and (curve.evaluate(i*0.05)[1][0] not in y_list):
                            pt_list.append((curve.evaluate(i*0.05)[0][0]+random.randint(-random_width,random_width),curve.evaluate(i*0.05)[1][0]+random.randint(-random_width,random_width)))
                            x_list.append(curve.evaluate(i*0.05)[0][0])
                            y_list.append(curve.evaluate(i*0.05)[1][0])
                mask_img_draw=ImageDraw.Draw(mask_img)
                mask_img_draw.polygon(pt_list,fill=0)
                image_mask = get_tensor(normalize=False, toTensor=True)(mask_img)[0]
            '''
            #else:
            
            obj_width = x1-x0
            obj_height = y1-y0
            if randomize_fg_mask and (random.uniform(0,1)<0.5) and (obj_height>=4) and (obj_width>=4):
                obj_mask = get_a_fg_mask(obj_height, obj_width)
                image_mask[y0:y1,x0:x1] = image_mask[y0:y1,x0:x1] * obj_mask # put obj mask into the inpainting mask 
            else:
                image_mask[y0:y1,x0:x1] = 0  # box itself is mask for the obj
        

        # So far we already drew all masks for obj, add bg mask if needed
        if random_add_bg_mask and (random.uniform(0,1)<0.5):
            bg_mask = get_a_bg_mask(size)
            image_mask *= bg_mask

        image_masks.append(image_mask)
    return torch.stack(image_masks).unsqueeze(1)





def get_a_fg_mask(height, width):
    """
    This will return an arbitrary mask for the obj, The overall masked region is ??? of all area. 
    I first start from a 64*64 mask (in other words, assume all object has the size of 64*64), 
    and use the empirically found parameters to generate a mask. Then I will resize (NEREAST) it into 
    given size. 

    Due to some hyper-paramters such as minBrushWidth, the input height and width must larger than 
    certain value. I set it as 4. In other words, for an object with size smaller than 4*4 (actual size is 32*32 in image space),
    we will not convert it into a random mask, but always box mask during training. 

    Since I still want to mask to cover most portion of the actual object, and also want to make box coordinate still makes sense 
    thus the hyper-parameters I set here will generate a mask with 75% overall area. 
    The chances of the mask touching all 4 edges (top, bottom, left, right) is high, otherwise the 
    grounding token information (based on box) will not be matched with mask here. (Once touching, the 
    box info in grounding token is still true, one can think that as box coordiante for the object mask)   

    """
    assert height>=4 and width>=4 
    size=64
    max_parts=6 
    maxVertex=10
    maxLength=80 
    minBrushWidth=10
    maxBrushWidth=32 
    maxAngle=360
    mask = generate_stroke_mask(im_size=(size,size), 
                                max_parts=max_parts, 
                                maxVertex=maxVertex,
                                maxLength=maxLength,
                                minBrushWidth=minBrushWidth,
                                maxBrushWidth=maxBrushWidth, 
                                maxAngle=maxAngle )
    mask = 1 - torch.tensor(mask)
    
    # resize the mask according to the actual size
    mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(height, width))
    mask = mask.squeeze(0).squeeze(0)

    return mask  







def get_a_bg_mask(size):
    """
    This will return an arbitrary mask for the entire image, The overall masked region is 30% of all area
    The 1 is visible region, 0 means masked unvisible region
    """
    assert size == 64, "The following args is I empirically set for 64*64, which is StableDiffsion Latent size"
    size = 64
    max_parts=4 
    maxVertex=10
    maxLength=32
    maxBrushWidth=12 
    minBrushWidth=3
    maxAngle=360
    mask = generate_stroke_mask( im_size=(size,size), 
                                max_parts=max_parts, 
                                maxVertex=maxVertex,
                                maxLength=maxLength,
                                minBrushWidth=minBrushWidth,
                                maxBrushWidth=maxBrushWidth, 
                                maxAngle=maxAngle )
    mask = 1 - torch.tensor(mask)
    return mask  







# The following code is from BAT-Fill, which is from some other inpainting work I think, maybe Gated Convolution?
# I also made some changes including adding minBrushWidth argument


def generate_stroke_mask(im_size, max_parts=10, maxVertex=20, maxLength=100, minBrushWidth=10, maxBrushWidth=24, maxAngle=360):
    assert minBrushWidth<=maxBrushWidth
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    parts = random.randint(1, max_parts)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, minBrushWidth, maxBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)
    # mask = np.concatenate([mask, mask, mask], axis = 2)
    return mask[...,0]

def np_free_form_mask(maxVertex, maxLength, minBrushWidth, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(1,maxVertex + 1)
    startY = np.random.randint(1,h)
    startX = np.random.randint(1,w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(1,maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(minBrushWidth, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)
        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(int)
        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask