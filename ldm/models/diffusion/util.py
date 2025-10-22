import abc
import torch
import cv2
import numpy as np
import os
import pdb
from PIL import Image

LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77


# pidray

coco_category_list_check = [
    'baton', 'bullet', 'gun', 'hammer', 'power', 'bank', 'wrench',
    'hand', 'cuffs', 'knife', 'lighter', 'p', 'liers', 'scissors', 'spra', 'yer'
]


# opiray
'''
coco_category_list_check = [
    'utility', 'knife', 'multi', 'tool', 'folding', 'straight', 'sciss', 'or'
]
'''

#hixray
'''
coco_category_list_check = [
    'portable', 'charger', '1', '2', 'water', 'mobile', 'phone', 'cosmetic', 'non', 'metallic', 'lighter', 'tablet', 'laptop'
]
'''


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        #if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
        self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        #with torch.no_grad():
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]#.detach()
        self.step_store = self.get_empty_store()

    '''
    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention
    '''
    
    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_average_global_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

'''
class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        self.attention_store = self.step_store
        if self.save_global_store:
            with torch.no_grad():
                if len(self.global_store) == 0:
                    self.global_store = self.step_store
                else:
                    for key in self.global_store:
                        for i in range(len(self.global_store[key])):
                            self.global_store[key][i] += self.step_store[key][i].detach()
        self.step_store = self.get_empty_store()
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_average_global_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.global_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}

    def __init__(self, save_global_store=False):

        super(AttentionStore, self).__init__()
        self.save_global_store = save_global_store
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}
        self.curr_step_index = 0
'''

def aggregate_attention(attention_store, res, from_where, is_cross, select, prompts=None):
    out = []
    attention_maps = attention_store.get_average_global_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts) if prompts is not None else 1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    return out.cpu()

def boxdiff_aggregate_attention(attention_store: AttentionStore,
                        res: int,
                        from_where,
                        is_cross: bool,
                        select: int) -> torch.Tensor:
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = []
    attention_maps = attention_store.get_average_attention()

    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out

def transform_ca_map(image):
    image = 255 * image / image.max()
    image = image.unsqueeze(-1).expand(*image.shape, 3)
    image = image.numpy().astype(np.uint8)
    image = np.array(Image.fromarray(image).resize((512, 512)))

    return image

def show_cross_attention(attention_store, from_where, select=0, prompts=None , tokenizer=None):
    
    tokens = tokenizer.encode(prompts[select])
    categories = prompts[select].split(', ')
    categories = [category.lower() for category in categories]

    decoder = tokenizer.decode

    attention_maps = aggregate_attention(attention_store, 16, from_where, True, select,prompts=prompts)
    attention_maps = [attention_maps.sum(0) / attention_maps.shape[0]]
    
    attention_maps_32 = aggregate_attention(attention_store, 32, from_where, True, select,prompts=prompts)
    attention_maps_32 = [attention_maps_32.sum(0) / attention_maps_32.shape[0]]
    
    attention_maps_64 = aggregate_attention(attention_store, 64, from_where, True, select,prompts=prompts)
    attention_maps_64 = [attention_maps_64.sum(0) / attention_maps_64.shape[0]]

    ca_maps = []
    
    for idx, (attention_map, attention_map_32,attention_map_64) in enumerate(zip(attention_maps,attention_maps_32,attention_maps_64)):
        
        for i in range(len(tokens)):
            class_current = decoder(int(tokens[i])) 
            gt_kernel_final = np.zeros((512,512))

            if class_current not in coco_category_list_check:
                continue
            
            # pidray
            if class_current in ['cuffs', 'liers', 'yer']:
                continue

            # opiray
            # if class_current in ['knife', 'or', 'tool']:
            #     continue

            #hixray
            # if class_current in ['charger', '1', '2', 'phone',  'lighter', 'metallic']:
            #     continue

            if class_current in ['hand', 'p', 'spra', 'power']: # pidray
            #if class_current in ['utility', 'multi', 'folding', 'straight', 'sciss']: # opixray
            #if class_current in ['portable', 'water', 'mobile', 'cosmetic', 'non', 'tablet', 'laptop']: # hixray
                number_gt = 0
                # pidray
                start, end = i, i+2 # pidray

                # opixray
                # if class_current in ['utility', 'folding', 'straight']:
                #     start, end = i, i+3
                # elif class_current in ['multi']:
                #     start, end = i, i+5
                # else:
                #     start, end = i, i+2

                #hixray
                
                # if class_current in ['portable']:
                #     start, end = i, i+5
                # elif class_current in ['water', 'cosmetic', 'tablet', 'laptop']:
                #     start, end = i, i+1
                # elif class_current in ['non']:
                #     start, end = i, i+4
                # else:
                #     start, end = i, i+3
                

                for j in range(start, end):
                    
                    image_16 = attention_map[:, :, j]
                    image_16 = 255 * image_16 / image_16.max()
                    image_16 = cv2.resize(image_16.numpy().astype(np.uint8), (512, 512), interpolation=cv2.INTER_CUBIC)
                    '''
                    image_32 = attention_map_32[:, :, j]
                    image_32 = 255 * image_32 / image_32.max()
                    image_32 = cv2.resize(image_32.numpy().astype(np.uint8), (512, 512), interpolation=cv2.INTER_CUBIC)
                    
                    image_64 = attention_map_64[:, :, j]
                    image_64 = 255 * image_64 / image_64.max()
                    image_64 = cv2.resize(image_64.numpy().astype(np.uint8), (512, 512), interpolation=cv2.INTER_CUBIC)
                    '''
                    #image = (image_16 + image_32 + image_64) / 3
                    image = image_16
                    #image = (image_16 + image_32) / 2
                    
                    gt_kernel_final += image.copy()
                    number_gt += 1
                gt_kernel_final = gt_kernel_final/number_gt
                '''
                gt_kernel_final = (gt_kernel_final>0.5) *1
                if class_current == 'hand':
                    class_one = 'handcuffs'
                elif class_current == 'p':
                    class_one = 'pliers'
                else:
                    class_one = 'sprayer'
                gt_kernel_final = coco_category_to_id_v1[class_one]* gt_kernel_final
                '''
            else:
                
                image_16 = attention_map[:, :, i]
                image_16 = 255 * image_16 / image_16.max()
                image_16 = cv2.resize(image_16.numpy().astype(np.uint8), (512, 512), interpolation=cv2.INTER_CUBIC)
                '''
                image_32 = attention_map_32[:, :, i]
                image_32 = 255 * image_32 / image_32.max()
                image_32 = cv2.resize(image_32.numpy().astype(np.uint8), (512, 512), interpolation=cv2.INTER_CUBIC)
                
                image_64 = attention_map_64[:, :, i]
                image_64 = 255 * image_64 / image_64.max()
                image_64 = cv2.resize(image_64.numpy().astype(np.uint8), (512, 512), interpolation=cv2.INTER_CUBIC)
                '''

                #gt_kernel_final = (image_16 + image_32 + image_64) / 3
                gt_kernel_final = image_16
                #gt_kernel_final = (image_16 + image_32) / 2
                '''
                gt_kernel_final = (image>0.5) *1
                gt_kernel_final = coco_category_to_id_v1[class_current]* gt_kernel_final
                '''

            ca_maps.append(gt_kernel_final.astype(np.uint8))
        
        #pdb.set_trace()

        #cv2.imwrite(".//DiffSeg_Data/All_Class/mask/" + out_put,gt_kernel_final)

        return ca_maps

def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        
        def reshape_heads_to_batch_dim(self, tensor):
            batch_size, seq_len, dim = tensor.shape
            head_size = self.heads
            tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
            tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
            return tensor

        def reshape_batch_dim_to_heads(self, tensor):
            batch_size, seq_len, dim = tensor.shape
            head_size = self.heads
            tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
            tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
            return tensor


        def forward(x, key, value, mask=None):

            q = self.to_q(x)     # B*N*(H*C)
            k = self.to_k(key)   # B*M*(H*C)
            v = self.to_v(value) # B*M*(H*C)
    
            B, N, HC = q.shape 
            _, M, _ = key.shape
            H = self.heads
            C = HC // H 

            q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
            k = k.view(B,M,H,C).permute(0,2,1,3).reshape(B*H,M,C) # (B*H)*M*C
            v = v.view(B,M,H,C).permute(0,2,1,3).reshape(B*H,M,C) # (B*H)*M*C

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale # (B*H)*N*M
            self.fill_inf_from_mask(sim, mask)
            attn = sim.softmax(dim=-1) # (B*H)*N*M

            is_cross = key is not None

            attn = controller(attn, is_cross, place_in_unet)

            out = torch.einsum('b i j, b j d -> b i d', attn, v) # (B*H)*N*C
            out = out.view(B,H,N,C).permute(0,2,1,3).reshape(B,N,(H*C)) # B*N*(H*C)

            return self.to_out(out)
        
        '''
        def forward(x, context=None, mask=None):
            #def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            # x = hidden_states
            # context = encoder_hidden_states
            # mask = attention_mask
            
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = reshape_heads_to_batch_dim(self,q)
            k = reshape_heads_to_batch_dim(self,k)
            v = reshape_heads_to_batch_dim(self,v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1).to(torch.bool)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = reshape_batch_dim_to_heads(self,out)
            return to_out(out)
        '''
        return forward
    
        

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet, module_name=None):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        # if module_name in ["attn1", "attn2"]:
        #     net_.forward = ca_forward(net_, place_in_unet)
        #     return count + 1
        elif hasattr(net_, 'children'):
            for k,net__ in net_.named_children():
                count = register_recr(net__, count, place_in_unet, module_name = k)
        return count

    cross_att_count = 0
    #sub_nets = model.unet.named_children()
    sub_nets = model.named_children()
    
    for net in sub_nets:
        if "input" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "output" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count