import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like

from .util import *
import pdb
from torch.nn import functional as F
from .gaussian_smoothing import GaussianSmoothing

from torchviz import make_dot

def check_in_graph(tensor, name):
    if tensor.grad_fn is not None:
        print(f"Tensor {name} is in the computation graph.")
    else:
        print(f"Tensor {name} is NOT in the computation graph.")

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

@torch.no_grad()
def latent_redistribution(args, latent, boxes, alpha, hidden_time, hidden_info):
    size = latent.shape[2]
    boxes = boxes[0] # batch dim
    hidden_latent = latent.clone().detach()
    img_orig = hidden_info['img_orig'].to(latent.device)

    if args.occlusion_method == 1:  # 遮挡1：自遮挡
        for box in boxes:  
            x0, y0, x1, y1 = box
            if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
                break
            x0, y0, x1, y1 = x0*size, y0*size, x1*size, y1*size
            x0,y0,x1,y1 = int(x0), int(y0), int(x1), int(y1)
            x_len, y_len = x1 - x0, y1 - y0

            out_x0, out_y0 = x0, y0
            
            out_x_len, out_y_len = x_len, y_len
            out_x1, out_y1 = out_x0 + out_x_len, out_y0 + out_y_len

            in_x0, in_y0 = x0, y0
            in_x1, in_y1 = in_x0 + out_x_len, in_y0 + out_y_len

            if hidden_time == 'after':
                hidden_latent[:, :, in_y0:in_y1, in_x0:in_x1] = alpha  * img_orig[:, :, out_y0:out_y1, out_x0:out_x1] + (1 - alpha) * latent[:, :, in_y0:in_y1, in_x0:in_x1]
            else:
                latent[:, :, in_y0:in_y1, in_x0:in_x1] = alpha  * img_orig[:, :, out_y0:out_y1, out_x0:out_x1] + (1 - alpha) * latent[:, :, in_y0:in_y1, in_x0:in_x1]
            #latent[:, :, :y_len, :x_len] = alpha  * img_orig[:, :, out_y0:out_y1, out_x0:out_x1] + (1 - alpha) * latent[:, :, :y_len, :x_len]
    
    elif args.occlusion_method == 3: #遮挡3：背景遮挡
        for in_coord, hidden_bg_box, hidden_bg_mask in zip(hidden_info['in_coords'], hidden_info['hidden_bg_boxs'], hidden_info['hidden_bg_masks']):
            in_x0, in_y0, in_x1, in_y1 = in_coord
            x_len, y_len = in_x1 - in_x0, in_y1 - in_y0
            bg_x0, bg_y0, bg_x1, bg_y1 = hidden_bg_box
            bg_width, bg_height = bg_x1 - bg_x0, bg_y1 - bg_y0
            min_x_len, min_y_len = min(x_len, bg_width), min(y_len, bg_height)
            hidden_bg_mask = hidden_bg_mask.to(latent.device).float()
            if hidden_time == 'after':
                # direct fuse
                hidden_latent[:, :, in_y0:in_y0+min_y_len, in_x0:in_x0+min_x_len] = alpha * img_orig[:, :, bg_y0:bg_y0+min_y_len, bg_x0:bg_x0+min_x_len] + (1 - alpha) * latent[:, :, in_y0:in_y0+min_y_len, in_x0:in_x0+min_x_len]
            else:
                latent[:, :, in_y0:in_y0+min_y_len, in_x0:in_x0+min_x_len] = alpha * img_orig[:, :, bg_y0:bg_y0+min_y_len, bg_x0:bg_x0+min_x_len] + (1 - alpha) * latent[:, :, in_y0:in_y0+min_y_len, in_x0:in_x0+min_x_len]
    
    return hidden_latent if hidden_time == 'after' else latent


class DDIMSampler(object):
    def __init__(self, diffusion, model, schedule="linear", alpha_generator_func=None, set_alpha_scale=None):
        super().__init__()
        self.diffusion = diffusion
        self.model = model
        self.device = diffusion.betas.device
        self.ddpm_num_timesteps = diffusion.num_timesteps
        self.schedule = schedule
        self.alpha_generator_func = alpha_generator_func
        self.set_alpha_scale = set_alpha_scale
        

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            attr = attr.to(self.device)
        setattr(self, name, attr)


    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=False)
        alphas_cumprod = self.diffusion.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)

        self.register_buffer('betas', to_torch(self.diffusion.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.diffusion.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=False)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)


    @torch.no_grad()
    def sample(self, S, shape, input, uc=None, guidance_scale=1, mask=None, x0=None, controller=None, args=None, boxes=None, do_hidden=None, hidden_info=None, inpaint=None):
        self.make_schedule(ddim_num_steps=S)
        return self.ddim_sampling(shape, input, uc, guidance_scale,  mask=mask, x0=x0, controller=controller, args=args, boxes=boxes, do_hidden=do_hidden, hidden_info=hidden_info, inpaint=inpaint)
    
    @torch.no_grad()
    def latent_redistribution(self, latent, boxes, mask, alpha, hidden_time, hidden_info):
        size = latent.shape[2]
        boxes = boxes[0] # batch dim
        hidden_latent = latent.clone().detach()
        img_orig = hidden_info['img_orig'].to(latent.device)

        if self.occlusion_method == 1:  # 遮挡1：自遮挡
            for box in boxes:  
                x0, y0, x1, y1 = box
                if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
                    break
                x0, y0, x1, y1 = x0*size, y0*size, x1*size, y1*size
                x0,y0,x1,y1 = int(x0), int(y0), int(x1), int(y1)
                x_len, y_len = x1 - x0, y1 - y0

                out_x0, out_y0 = x0, y0
                
                out_x_len, out_y_len = x_len, y_len
                out_x1, out_y1 = out_x0 + out_x_len, out_y0 + out_y_len

                in_x0, in_y0 = x0, y0
                in_x1, in_y1 = in_x0 + out_x_len, in_y0 + out_y_len

                if hidden_time == 'after':
                    hidden_latent[:, :, in_y0:in_y1, in_x0:in_x1] = alpha  * img_orig[:, :, out_y0:out_y1, out_x0:out_x1] + (1 - alpha) * latent[:, :, in_y0:in_y1, in_x0:in_x1]
                else:
                    latent[:, :, in_y0:in_y1, in_x0:in_x1] = alpha  * img_orig[:, :, out_y0:out_y1, out_x0:out_x1] + (1 - alpha) * latent[:, :, in_y0:in_y1, in_x0:in_x1]
                #latent[:, :, :y_len, :x_len] = alpha  * img_orig[:, :, out_y0:out_y1, out_x0:out_x1] + (1 - alpha) * latent[:, :, :y_len, :x_len]
        
        elif self.occlusion_method == 3: #遮挡3：背景遮挡
            for in_coord, hidden_bg_box, hidden_bg_mask in zip(hidden_info['in_coords'], hidden_info['hidden_bg_boxs'], hidden_info['hidden_bg_masks']):
                in_x0, in_y0, in_x1, in_y1 = in_coord
                x_len, y_len = in_x1 - in_x0, in_y1 - in_y0
                bg_x0, bg_y0, bg_x1, bg_y1 = hidden_bg_box
                bg_width, bg_height = bg_x1 - bg_x0, bg_y1 - bg_y0
                min_x_len, min_y_len = min(x_len, bg_width), min(y_len, bg_height)
                hidden_bg_mask = hidden_bg_mask.to(latent.device).float()
                if hidden_time == 'after':
                    # direct fuse
                    hidden_latent[:, :, in_y0:in_y0+min_y_len, in_x0:in_x0+min_x_len] = alpha * img_orig[:, :, bg_y0:bg_y0+min_y_len, bg_x0:bg_x0+min_x_len] + (1 - alpha) * latent[:, :, in_y0:in_y0+min_y_len, in_x0:in_x0+min_x_len]
                else:
                    latent[:, :, in_y0:in_y0+min_y_len, in_x0:in_x0+min_x_len] = alpha * img_orig[:, :, bg_y0:bg_y0+min_y_len, bg_x0:bg_x0+min_x_len] + (1 - alpha) * latent[:, :, in_y0:in_y0+min_y_len, in_x0:in_x0+min_x_len]
        
        return hidden_latent if hidden_time == 'after' else latent
 

    @torch.no_grad()
    def ddim_sampling(self, shape, input, uc, guidance_scale=1, mask=None, x0=None, controller=None, args=None, boxes=None, do_hidden=None, hidden_info=None, inpaint=None):
        if controller is not None:
            register_attention_control(self.model, controller)

        b = shape[0]
        
        #img = input["x"]
        # if img == None:     
        #     img = torch.randn(shape, device=self.device)
        #     input["x"] = img


        time_range = np.flip(self.ddim_timesteps)
        total_steps = self.ddim_timesteps.shape[0]

        #iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        iterator = time_range
  
        if self.alpha_generator_func != None:
            alphas = self.alpha_generator_func(len(iterator))

        if do_hidden:
            self.occlusion_method = args.occlusion_method
            if args.occlusion_method == 3:
                hidden_bg_boxs = hidden_info['hidden_bg_boxs']
                hidden_bg_masks = hidden_info['hidden_bg_masks']
                hidden_info['hidden_bg_boxs'] = []
                hidden_info['hidden_bg_masks'] = []
                for hidden_bg_box, hidden_bg_mask in zip(hidden_bg_boxs, hidden_bg_masks):
                    bg_x0, bg_y0, bg_x1, bg_y1 = hidden_bg_box
                    bg_x0, bg_y0, bg_x1, bg_y1 = bg_x0*64, bg_y0*64, bg_x1*64, bg_y1*64
                    bg_x0, bg_y0, bg_x1, bg_y1 = int(bg_x0), int(bg_y0), int(bg_x1), int(bg_y1) 
                    hidden_info['hidden_bg_boxs'].append([bg_x0, bg_y0, bg_x1, bg_y1])
                    hidden_info['hidden_bg_masks'].append(hidden_bg_mask)

                in_coords = []
                # for in_idx in range(len(boxes[0])):
                #     in_box = boxes[0][in_idx]
                hidden_fg_boxes = hidden_info['hidden_fg_boxes']
                for in_idx in range(len(hidden_fg_boxes)):
                    in_box = hidden_fg_boxes[in_idx]
                    in_x0, in_y0, in_x1, in_y1 = in_box
                    if in_x0 == 0 and in_y0 == 0 and in_x1 == 0 and in_y1 == 0:
                        break
                    hidden_bg_box = hidden_info['hidden_bg_boxs'][in_idx]
                    bg_x0, bg_y0, bg_x1, bg_y1 = hidden_bg_box
                    bg_width, bg_height = bg_x1 - bg_x0, bg_y1 - bg_y0
                    in_x0, in_y0, in_x1, in_y1 = in_x0*64, in_y0*64, in_x1*64, in_y1*64
                    in_x0, in_y0, in_x1, in_y1 = int(in_x0), int(in_y0), int(in_x1), int(in_y1)
                    try:
                        in_x0 = np.random.randint(max(in_x0 - bg_width, 0), in_x1)
                        in_y0 = np.random.randint(max(in_y0 - bg_height, 0), in_y1)
                        in_coords.append([in_x0, in_y0, in_x0+bg_width if in_x0+bg_width < 64 else 64, in_y0+bg_height if in_y0+bg_height < 64 else 64 ])
                    except:
                        in_coords.append([in_x0, in_y0, in_x0+bg_width if in_x0+bg_width < 64 else 64, in_y0+bg_height if in_y0+bg_height < 64 else 64 ])
                hidden_info['in_coords'] = in_coords
            

        for i, step in enumerate(iterator):

            # set alpha 
            if self.alpha_generator_func != None:
                self.set_alpha_scale(self.model, alphas[i])
                if  alphas[i] == 0:
                    self.model.restore_first_conv_from_SD()
                    
            # run 
            index = total_steps - i - 1
            input["timesteps"] = torch.full((b,), step, device=self.device, dtype=torch.long)
            
            if mask is not None:
                assert x0 is not None
                img_orig = self.diffusion.q_sample( x0, input["timesteps"] ) 
                if do_hidden:
                   hidden_info['img_orig'] = img_orig
                if inpaint:
                    input['x'] = img_orig * mask + (1. - mask) * input['x']
                #input["x"] = img
            
            if args.scar:
                with torch.enable_grad():
                    #pdb.set_trace()
                    input['x'] = input['x'].clone().detach().requires_grad_(True)
                    #latents = input['x'].clone().requires_grad_(True)
                    timesteps = input['timesteps']
                    context = input['context']
                    uc_context = uc
                    inpainting_extra_input = input['inpainting_extra_input']
                    grounding_input = input['grounding_input']

                    check_in_graph(input['x'], 'first_latents')

                    new_input = dict(x=input['x'], timesteps=timesteps, context=context, inpainting_extra_input=inpainting_extra_input, grounding_input = grounding_input, grounding_extra_input = None)

                    noise_pred_text = self.model(new_input)
                    check_in_graph(input['x'], 'first_after_forward_latents')

                    self.model.zero_grad()
                    check_in_graph(noise_pred_text, 'noise_pred_text')

                    if args.boxdiff:

                        # Get max activation value for each subject token
                        max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y = self._aggregate_and_get_max_attention_per_token(
                            attention_store=controller,
                            indices_to_alter=args.token_indices,
                            attention_res=args.attention_res,
                            smooth_attentions=args.smooth_attentions,
                            sigma=args.sigma,
                            kernel_size=args.kernel_size,
                            normalize_eot=args.normalize_eot,
                            bbox=args.vis_boxes,
                            config=args.boxdiff_config,
                        )

                        loss_fg, loss = self._compute_loss(max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y)

                        # Refinement from attend-and-excite (not necessary)
                        if i in args.boxdiff_config.thresholds.keys() and loss_fg > 1. - args.boxdiff_config.thresholds[i] and args.boxdiff_config.refine:
                            del noise_pred_text
                            torch.cuda.empty_cache()
                            loss_fg, input['x'], max_attention_per_index_fg = self._perform_iterative_refinement_step(
                                latents=input['x'],
                                indices_to_alter=args.token_indices,
                                loss_fg=loss_fg,
                                threshold=args.boxdiff_config.thresholds[i],
                                text_embeddings=context,
                                uc_text_embeddings=uc_context,
                                attention_store=controller,
                                step_size=args.boxdiff_config.scale_factor * np.sqrt(scale_range[i]),
                                t=timesteps,
                                inpainting_extra_input=inpainting_extra_input,
                                attention_res=args.attention_res,
                                smooth_attentions=args.boxdiff_config.smooth_attentions,
                                sigma=args.boxdiff_config.sigma,
                                kernel_size=args.boxdiff_config.kernel_size,
                                normalize_eot=args.normalize_eot,
                                bbox=args.vis_boxes,
                                config=args.boxdiff_config,
                            )

                        # Perform gradient update
                        if i < args.boxdiff_config.max_iter_to_alter:
                            _, loss = self._compute_loss(max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y)
                            if loss != 0:
                                input['x'] = self._update_latent(latents=input['x'], loss=loss,
                                                                step_size=args.boxdiff_config.scale_factor * np.sqrt(scale_range[i]))
                    
                    elif args.attend_and_excite:
                        # Get max activation value for each subject token
                        max_attention_per_index = self._aggregate_and_get_max_attention_per_token_attend(
                            attention_store=controller,
                            indices_to_alter=args.token_indices,
                            attention_res=args.attention_res,
                            smooth_attentions=args.smooth_attentions,
                            sigma=args.sigma,
                            kernel_size=args.kernel_size,
                            normalize_eot=args.normalize_eot,
                            )
                        
                        pdb.set_trace()

                        loss = self._compute_loss_attend(max_attention_per_index=max_attention_per_index)

                        # If this is an iterative refinement step, verify we have reached the desired threshold for all
                        if i in args.boxdiff_config.thresholds.keys() and loss > 1. - args.boxdiff_config.thresholds[i]:
                            del noise_pred_text
                            torch.cuda.empty_cache()
                            loss, input['x'], max_attention_per_index = self._perform_iterative_refinement_step_attend(
                                latents=input['x'],
                                indices_to_alter=args.token_indices,
                                loss=loss,
                                threshold=args.boxdiff_config.thresholds[i],
                                text_embeddings=context,
                                uc_text_embeddings=uc_context,
                                attention_store=controller,
                                step_size=args.boxdiff_config.scale_factor * np.sqrt(scale_range[i]),
                                t=timesteps,
                                inpainting_extra_input=inpainting_extra_input,
                                attention_res=args.attention_res,
                                smooth_attentions=args.smooth_attentions,
                                sigma=args.sigma,
                                kernel_size=args.kernel_size,
                                normalize_eot=args.normalize_eot)
                        

                        # Perform gradient update
                        if i < args.boxdiff_config.max_iter_to_alter:
                            loss = self._compute_loss_attend(max_attention_per_index=max_attention_per_index)
                            if loss != 0:
                                print(f'i = {i}')
                                input['x'] = self._update_latent_attend(latents=input['x'], loss=loss,
                                                            step_size=args.boxdiff_config.scale_factor * np.sqrt(scale_range[i]))
                            print(f'Iteration {i} | Loss: {loss:0.4f}')
                
                #input["x"] = latents.clone().detach()
                
                input['x'], pred_x0 = self.p_sample_ddim(input, index=index, uc=uc, guidance_scale=guidance_scale, controller=controller)

            else:
                input['x'], pred_x0 = self.p_sample_ddim(input, index=index, uc=uc, guidance_scale=guidance_scale, controller=controller)
                
                if do_hidden and args.hidden_time == 'during': # 生成前景遮挡
                    # 计算遮挡系数
                    #alpha = hidden_alpha(i, total_steps)
                    alpha = args.alpha
                    input['x'] = self.latent_redistribution(input['x'], boxes, alpha, mask, args.hidden_time, hidden_info)
                

            #input["x"] = img 

        #return img
        if do_hidden and args.hidden_time == 'after': # 生成前景遮挡
            # 计算遮挡系数
            #alpha = hidden_alpha(i, total_steps)
            hidden_latents = []
            for alpha in args.alpha:
                hidden_latent = self.latent_redistribution(input['x'], boxes, mask, alpha, args.hidden_time, hidden_info)
                hidden_latents.append(hidden_latent)
            #hidden_latent = self.latent_redistribution(input['x'], boxes, mask, args.alpha, args.hidden_time, hidden_info)
            #return input['x'], hidden_latent
            return input['x'], hidden_latents
        else:
            return input['x']


    @torch.no_grad()
    def p_sample_ddim(self, input, index, uc=None, guidance_scale=1, controller=None):


        e_t = self.model(input) 
        if uc is not None and guidance_scale != 1:
            unconditional_input = dict(x=input["x"], timesteps=input["timesteps"], context=uc, inpainting_extra_input=input["inpainting_extra_input"], grounding_extra_input=input['grounding_extra_input'])
            e_t_uncond = self.model( unconditional_input ) 
            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)

        # select parameters corresponding to the currently considered timestep
        b = input["x"].shape[0] 
        a_t = torch.full((b, 1, 1, 1), self.ddim_alphas[index], device=self.device)
        a_prev = torch.full((b, 1, 1, 1), self.ddim_alphas_prev[index], device=self.device)
        sigma_t = torch.full((b, 1, 1, 1), self.ddim_sigmas[index], device=self.device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), self.ddim_sqrt_one_minus_alphas[index],device=self.device)

        # current prediction for x_0
        pred_x0 = (input["x"] - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * torch.randn_like( input["x"] ) 
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        if controller is not None:
            x_prev = controller.step_callback(x_prev)

        return x_prev, pred_x0
    
    def _compute_max_attention_per_index(self,
                                         attention_maps: torch.Tensor,
                                         indices_to_alter,
                                         smooth_attentions: bool = False,
                                         sigma: float = 0.5,
                                         kernel_size: int = 3,
                                         normalize_eot: bool = False,
                                         bbox = None,
                                         config=None,
                                         ):
        """ Computes the maximum attention value for each of the tokens we wish to alter. """
        last_idx = -1
        if normalize_eot:
            prompt = self.prompt
            if isinstance(self.prompt, list):
                prompt = self.prompt[0]
            last_idx = len(self.tokenizer(prompt)['input_ids']) - 1
        attention_for_text = attention_maps[:, :, 1:last_idx]
        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # Shift indices since we removed the first token
        indices_to_alter = [index - 1 for index in indices_to_alter]

        # Extract the maximum values
        max_indices_list_fg = []
        max_indices_list_bg = []
        dist_x = []
        dist_y = []

        cnt = 0
        for i in indices_to_alter:
            image = attention_for_text[:, :, i]

            box = [max(round(b / (512 / image.shape[0])), 0) for b in bbox[cnt]]
            x1, y1, x2, y2 = box
            cnt += 1

            # coordinates to masks
            obj_mask = torch.zeros_like(image)
            ones_mask = torch.ones([y2 - y1, x2 - x1], dtype=obj_mask.dtype).to(obj_mask.device)
            #pdb.set_trace()
            obj_mask[y1:y2, x1:x2] = ones_mask
            bg_mask = 1 - obj_mask

            if smooth_attentions:
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                image = smoothing(input).squeeze(0).squeeze(0)

            # Inner-Box constraint
            k = (obj_mask.sum() * config.P).long()
            max_indices_list_fg.append((image * obj_mask).reshape(-1).topk(k)[0].mean())

            # Outer-Box constraint
            k = (bg_mask.sum() * config.P).long()
            max_indices_list_bg.append((image * bg_mask).reshape(-1).topk(k)[0].mean())

            # Corner Constraint
            gt_proj_x = torch.max(obj_mask, dim=0)[0]
            gt_proj_y = torch.max(obj_mask, dim=1)[0]
            corner_mask_x = torch.zeros_like(gt_proj_x)
            corner_mask_y = torch.zeros_like(gt_proj_y)

            # create gt according to the number config.L
            N = gt_proj_x.shape[0]
            corner_mask_x[max(box[0] - config.L, 0): min(box[0] + config.L + 1, N)] = 1.
            corner_mask_x[max(box[2] - config.L, 0): min(box[2] + config.L + 1, N)] = 1.
            corner_mask_y[max(box[1] - config.L, 0): min(box[1] + config.L + 1, N)] = 1.
            corner_mask_y[max(box[3] - config.L, 0): min(box[3] + config.L + 1, N)] = 1.
            dist_x.append((F.l1_loss(image.max(dim=0)[0], gt_proj_x, reduction='none') * corner_mask_x).mean())
            dist_y.append((F.l1_loss(image.max(dim=1)[0], gt_proj_y, reduction='none') * corner_mask_y).mean())

        return max_indices_list_fg, max_indices_list_bg, dist_x, dist_y

    def _aggregate_and_get_max_attention_per_token(self, attention_store: AttentionStore,
                                                   indices_to_alter,
                                                   attention_res: int = 16,
                                                   smooth_attentions: bool = False,
                                                   sigma: float = 0.5,
                                                   kernel_size: int = 3,
                                                   normalize_eot: bool = False,
                                                   bbox = None,
                                                   config=None,
                                                   ):
        """ Aggregates the attention for each token and computes the max activation value for each token to alter. """
        attention_maps = boxdiff_aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0)

        max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y = self._compute_max_attention_per_index(
            attention_maps=attention_maps,
            indices_to_alter=indices_to_alter,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot,
            bbox=bbox,
            config=config,
        )
        
        return max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y

    @staticmethod
    def _compute_loss(max_attention_per_index_fg, max_attention_per_index_bg,
                      dist_x, dist_y, return_losses = False):
        """ Computes the attend-and-excite loss using the maximum attention value for each token. """
        losses_fg = [max(0, 1. - curr_max) for curr_max in max_attention_per_index_fg]
        losses_bg = [max(0, curr_max) for curr_max in max_attention_per_index_bg]
        loss = sum(losses_fg) + sum(losses_bg) + sum(dist_x) + sum(dist_y)
        if return_losses:
            return max(losses_fg), losses_fg
        else:
            return max(losses_fg), loss

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """ Update the latent according to the computed loss. """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond
        return latents

    def _perform_iterative_refinement_step(self,
                                           latents: torch.Tensor,
                                           indices_to_alter,
                                           loss_fg: torch.Tensor,
                                           threshold: float,
                                           text_embeddings: torch.Tensor,
                                           uc_text_embeddings: torch.Tensor,
                                           attention_store: AttentionStore,
                                           step_size: float,
                                           t: int,
                                           inpainting_extra_input,
                                           attention_res: int = 16,
                                           smooth_attentions: bool = True,
                                           sigma: float = 0.5,
                                           kernel_size: int = 3,
                                           max_refinement_steps: int = 20,
                                           normalize_eot: bool = False,
                                           bbox = None,
                                           config=None,
                                           ):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent
        code according to our loss objective until the given threshold is reached for all tokens.
        """
        check_in_graph(latents, 'first_in perform')
        iteration = 0
        target_loss = max(0, 1. - threshold)
        while loss_fg > target_loss:
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            #latents = latents.clone().requires_grad_(True)
            inputs = dict(x=latents, timesteps=t, context=text_embeddings, inpainting_extra_input=inpainting_extra_input, grounding_extra_input = None)
            noise_pred_text = self.model(inputs)
            self.model.zero_grad()

            # Get max activation value for each subject token
            max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y = self._aggregate_and_get_max_attention_per_token(
                attention_store=attention_store,
                indices_to_alter=indices_to_alter,
                attention_res=attention_res,
                smooth_attentions=smooth_attentions,
                sigma=sigma,
                kernel_size=kernel_size,
                normalize_eot=normalize_eot,
                bbox=bbox,
                config=config,
                )

            loss_fg, losses_fg = self._compute_loss(max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y, return_losses=True)

            if loss_fg != 0:
                latents = self._update_latent(latents, loss_fg, step_size)

            with torch.no_grad():
                inputs = dict(x=latents, timesteps=t, context=uc_text_embeddings, inpainting_extra_input=inpainting_extra_input, grounding_extra_input = None)
                noise_pred_uncond = self.model(inputs)
                inputs = dict(x=latents, timesteps=t, context=text_embeddings, inpainting_extra_input=inpainting_extra_input, grounding_extra_input = None)
                noise_pred_text = self.model(inputs)

            try:
                low_token = np.argmax([l.item() if type(l) != int else l for l in losses_fg])
            except Exception as e:
                print(e)  # catch edge case :)

                low_token = np.argmax(losses_fg)

            #low_word = self.tokenizer.decode(text_input.input_ids[0][indices_to_alter[low_token]])
            # print(f'\t Try {iteration}. {low_word} has a max attention of {max_attention_per_index_fg[low_token]}')

            if iteration >= max_refinement_steps:
                # print(f'\t Exceeded max number of iterations ({max_refinement_steps})! '
                #       f'Finished with a max attention of {max_attention_per_index_fg[low_token]}')
                break

        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        #latents = latents.clone().requires_grad_(True)
        inputs = dict(x=latents, timesteps=t, context=text_embeddings, inpainting_extra_input=inpainting_extra_input, grounding_extra_input = None)
        noise_pred_text = self.model(inputs)
        self.model.zero_grad()

        check_in_graph(latents, 'out while in perform')

        # Get max activation value for each subject token
        max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y = self._aggregate_and_get_max_attention_per_token(
            attention_store=attention_store,
            indices_to_alter=indices_to_alter,
            attention_res=attention_res,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot,
            bbox=bbox,
            config=config,
        )
        loss_fg, losses_fg = self._compute_loss(max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y, return_losses=True)
        # print(f"\t Finished with loss of: {loss_fg}")
        return loss_fg, latents, max_attention_per_index_fg
    
    
    # Attend and Excite

    def _compute_max_attention_per_index_attend(self,
                                         attention_maps: torch.Tensor,
                                         indices_to_alter,
                                         smooth_attentions: bool = False,
                                         sigma: float = 0.5,
                                         kernel_size: int = 3,
                                         normalize_eot: bool = False):
        """ Computes the maximum attention value for each of the tokens we wish to alter. """
        last_idx = -1
        if normalize_eot:
            prompt = self.prompt
            if isinstance(self.prompt, list):
                prompt = self.prompt[0]
            last_idx = len(self.tokenizer(prompt)['input_ids']) - 1
        attention_for_text = attention_maps[:, :, 1:last_idx]
        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # Shift indices since we removed the first token
        indices_to_alter = [index - 1 for index in indices_to_alter]

        # Extract the maximum values
        max_indices_list = []
        for i in indices_to_alter:
            image = attention_for_text[:, :, i]
            if smooth_attentions:
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                image = smoothing(input).squeeze(0).squeeze(0)
            max_indices_list.append(image.max())
        
        return max_indices_list

    def _aggregate_and_get_max_attention_per_token_attend(self, attention_store: AttentionStore,
                                                   indices_to_alter,
                                                   attention_res: int = 16,
                                                   smooth_attentions: bool = False,
                                                   sigma: float = 0.5,
                                                   kernel_size: int = 3,
                                                   normalize_eot: bool = False,
                                                   latents=None):
        """ Aggregates the attention for each token and computes the max activation value for each token to alter. """
        attention_maps = boxdiff_aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0)
        

        max_attention_per_index = self._compute_max_attention_per_index_attend(
            attention_maps=attention_maps,
            indices_to_alter=indices_to_alter,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot)
        
        #grad_cond = torch.autograd.grad(max_attention_per_index[0].requires_grad_(True), attention_maps.requires_grad_(True), retain_graph=True)[0]
        pdb.set_trace()

        return max_attention_per_index

    @staticmethod
    def _compute_loss_attend(max_attention_per_index, return_losses: bool = False) -> torch.Tensor:
        """ Computes the attend-and-excite loss using the maximum attention value for each token. """
        losses = [max(0, 1. - curr_max) for curr_max in max_attention_per_index]
        loss = max(losses)

        #grad_cond = torch.autograd.grad(loss.requires_grad_(True), losses, retain_graph=True)[0]

        #pdb.set_trace()

        if return_losses:
            return loss, losses
        else:
            return loss

    @staticmethod
    def _update_latent_attend(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """ Update the latent according to the computed loss. """


        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]

        latents = latents - step_size * grad_cond
        return latents

    def _perform_iterative_refinement_step_attend(self,
                                           latents: torch.Tensor,
                                           indices_to_alter,
                                           loss: torch.Tensor,
                                           threshold: float,
                                           text_embeddings: torch.Tensor,
                                           uc_text_embeddings: torch.Tensor,
                                           attention_store: AttentionStore,
                                           step_size: float,
                                           t: int,
                                           inpainting_extra_input,
                                           attention_res: int = 16,
                                           smooth_attentions: bool = True,
                                           sigma: float = 0.5,
                                           kernel_size: int = 3,
                                           max_refinement_steps: int = 20,
                                           normalize_eot: bool = False):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent
        code according to our loss objective until the given threshold is reached for all tokens.
        """
        check_in_graph(latents, 'first_in perform')
        iteration = 0
        target_loss = max(0, 1. - threshold)
        while loss > target_loss:
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            #latents = latents.clone().requires_grad_(True)
            inputs = dict(x=latents, timesteps=t, context=text_embeddings, inpainting_extra_input=inpainting_extra_input, grounding_extra_input = None)
            noise_pred_text = self.model(inputs)
            self.model.zero_grad()

            # Get max activation value for each subject token
            max_attention_per_index = self._aggregate_and_get_max_attention_per_token_attend(
                attention_store=attention_store,
                indices_to_alter=indices_to_alter,
                attention_res=attention_res,
                smooth_attentions=smooth_attentions,
                sigma=sigma,
                kernel_size=kernel_size,
                normalize_eot=normalize_eot
                )

            loss, losses = self._compute_loss_attend(max_attention_per_index, return_losses=True)

            if loss != 0:
                latents = self._update_latent_attend(latents, loss, step_size)

            with torch.no_grad():
                inputs = dict(x=latents, timesteps=t, context=uc_text_embeddings, inpainting_extra_input=inpainting_extra_input, grounding_extra_input = None)
                noise_pred_text = self.model(inputs)
                inputs = dict(x=latents, timesteps=t, context=text_embeddings, inpainting_extra_input=inpainting_extra_input, grounding_extra_input = None)
                noise_pred_text = self.model(inputs)

            try:
                low_token = np.argmax([l.item() if type(l) != int else l for l in losses])
            except Exception as e:
                print(e)  # catch edge case :)
                low_token = np.argmax(losses)

            #low_word = self.tokenizer.decode(text_input.input_ids[0][indices_to_alter[low_token]])
            #print(f'\t Try {iteration}. {low_word} has a max attention of {max_attention_per_index[low_token]}')

            if iteration >= max_refinement_steps:
                print(f'\t Exceeded max number of iterations ({max_refinement_steps})! '
                      f'Finished with a max attention of {max_attention_per_index[low_token]}')
                break

        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        #latents = latents.clone().requires_grad_(True)
        inputs = dict(x=latents, timesteps=t, context=text_embeddings, inpainting_extra_input=inpainting_extra_input, grounding_extra_input = None)
        noise_pred_text = self.model(inputs)
        self.model.zero_grad()

        # Get max activation value for each subject token
        max_attention_per_index = self._aggregate_and_get_max_attention_per_token_attend(
            attention_store=attention_store,
            indices_to_alter=indices_to_alter,
            attention_res=attention_res,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot)
        loss, losses = self._compute_loss_attend(max_attention_per_index, return_losses=True)
        print(f"\t Finished with loss of: {loss}")
        return loss, latents, max_attention_per_index

