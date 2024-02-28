import torch
import argparse, os, datetime, gc, yaml
# import linklink as link
import logging
from qdiff.quant_layer import QuantModule, StraightThrough, lp_loss, new_lp_loss, TimewiseUniformQuantizer
from qdiff.quant_block import QuantBasicTransformerBlock
from qdiff.quant_model import QuantModel
from qdiff.block_recon import LinearTempDecay
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.utils import save_grad_data, save_inp_oup_data
import os
import torch.nn.functional as F
from tqdm import tqdm, trange
from PIL import Image
from torchvision.utils import make_grid
import time
from torch import autocast
from contextlib import nullcontext
from einops import rearrange
import numpy as np
import copy
from torch import einsum
logger = logging.getLogger(__name__)


def pd_optimize_timewise(model, cali_data, opt, logger, iters: int = 20000, timesteps: list=None, outpath: str=None,):
    for m in model.modules():
        if isinstance(m, TimewiseUniformQuantizer):
            delta_data = m.delta_list.data
            delattr(m, "delta_list")
            m.delta_list = delta_data
    opt_params_w = []

    for name, module in model.named_modules():
        if isinstance(module, (QuantModule)):
            if "qkv" in name:
                module.weight.requires_grad = True
                opt_params_w += [module.weight]
                if module.bias is not None:
                    module.bias.requires_grad = True
                    opt_params_w += [module.bias]
    optimizer_w = torch.optim.Adam(opt_params_w, lr=1e-5)

    scheduler_w = None
    cali_data = (cali_data[0].cpu(), cali_data[1].cpu())
    # Get intermediate activations
    activation = {}
    def get_output(name):
        def hook(model, input, output):
            activation[name] = output
        return hook
    loss_func = torch.nn.MSELoss()

    layer_list = ["qkv"]
    save_folder = "output/fp_activations_timewise0"

    for epoch in range(20):
        for t in timesteps:
            logger.info(f"[Epoch {epoch}] Current timestep: {t}")
            model.set_timestep(t)
            opt_params_a = []
            for n, m in model.named_modules():
                # No timeembed param retraining
                if "act_quantizer" in n and "emb" not in n:
                    m.current_delta.requires_grad = True
                    opt_params_a += [m.current_delta]
            optimizer_a = torch.optim.Adam(opt_params_a, lr=1e-4)
            
            scheduler_a = None
            optimizer_a.zero_grad()
            optimizer_w.zero_grad()

            get_labels_timewise(model, cali_data, t, layer_list, save_folder)

            hook_handles = []
            for n, module in model.named_modules():
                if isinstance(module, (QuantModule)):
                    for layer_name in layer_list:
                        if layer_name in n: 
                            handle = module.register_forward_hook(get_output(n))
                            hook_handles.append(handle)
                
            num = len(os.listdir(save_folder))
            model.train()
            model.set_quant_state(True, True)
            for j in range(num):
                
                activation_fp = torch.load(os.path.join(save_folder, f"activation_{j}.pth"))
                for i in range(activation_fp["input"][0].shape[0]):
                    err = 0
                    output_quant = model(activation_fp["input"][0][i].unsqueeze(0).cuda(), 
                                        activation_fp["input"][1][i].unsqueeze(0).cuda())
                    for k in activation.keys():
                        head = int(activation_fp[k].shape[0] / activation_fp["input"][0].shape[0]) # designed for act_quantizer_w/v's output
                        # err += loss_func(activation[k], activation_fp[k][i*head:(i+1)*head].cuda())
                        err += loss_func(output_quant, activation_fp["output"][i].unsqueeze(0).cuda())
                    
                    err /= activation_fp["input"][0].shape[0]
                    err.backward()
             
                # Using gradient accumulation to deal with small batch size
                if j % 2 == 0:
                    logger.info(f"Error: {err}")
                optimizer_w.step()
                optimizer_a.step()
                if scheduler_w:
                    scheduler_w.step()
                if scheduler_a:
                    scheduler_a.step()
               
                optimizer_a.zero_grad()
                optimizer_w.zero_grad()
              
            for handle in hook_handles:
                handle.remove()
            torch.cuda.empty_cache()
    model.eval()
    

def get_labels_timewise(model, cali_data, timestep, layer_list, save_folder):
    model.eval()
    model.set_quant_state(False, False)
    
    def get_output(name):
        def hook(model, input, output): 
            activation[name] = output.detach().cpu()
        return hook
    hook_handles = []   
    for n, module in model.named_modules():
        if isinstance(module, (QuantModule)):
            for layer_name in layer_list:
                if layer_name in n: 
                    handle = module.register_forward_hook(get_output(n))
                    hook_handles.append(handle)
    cali_xs, cali_ts = cali_data
    
    model.set_timestep(timestep)
    t_idx = torch.where(cali_ts == timestep)[0]
    cali_xs_t = cali_xs[t_idx]
    cali_ts_t = cali_ts[t_idx]

    print("Generating FP outputs for timestep {}".format(timestep))
    sample_idx = torch.randperm(cali_xs_t.size(0))[:64]
    b_size = 4
    with torch.no_grad():
        for k in trange(16):
            activation = {}
            idx = sample_idx[k*b_size:(k+1)*b_size]
            output_fp = model(cali_xs_t[idx].cuda(), cali_ts_t[idx].cuda())
            activation['input'] = (cali_xs_t[idx].detach().cpu(), cali_ts_t[idx].detach().cpu())
            activation['output'] = output_fp.detach().cpu()
            torch.save(activation, os.path.join(save_folder, f"activation_{k}.pth"))
    for handle in hook_handles:
        handle.remove()
    print("Generation finished")


def pd_optimize_timeembed(model, cali_data, opt, logger, iters: int = 20000, timesteps: list=None, outpath: str=None,):
    for m in model.modules():
        if isinstance(m, TimewiseUniformQuantizer):
            delta_data = m.delta_list.data
            delattr(m, "delta_list")
            m.delta_list = delta_data
    opt_params_w = []
    
    for name, module in model.named_modules():
        if isinstance(module, (QuantModule)):
            if "emb_layer" in name:
                logger.info(f"{name} added") 
                module.weight.requires_grad = True
                opt_params_w += [module.weight]
                if module.bias is not None:
                    module.bias.requires_grad = True
                    opt_params_w += [module.bias]
                module.weight_quantizer.alpha.requires_grad = True
    optimizer_w = torch.optim.Adam(opt_params_w, lr=1e-5)
   
    cali_data = (cali_data[0].cpu(), cali_data[1].cpu())
    # Get intermediate activations
    activation = {}
    def get_output(name):
        def hook(model, input, output):
            activation[name] = output
        return hook
    loss_func = torch.nn.MSELoss()

    layer_list = ["emb_layer"]
    save_folder = "output/fp_activations_timeembed/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for t in timesteps:
        model.set_timestep(t)
        opt_params_a = []
        for n, m in model.named_modules():
            if "act_quantizer" in n and "emb_layer" in n:
                m.current_delta.requires_grad = True
                opt_params_a += [m.current_delta]
        optimizer_a = torch.optim.Adam(opt_params_a, lr=1e-5)
        scheduler_a = None
        scheduler_w = None
        optimizer_a.zero_grad()
        optimizer_w.zero_grad()
        get_timeembed_labels(model, cali_data, t, layer_list, save_folder)
        model.train()

        hook_handles = []
        for n, module in model.named_modules():
            if isinstance(module, (QuantModule)):
                for layer_name in layer_list:
                    if layer_name in n: 
                        handle = module.register_forward_hook(get_output(n))
                        hook_handles.append(handle)
        num = len(os.listdir(save_folder))
        model.set_quant_state(True, True)
        for epoch in range(200):
            total_err = 0
            for j in range(num):
                activation_fp = torch.load(os.path.join(save_folder, f"activation_{j}.pth"))
                for i in range(activation_fp["input"][0].shape[0]):
                    err = 0
                    output_quant = model(activation_fp["input"][0][i].unsqueeze(0).cuda(), 
                                        activation_fp["input"][1][i].unsqueeze(0).cuda())
                    for k in activation.keys():
                        head = int(activation_fp[k].shape[0] / activation_fp["input"][0].shape[0]) # designed for act_quantizer_w/v's output
                        err += loss_func(activation[k], activation_fp[k][i*head:(i+1)*head].cuda())
                        err += loss_func(output_quant, activation_fp["output"][i].unsqueeze(0).cuda())
                    
                    err /= activation_fp["input"][0].shape[0]
                    err.backward()
                    total_err += err

                # Using gradient accumulation to deal with small batch size
                optimizer_w.step()
                optimizer_a.step()
                if scheduler_w:
                    scheduler_w.step()
                if scheduler_a:
                    scheduler_a.step()
                optimizer_a.zero_grad()
                optimizer_w.zero_grad()
            if epoch % 10 == 0:
                logger.info(f"[Timestep {t}, Epoch {epoch}] Error: {total_err/num}")
            
        for handle in hook_handles:
            handle.remove()
        torch.cuda.empty_cache()
    
    # Reset the params to not require grad
    for name, module in model.named_modules():
        if isinstance(module, (QuantModule)):
            if "emb_layer" in name:
                logger.info(f"{name} added") 
                module.weight.requires_grad = False
                opt_params_w += [module.weight]
                if module.bias is not None:
                    module.bias.requires_grad = False
                    opt_params_w += [module.bias]
                module.weight_quantizer.alpha.requires_grad = False


    model.eval()


def get_timeembed_labels(model, cali_data, timestep, layer_list, save_folder):
    model.eval()
    model.set_quant_state(False, False)
    
    def get_output(name):
        def hook(model, input, output): 
            # activation[name] = input[0]
            activation[name] = output.detach().cpu()
        return hook
    hook_handles = []   
    for n, module in model.named_modules():
        if isinstance(module, (QuantModule)):
            for layer_name in layer_list:
                if layer_name in n: 
                    handle = module.register_forward_hook(get_output(n))
                    hook_handles.append(handle)
    cali_xs, cali_ts = cali_data
    
    model.set_timestep(timestep)
    t_idx = torch.where(cali_ts == timestep)[0]
    cali_xs_t = cali_xs[t_idx]
    cali_ts_t = cali_ts[t_idx]
    print("Generating FP outputs for timestep {}".format(timestep))
    sample_idx = torch.randperm(cali_xs_t.size(0))[:64]
    b_size = 2
    with torch.no_grad():
        for k in trange(4):
            activation = {}
            idx = sample_idx[k*b_size:(k+1)*b_size]
            output_fp = model(cali_xs_t[idx].cuda(), cali_ts_t[idx].cuda())
            activation['input'] = (cali_xs_t[idx].detach().cpu(), cali_ts_t[idx].detach().cpu())
            activation['output'] = output_fp.detach().cpu()
            torch.save(activation, os.path.join(save_folder, f"activation_{k}.pth"))
    for handle in hook_handles:
        handle.remove()
    print("Generation finished")


class LossFunction:
    def __init__(self,
                 soft_targets : dict = {},
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.):

        self.soft_targets = soft_targets
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, pred, tgt, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            round_vals = self.soft_targets
            round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        # if self.count % 500 == 0:
        #     logger.info('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
        #           float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss

