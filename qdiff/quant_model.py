import logging
import torch.nn as nn
from qdiff.quant_block import get_specials, BaseQuantBlock
from qdiff.quant_block import QuantBasicTransformerBlock, QuantResBlock
from qdiff.quant_block import QuantQKMatMul, QuantSMVMatMul, QuantBasicTransformerBlock#, QuantAttnBlock
from qdiff.quant_layer import QuantModule, StraightThrough, TimewiseUniformQuantizer
from ldm.modules.attention import BasicTransformerBlock

logger = logging.getLogger(__name__)


class QuantModel(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, **kwargs):
        super().__init__()
        self.model = model
        timewise = kwargs.get('timewise', True)
        self.timewise = timewise
        list_timesteps = kwargs.get('list_timesteps', 50)
        self.timesteps = list_timesteps
        self.sm_abit = kwargs.get('sm_abit', 8)
        self.in_channels = model.in_channels
        if hasattr(model, 'image_size'):
            self.image_size = model.image_size
        self.specials = get_specials(act_quant_params['leaf_param'])
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params, timewise, list_timesteps)
        self.quant_block_refactor(self.model, weight_quant_params, act_quant_params, timewise, list_timesteps)

    def quant_module_refactor(self, module, weight_quant_params, act_quant_params, timewise, list_timesteps):
        """
        Recursively replace the normal layers (conv2D, conv1D, Linear etc.) to QuantModule
        :param module: nn.Module with nn.Conv2d, nn.Conv1d, or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear)): # nn.Conv1d
                setattr(module, name, QuantModule(
                    child_module, weight_quant_params, act_quant_params, timewise=timewise, list_timesteps=list_timesteps))
                prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params, timewise, list_timesteps)

    def quant_block_refactor(self, module, weight_quant_params, act_quant_params, timewise, list_timesteps):
        for name, child_module in module.named_children():
            if type(child_module) in self.specials:
                if self.specials[type(child_module)] in [QuantBasicTransformerBlock]:
                    setattr(module, name, self.specials[type(child_module)](child_module,
                        act_quant_params, sm_abit=self.sm_abit, timewise=timewise, list_timesteps=list_timesteps))
                elif self.specials[type(child_module)] == QuantSMVMatMul:
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params, sm_abit=self.sm_abit, timewise=timewise, list_timesteps=list_timesteps))
                elif self.specials[type(child_module)] == QuantQKMatMul:
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params, timewise=timewise, list_timesteps=list_timesteps))
                else:
                    setattr(module, name, self.specials[type(child_module)](child_module, 
                        act_quant_params))
            else:
                self.quant_block_refactor(child_module, weight_quant_params, act_quant_params, timewise, list_timesteps)


    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def set_timestep(self, t):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_timestep(t)

    def forward(self, x, timesteps=None, context=None):
        return self.model(x, timesteps, context)
    
    def set_running_stat(self, running_stat: bool, sm_only=False):
        # Only consider timewise=True here
        for m in self.model.modules():
            if isinstance(m, QuantBasicTransformerBlock):
                if sm_only:
                    m.attn1.act_quantizer_w.set_running_stat(running_stat)
                    m.attn2.act_quantizer_w.set_running_stat(running_stat)
                else:
                    m.attn1.act_quantizer_q.set_running_stat(running_stat)
                    m.attn1.act_quantizer_k.set_running_stat(running_stat)
                    m.attn1.act_quantizer_v.set_running_stat(running_stat)
                    m.attn1.act_quantizer_w.set_running_stat(running_stat)
                    m.attn2.act_quantizer_q.set_running_stat(running_stat)
                    m.attn2.act_quantizer_k.set_running_stat(running_stat)
                    m.attn2.act_quantizer_v.set_running_stat(running_stat)
                    m.attn2.act_quantizer_w.set_running_stat(running_stat)
            if isinstance(m , QuantQKMatMul):
                m.act_quantizer_q.set_running_stat(running_stat)
                m.act_quantizer_k.set_running_stat(running_stat)
            if isinstance(m , QuantSMVMatMul):
                m.act_quantizer_v.set_running_stat(running_stat)
                m.act_quantizer_w.set_running_stat(running_stat)
            if isinstance(m, QuantModule) and not sm_only:
                m.set_running_stat(running_stat)

    def save_dict_params(self):
        for m in self.model.modules():
            if isinstance(m, TimewiseUniformQuantizer):
                m.save_dict_params()

    def load_dict_params(self):
        for m in self.model.modules():
            if isinstance(m, TimewiseUniformQuantizer):
                m.load_dict_params()

    def set_grad_ckpt(self, grad_ckpt: bool):
        for name, m in self.model.named_modules():
            if isinstance(m, (QuantBasicTransformerBlock, BasicTransformerBlock)):
                # logger.info(name)
                m.checkpoint = grad_ckpt
            # elif isinstance(m, QuantResBlock):
                # logger.info(name)
                # m.use_checkpoint = grad_ckpt
