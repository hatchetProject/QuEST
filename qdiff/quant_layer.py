import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import numpy as np

logger = logging.getLogger(__name__)


class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def floor_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for floor operation.
    """
    return (x.floor() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()


def new_lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    Try different loss functions
    """
    return F.cosine_similarity(pred, tgt).mean()
    # return (pred-tgt).abs().pow(p).mean() + torch.norm(pred, p=1).abs().mean() #+ torch.norm(tgt, p=1).mean()


class TimewiseUniformQuantizer(nn.Module):
    def __init__(self, list_timesteps, **kwargs):
        """
        timestep: number of timesteps for diffusion process
        kwargs: arguments for UniformAffineQuantizer
        """
        super().__init__()
        self.quantizer_dict = {}
        self.list_timesteps = list_timesteps
        self.zero_point_list = None
        self.delta_list = None
        self.channel_num = None
        self.current_delta = None

        for t in list_timesteps:
            self.quantizer_dict[t] = ActUniformQuantizer(**kwargs)

    def forward(self, x, t):
        if self.channel_num is None:
            if len(x.shape) == 4:
                self.channel_num = x.shape[1]
                self.zero_point_list = torch.zeros((len(self.list_timesteps), self.channel_num, 1, 1))
                self.delta_list = torch.zeros((len(self.list_timesteps), self.channel_num, 1, 1))
            elif len(x.shape) == 3:
                self.channel_num = x.shape[-1]
                self.zero_point_list = torch.zeros((len(self.list_timesteps), 1, self.channel_num))
                self.delta_list = torch.zeros((len(self.list_timesteps), 1, self.channel_num))
            else:
                self.channel_num = x.shape[1]
                self.zero_point_list = torch.zeros((len(self.list_timesteps), self.channel_num))
                self.delta_list = torch.zeros((len(self.list_timesteps), self.channel_num))
        return self.quantizer_dict[t](x)

    def set_running_stat(self, running_stat: bool):
        for t in self.list_timesteps:
            self.quantizer_dict[t].running_stat = running_stat

    def save_dict_params(self):
        with torch.no_grad():
            for i, t in enumerate(self.list_timesteps):
                self.zero_point_list[i] = self.quantizer_dict[t].zero_point
                self.delta_list[i] = self.quantizer_dict[t].delta
        self.delta_list = nn.Parameter(self.delta_list)

    def load_dict_params(self):
        for i, t in enumerate(self.list_timesteps):
            self.quantizer_dict[t].zero_point = self.zero_point_list[i].cuda()
            self.quantizer_dict[t].delta = nn.Parameter(self.delta_list[i].cuda())
    
    def __repr__(self) -> str:
        s = f"TimewiseUniformQuantizer {len(self.quantizer_dict)} timesteps, now timestep {len(self.list_timesteps)}\n"
        for k, v in self.quantizer_dict.items():
            s += f"   timestep {k}: {v}\n"
            if len(s) > 100:
                break
        return s


class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False, always_zero: bool = False):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        # assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method
        self.running_stat = False
        self.always_zero = always_zero
        if self.leaf_param:
            self.x_min, self.x_max = None, None

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            if self.leaf_param:
                delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
                self.delta = torch.nn.Parameter(delta)
                # self.zero_point = torch.nn.Parameter(self.zero_point)
            else:
                self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.inited = True

        if self.running_stat:
            self.act_momentum_update(x)

        # start quantization
        # print(f"x shape {x.shape} delta shape {self.delta.shape} zero shape {self.zero_point.shape}")
        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        if self.sym:
            x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        else:
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta
        
        return x_dequant
    
    def act_momentum_update(self, x: torch.Tensor, act_range_momentum: float = 0.95):
        assert(self.inited)
        assert(self.leaf_param)

        x_min = x.data.min()
        x_max = x.data.max()
        self.x_min = self.x_min * act_range_momentum + x_min * (1 - act_range_momentum)
        self.x_max = self.x_max * act_range_momentum + x_max * (1 - act_range_momentum)

        if self.sym:
            # x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax
            delta = torch.max(self.x_min.abs(), self.x_max.abs()) / self.n_levels
        else:
            delta = (self.x_max - self.x_min) / (self.n_levels - 1) if not self.always_zero \
                else self.x_max / (self.n_levels - 1)
        
        delta = torch.clamp(delta, min=1e-8)
        if not self.sym:
            self.zero_point = (-self.x_min / delta).round() if not (self.sym or self.always_zero) else 0
        self.delta = torch.nn.Parameter(delta)

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 3:
                delta = delta.view(-1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        else:
            if self.leaf_param:
                self.x_min = x.data.min()
                self.x_max = x.data.max()

            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    # x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax
                    delta = x_absmax / self.n_levels
                else:
                    delta = float(x.max().item() - x.min().item()) / (self.n_levels - 1)
                if delta < 1e-8:
                    warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                    delta = 1e-8

                zero_point = round(-x_min / delta) if not (self.sym or self.always_zero) else 0
                delta = torch.tensor(delta).type_as(x)

            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                for i in range(80):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1) \
                            if not self.always_zero else new_max / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / delta).round() if not self.always_zero else 0
            else:
                raise NotImplementedError

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1) if not self.always_zero else max / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round() if not self.always_zero else 0
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        # assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)


class ActUniformQuantizer(nn.Module):
    """
    Based on RepQ
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False, always_zero: bool = False):
        super(ActUniformQuantizer, self).__init__()
        # assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.sym = symmetric
        self.always_zero = always_zero
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.leaf_param = leaf_param
        self.running_stat = False
        self.channel_wise = channel_wise
        self.x_min, self.x_max = None, None
        self.scale_method = scale_method
    
    def __repr__(self):
        s = super(ActUniformQuantizer, self).__repr__()
        s = "(" + s + " inited={}, channel_wise={})".format(self.inited, self.channel_wise)
        return s

    def forward(self, x: torch.Tensor):

        if self.inited is False:
            self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.delta = torch.nn.Parameter(self.delta)
            self.inited = True

        if self.running_stat:
            self.act_momentum_update(x)

        x_int = round_ste(x / self.delta) + self.zero_point
        # x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        if self.sym:
            x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        else:
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        return x_dequant
        
    def act_momentum_update(self, x: torch.Tensor, act_range_momentum: float = 0.95):
        assert(self.inited)
        
        if self.channel_wise:
            assert(self.channel_wise)
            if len(x.shape) == 3:
                # x_min = torch.amin(x, dim=(0, 2)).reshape(1, -1, 1)
                # x_max = torch.amax(x, dim=(0, 2)).reshape(1, -1, 1)
                x_min = torch.amin(x, dim=(0, 1)).reshape(1, 1, -1)
                x_max = torch.amax(x, dim=(0, 1)).reshape(1, 1, -1)
            elif len(x.shape) == 4:
                x_min = torch.amin(x, dim=(0, 2, 3)).reshape(1, -1, 1, 1)
                x_max = torch.amax(x, dim=(0, 2, 3)).reshape(1, -1, 1, 1)
            else:
                x_min = torch.amin(x, dim=0).reshape(1, -1)
                x_max = torch.amax(x, dim=0).reshape(1, -1)
            self.x_min = self.x_min * act_range_momentum + x_min * (1 - act_range_momentum)
            self.x_max = self.x_max * act_range_momentum + x_max * (1 - act_range_momentum)

            delta = (self.x_max - self.x_min) / (self.n_levels - 1)

            delta = torch.clamp(delta, min=1e-8)
            if not self.sym:
                self.zero_point = (-self.x_min / delta).round() if not (self.sym or self.always_zero) else 0
            self.delta = torch.nn.Parameter(delta)
        else:
            x_min = x.data.min()
            x_max = x.data.max()
            self.x_min = self.x_min * act_range_momentum + x_min * (1 - act_range_momentum)
            self.x_max = self.x_max * act_range_momentum + x_max * (1 - act_range_momentum)
            if self.sym:
                # x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax
                delta = torch.max(self.x_min.abs(), self.x_max.abs()) / self.n_levels
            else:
                delta = (self.x_max - self.x_min) / (self.n_levels - 1) if not self.always_zero \
                    else self.x_max / (self.n_levels - 1)
            
            delta = torch.clamp(delta, min=1e-8)
            if not self.sym:
                self.zero_point = (-self.x_min / delta).round() if not (self.sym or self.always_zero) else 0
            self.delta = torch.nn.Parameter(delta)

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            if len(x.shape) == 4:
                n_channels = x_clone.shape[1]
            elif len(x.shape) == 3: 
                n_channels = x_clone.shape[-1] # tokenwise quantization for linear layers
            else:
                n_channels = x_clone.shape[1]

            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=1)[0].max(dim=1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=0)[0]
            elif len(x.shape) == 3:
                # x_max = x_clone.abs().max(dim=0)[0].max(dim=1)[0]
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                if len(x.shape) == 3:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:, :, c], channel_wise=False)
                elif len(x.shape) == 4:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c,:,:], channel_wise=False)
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:, c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(1, -1, 1, 1)
                zero_point = zero_point.view(1, -1, 1, 1)
            elif len(x.shape) == 2:
                delta = delta.view(1, -1)
                zero_point = zero_point.view(1, -1)
            elif len(x.shape) == 3:
                # delta = delta.view(1, -1, 1)
                # zero_point = zero_point.view(1, -1, 1)
                delta = delta.view(1, 1, -1)
                zero_point = zero_point.view(1, 1, -1)
            else:
                raise NotImplementedError
        else:
            if self.leaf_param:
                self.x_min = x.data.min()
                self.x_max = x.data.max()

            if "max" in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)

                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    # x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax
                    delta = x_absmax / self.n_levels
                else:
                    delta = float(x.max().item() - x.min().item()) / (self.n_levels - 1)
                if delta < 1e-8:
                    # warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                    delta = 1e-8

                zero_point = round(-x_min / delta) if not (self.sym or self.always_zero) else 0
                delta = torch.tensor(delta).type_as(x)
            else:
                x_clone = x.clone().detach()
                x_max = x_clone.max()
                x_min = x_clone.min()
                best_score = 1e+10
                self.x_min = x_min
                self.x_max = x_max
                # RepQ method
                for pct in [0.999, 0.9999, 0.99999]:
                    try:
                        new_max = torch.quantile(x_clone.reshape(-1), pct)
                        new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                    except:
                        new_max = torch.tensor(np.percentile(
                            x_clone.reshape(-1).cpu(), pct * 100),
                            device=x_clone.device,
                            dtype=torch.float32)
                        new_min = torch.tensor(np.percentile(
                            x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                            device=x_clone.device,
                            dtype=torch.float32)   
                    x_q = self.quantize(x_clone, new_max, new_min)
                    score = lp_loss(x_clone, x_q, p=2, reduction='all')
                    # score = new_lp_loss(x_clone.view(x_clone.shape[0], -1), x_q.view(x_q.shape[0], -1))

                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        delta = torch.clamp(delta, min=1e-8)  # TODO: Added, examine effect
                        zero_point = (- new_min / delta).round()
        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        delta = torch.clamp(delta, min=1e-8)  # TODO: Added, examine effect
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        if self.sym:
            x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        else:
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        # assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits


class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear, nn.Conv1d], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False, act_quant_mode: str = 'qdiff', 
                 timewise=True, list_timesteps=[]):
        super(QuantModule, self).__init__()
        self.weight_quant_params = weight_quant_params
        self.act_quant_params = act_quant_params

        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        elif isinstance(org_module, nn.Conv1d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv1d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.act_quant_mode = act_quant_mode
        self.disable_act_quant = disable_act_quant
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**self.weight_quant_params)

        self.timewise = timewise
        self.timestep = None
        self.list_timesteps = list_timesteps
        if timewise:
            self.act_quantizer = TimewiseUniformQuantizer(self.list_timesteps, **self.act_quant_params)
        else:
            self.act_quantizer = ActUniformQuantizer(**self.act_quant_params)
        self.split = 0

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False
        self.extra_repr = org_module.extra_repr

        if self.bias is not None:
            self.bias = nn.Parameter(self.bias)
        self.weight = nn.Parameter(self.weight)

    def set_timestep(self, t):
        self.timestep = t
        self.act_quantizer.current_delta = self.act_quantizer.quantizer_dict[t].delta
        if self.split != 0:
            self.act_quantizer_0.current_delta = self.act_quantizer_0.quantizer_dict[t].delta


    def forward(self, input: torch.Tensor, split: int = 0):
        if split != 0 and self.split != 0:
            assert(split == self.split)
        elif split != 0:
            logger.info(f"split at {split}!")
            self.split = split
            self.set_split()

        # input_orig = input.clone()

        if not self.disable_act_quant and self.use_act_quant:
            assert self.timestep is not None
            if self.split != 0:
                if self.timewise:
                    input_0 = self.act_quantizer(input[:, :self.split, :, :], self.timestep)
                    input_1 = self.act_quantizer_0(input[:, self.split:, :, :], self.timestep)
                else:
                    input_0 = self.act_quantizer(input[:, :self.split, :, :])
                    input_1 = self.act_quantizer_0(input[:, self.split:, :, :])
                input = torch.cat([input_0, input_1], dim=1)
            else:
                if self.timewise:
                    input = self.act_quantizer(input, self.timestep)
                else:
                    input = self.act_quantizer(input)
        if self.use_weight_quant:
            if self.split != 0:
                weight_0 = self.weight_quantizer(self.weight[:, :self.split, ...])
                weight_1 = self.weight_quantizer_0(self.weight[:, self.split:, ...])
                weight = torch.cat([weight_0, weight_1], dim=1)
            else:
                weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias
        
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        out = self.activation_function(out)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def set_split(self):
        self.weight_quantizer_0 = UniformAffineQuantizer(**self.weight_quant_params)
        if self.timewise:
            self.act_quantizer_0 = TimewiseUniformQuantizer(self.list_timesteps, **self.act_quant_params)
        else:
            # self.act_quantizer_0 = UniformAffineQuantizer(**self.act_quant_params)
            self.act_quantizer_0 = ActUniformQuantizer(**self.act_quant_params)

    def set_running_stat(self, running_stat: bool):
        if self.timewise:
            self.act_quantizer.set_running_stat(running_stat)
            if self.split != 0:
                self.act_quantizer_0.set_running_stat(running_stat)
        else:
            self.act_quantizer.running_stat = running_stat
            if self.split != 0:
                self.act_quantizer_0.running_stat = running_stat

