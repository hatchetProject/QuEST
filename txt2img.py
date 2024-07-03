import argparse, os, datetime, gc, yaml
import logging
import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
# from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
import torch
import torch.nn as nn
from torch import autocast
from contextlib import nullcontext
from collections import Counter
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler, PLMSSampler_Timewise
from qdiff import (
    QuantModel, QuantModule, BaseQuantBlock, 
    block_reconstruction, layer_reconstruction,
)
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.quant_layer import UniformAffineQuantizer, TimewiseUniformQuantizer, ActUniformQuantizer
from qdiff.utils import resume_cali_model, get_train_samples
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from qdiff.quant_block import QuantBasicTransformerBlock, QuantResBlock, QuantQKMatMul, QuantSMVMatMul
from qdiff.post_layer_recon_sd import *
import sys
import shutil
import copy

logger = logging.getLogger(__name__)

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    logging.info(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        logging.info(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        logging.info("missing keys:")
        logging.info(m)
    if len(u) > 0 and verbose:
        logging.info("unexpected keys:")
        logging.info(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    # linear quantization configs
    parser.add_argument(
        "--ptq", action="store_true", help="apply post-training quantization"
    )
    parser.add_argument(
        "--quant_act", action="store_true", 
        help="if to quantize activations when ptq==True"
    )
    parser.add_argument(
        "--weight_bit",
        type=int,
        default=8,
        help="int bit for weight quantization",
    )
    parser.add_argument(
        "--act_bit",
        type=int,
        default=8,
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--quant_mode", type=str, default="symmetric", 
        choices=["linear", "squant", "qdiff"], 
        help="quantization mode to use"
    )

    # qdiff specific configs
    parser.add_argument(
        "--cali_st", type=int, default=1, 
        help="number of timesteps used for calibration"
    )
    parser.add_argument(
        "--cali_batch_size", type=int, default=32, 
        help="batch size for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_n", type=int, default=1024, 
        help="number of samples for each timestep for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_iters", type=int, default=20000, 
        help="number of iterations for each qdiff reconstruction"
    )
    parser.add_argument('--cali_iters_a', default=5000, type=int, 
        help='number of iteration for LSQ')
    parser.add_argument('--cali_lr', default=4e-4, type=float, 
        help='learning rate for LSQ')
    parser.add_argument('--cali_p', default=2.4, type=float, 
        help='L_p norm minimization for LSQ')
    parser.add_argument(
        "--cali_ckpt", type=str,
        help="path for calibrated model ckpt"
    )
    parser.add_argument(
        "--cali_data_path", type=str, default="sd_coco_sample1024_allst.pt",
        help="calibration dataset name"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume the calibrated qdiff model"
    )
    parser.add_argument(
        "--resume_w", action="store_true",
        help="resume the calibrated qdiff model weights only"
    )
    parser.add_argument(
        "--cond", action="store_true",
        help="whether to use conditional guidance"
    )
    parser.add_argument(
        "--no_grad_ckpt", action="store_true",
        help="disable gradient checkpointing"
    )
    parser.add_argument(
        "--split", action="store_true",
        help="use split strategy in skip connection"
    )
    parser.add_argument(
        "--running_stat", action="store_true",
        help="use running statistics for act quantizers"
    )
    parser.add_argument(
        "--rs_sm_only", action="store_true",
        help="use running statistics only for softmax act quantizers"
    )
    parser.add_argument(
        "--sm_abit",type=int, default=8,
        help="attn softmax activation bit"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print out info like quantized model arch"
    )
    opt = parser.parse_args()
    return opt


def main():
    opt = get_argparse()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = os.path.join(opt.outdir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(outpath)

    log_path = os.path.join(outpath, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        # sampler = PLMSSampler(model)
        sampler = PLMSSampler_Timewise(model)
    else:
        sampler = DDIMSampler(model)

    assert(opt.cond)
    is_recon = False
    if opt.ptq:
        if opt.split:
            setattr(sampler.model.model.diffusion_model, "split", True)
        if opt.quant_mode == 'qdiff':
            wq_params = {'n_bits': opt.weight_bit, 'channel_wise': True, 'scale_method': 'mse'}
            aq_params = {'n_bits': opt.act_bit, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param':  opt.quant_act}
            if opt.resume:
                logger.info('Load with min-max quick initialization')
                wq_params['scale_method'] = 'max'
                aq_params['scale_method'] = 'max'
            if opt.resume_w:
                wq_params['scale_method'] = 'max'
            # Tokenwise activation is necessary
            if opt.act_bit == 4:
                aq_params['channel_wise'] = True

            logger.info(f"Sampling data from {opt.cali_st} timesteps for calibration")
            sample_data = torch.load(opt.cali_data_path)
            cali_data = get_train_samples(opt, sample_data, opt.ddim_steps)

            del(sample_data)
            gc.collect()
            logger.info(f"Calibration data shape: {cali_data[0].shape} {cali_data[1].shape} {cali_data[2].shape}")
            timesteps = [k for k, v in Counter(list(np.array(cali_data[1]))).items()]
            print("Number of timesteps and values:", len(timesteps), timesteps)

            qnn = QuantModel(
                model=sampler.model.model.diffusion_model, weight_quant_params=wq_params, act_quant_params=aq_params,
                act_quant_mode="qdiff", sm_abit=opt.sm_abit, timewise=True, list_timesteps=timesteps)
            qnn.cuda()
            qnn.eval()

            # Set the first and last layer to be 8 bit
            for n, m in qnn.named_modules():
                if isinstance(m, QuantModule):
                    if ".out.2" in n or "input_blocks.0.0" in n:
                        print(n)
                        for m_act in m.act_quantizer.quantizer_dict.values():
                            m_act.n_bits = 8
                            m_act.n_levels = 2 ** 8

            if opt.no_grad_ckpt:
                logger.info('Not use gradient checkpointing for transformer blocks')
                qnn.set_grad_ckpt(False)
            
            if opt.resume:
                cali_data_resume = (torch.randn(1, 4, 64, 64), torch.randint(0, 1000, (1,)), torch.randn(1, 77, 768))
                resume_cali_model(qnn, opt.cali_ckpt, cali_data_resume, opt.quant_act, cond=opt.cond, timesteps=timesteps)
                qnn.set_quant_state(True, True)
            else:
                cali_xs, cali_ts, cali_cs = cali_data
                cali_xs = cali_xs.contiguous()
                cali_ts = cali_ts.contiguous()
                cali_cs = cali_cs.contiguous()
                if opt.resume_w:
                    resume_cali_model(qnn, opt.cali_ckpt, cali_data, False, cond=opt.cond, timesteps=timesteps)
                else:
                    logger.info("Initializing weight quantization parameters")
                    qnn.set_quant_state(True, False) # enable weight quantization, disable act quantization
                    qnn.set_timestep(timesteps[0])
                    _ = qnn(cali_xs[:4].cuda(), cali_ts[:4].cuda(), cali_cs[:4].cuda())
                    torch.cuda.empty_cache()
                    logger.info("Initializing has done!")

                # Kwargs for weight rounding calibration
                kwargs = dict(cali_data=cali_data, batch_size=int(opt.cali_batch_size / 2), 
                            iters=opt.cali_iters, weight=0.01, asym=True, b_range=(20, 2),
                            warmup=0.2, act_quant=False, opt_mode='mse', cond=opt.cond, outpath=outpath)
                
                def recon_model(model):
                    '''
                    Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
                    '''
                    for name, module in model.named_children():
                        logger.info(f"{name} {isinstance(module, BaseQuantBlock)}")
                        if name == 'output_blocks':
                            logger.info("Finished calibrating input and mid blocks, saving temporary checkpoint...")
                            torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt.pth"))
                        if name.isdigit() and int(name) >= 9:
                            logger.info(f"Saving temporary checkpoint at {name}...")
                            torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt.pth"))
                        
                        if isinstance(module, QuantModule):
                            if module.ignore_reconstruction is True:
                                logger.info('Ignore reconstruction of layer {}'.format(name))
                                continue
                            else:
                                logger.info('Reconstruction for layer {}'.format(name))
                                layer_reconstruction(qnn, module, **kwargs)
                        elif isinstance(module, BaseQuantBlock):
                            if module.ignore_reconstruction is True:
                                logger.info('Ignore reconstruction of block {}'.format(name))
                                continue
                            else:
                                logger.info('Reconstruction for block {}'.format(name))
                                block_reconstruction(qnn, module, **kwargs)
                        else:
                            recon_model(module)

                if not opt.resume_w:
                    logger.info("Doing weight calibration")
                    recon_model(qnn)
                    is_recon = True
                    qnn.set_quant_state(weight_quant=True, act_quant=False)
                    torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt_w4.pth"))
                    torch.cuda.empty_cache()
                
                if opt.quant_act:
                    logger.info("UNet model")
                    logger.info("Doing activation calibration")
                    # Initialize activation quantization parameters as the same for each timestep as the same at the beginning
                    b_size = 8
                    qnn.set_quant_state(True, True)

                    if aq_params['channel_wise']:
                        logger.info("Channel-wise initialization for activation quantization parameters")
                    # TODO 0: Baseline init method, also can try different timesteps 
                    with torch.no_grad():
                        chosen_timestep = timesteps[0]
                        qnn.set_timestep(chosen_timestep)
                        inds = np.random.choice(cali_xs.shape[0], b_size, replace=False)
                        _ = qnn(cali_xs[inds].cuda(), cali_ts[inds].cuda(), cali_cs[inds].cuda())

                        # Copy initialized parameters
                        logger.info("Copying parameters to other timesteps")
                        for _, module in qnn.named_modules():
                            if isinstance(module, (QuantModule)):
                                for k in timesteps:
                                    if k != chosen_timestep:
                                        module.act_quantizer.quantizer_dict[k] = copy.deepcopy(module.act_quantizer.quantizer_dict[chosen_timestep])
                                        if module.split != 0:
                                            module.act_quantizer_0.quantizer_dict[k] = copy.deepcopy(module.act_quantizer_0.quantizer_dict[chosen_timestep])
                            elif isinstance(module, (QuantBasicTransformerBlock)):
                                for k in timesteps:
                                    if k != chosen_timestep:
                                        module.attn1.act_quantizer_q.quantizer_dict[k] = copy.deepcopy(module.attn1.act_quantizer_q.quantizer_dict[chosen_timestep])
                                        module.attn1.act_quantizer_k.quantizer_dict[k] = copy.deepcopy(module.attn1.act_quantizer_k.quantizer_dict[chosen_timestep])
                                        module.attn1.act_quantizer_v.quantizer_dict[k] = copy.deepcopy(module.attn1.act_quantizer_v.quantizer_dict[chosen_timestep])
                                        module.attn1.act_quantizer_w.quantizer_dict[k] = copy.deepcopy(module.attn1.act_quantizer_w.quantizer_dict[chosen_timestep])
                                        module.attn2.act_quantizer_q.quantizer_dict[k] = copy.deepcopy(module.attn2.act_quantizer_q.quantizer_dict[chosen_timestep])
                                        module.attn2.act_quantizer_k.quantizer_dict[k] = copy.deepcopy(module.attn2.act_quantizer_k.quantizer_dict[chosen_timestep])
                                        module.attn2.act_quantizer_v.quantizer_dict[k] = copy.deepcopy(module.attn2.act_quantizer_v.quantizer_dict[chosen_timestep])
                                        module.attn2.act_quantizer_w.quantizer_dict[k] = copy.deepcopy(module.attn2.act_quantizer_w.quantizer_dict[chosen_timestep])
                        logger.info("Copying done!")

                        if opt.running_stat:
                            logger.info('Running stat for activation quantization')
                            qnn.set_running_stat(True, opt.rs_sm_only)  # rs_sm_only=False
                            # None Timestep wise
                            qnn.set_timestep(chosen_timestep)
                            inds = np.arange(cali_xs.shape[0])
                            np.random.shuffle(inds)
                            for i in trange(int(cali_xs.size(0) / b_size)):
                                _ = qnn(cali_xs[inds[i * b_size:(i + 1) * b_size]].cuda(), 
                                    cali_ts[inds[i * b_size:(i + 1) * b_size]].cuda(),
                                    cali_cs[inds[i * b_size:(i + 1) * b_size]].cuda())
                            qnn.set_running_stat(False, opt.rs_sm_only)

                    # # Use this for activation calibration, which we do not recommend
                    # logger.info("Doing activation reconstruction")
                    # kwargs = dict(
                    #     cali_data=cali_data, batch_size=int(opt.cali_batch_size), iters=opt.cali_iters_a, act_quant=True, 
                    #     opt_mode='mse', lr=opt.cali_lr, p=opt.cali_p, cond=opt.cond, outpath=outpath, 
                    #     asym=False)
                    # recon_model(qnn)
                    # is_recon = True

                logger.info("Saving calibrated quantized UNet model")
                # Save quantization parameters as model parameters
                if opt.quant_act:
                    qnn.save_dict_params()
                
                # Save the quantized model
                for m in qnn.model.modules():
                    if isinstance(m, AdaRoundQuantizer):
                        m.zero_point = nn.Parameter(m.zero_point)
                        m.delta = nn.Parameter(m.delta)
                    elif isinstance(m, UniformAffineQuantizer) and opt.quant_act:
                        if m.zero_point is not None:
                            if not torch.is_tensor(m.zero_point):
                                m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                            else:
                                m.zero_point = nn.Parameter(m.zero_point)
                    elif isinstance(m, TimewiseUniformQuantizer) and opt.quant_act:
                        if m.zero_point_list is not None:
                            if not torch.is_tensor(m.zero_point_list):
                                m.zero_point_list = nn.Parameter(torch.tensor(float(m.zero_point_list)))
                            else:
                                m.zero_point_list = nn.Parameter(m.zero_point_list)
                torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt.pth"))
            torch.cuda.empty_cache()

            # You can do the following two steps individually
            if opt.quant_act:
                pd_optimize_timeembed(qnn, cali_data, sampler, opt, logger, iters=1000, timesteps=timesteps, outpath=outpath)
                pd_optimize_timewise(qnn, cali_data, sampler, opt, logger, iters=1000, timesteps=timesteps, outpath=outpath)

            qnn.set_quant_state(True, True)

            logger.info("Saving calibrated quantized UNet model")
            # Save quantization parameters as model parameters
            if opt.quant_act:
                qnn.save_dict_params()    
            # Save the quantized model
            for m in qnn.model.modules():
                if isinstance(m, AdaRoundQuantizer):
                    m.zero_point = nn.Parameter(m.zero_point)
                    m.delta = nn.Parameter(m.delta)
                elif isinstance(m, UniformAffineQuantizer) and opt.quant_act:
                    if m.zero_point is not None:
                        if not torch.is_tensor(m.zero_point):
                            m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                        else:
                            m.zero_point = nn.Parameter(m.zero_point)
                elif isinstance(m, TimewiseUniformQuantizer) and opt.quant_act:
                    if m.zero_point_list is not None:
                        if not torch.is_tensor(m.zero_point_list):
                            m.zero_point_list = nn.Parameter(torch.tensor(float(m.zero_point_list)))
                        else:
                            m.zero_point_list = nn.Parameter(m.zero_point_list.float())
            torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt.pth"))
            qnn.set_quant_state(True, True)
            sampler.model.model.diffusion_model = qnn

    if not opt.resume and is_recon:
        logger.info("Delete cached data to save disk usage")
        shutil.rmtree(os.path.join(outpath, "tmp_cached"))

    batch_size = min(opt.n_samples, 5)
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]] * (opt.n_samples//batch_size)
    else:
        logging.info(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    # write config out
    sampling_file = os.path.join(outpath, "sampling_config.yaml")
    sampling_conf = vars(opt)
    with open(sampling_file, 'a+') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    if opt.verbose:
        logger.info("UNet model")
        logger.info(model.model)

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([batch_size, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)  # [3, 77, 768]
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, intermediates = sampler.sample(S=opt.ddim_steps,  # intermediates records all of the intermediate samples
                                                         conditioning=c,
                                                         batch_size=batch_size,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code, 
                                                         log_every_t=1)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image = x_samples_ddim
                        # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                        if not opt.skip_save:
                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                # img = put_watermark(img, wm_encoder)
                                img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                base_count += 1

                        if not opt.skip_grid:
                            all_samples.append(x_checked_image_torch)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    # img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

    logging.info(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
