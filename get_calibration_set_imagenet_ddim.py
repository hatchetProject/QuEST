"""
This is an example script to generate calibration set for ImageNet using DDIM sampling.
"""
import argparse, os, sys, gc, glob, datetime, yaml
import logging
import time
import numpy as np
from tqdm import trange
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from PIL import Image

import torch
import torch.nn as nn
import sys
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.util import instantiate_from_config

from qdiff import (
    QuantModel, QuantModule, BaseQuantBlock, 
    block_reconstruction, layer_reconstruction,
)
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.quant_layer import UniformAffineQuantizer
from qdiff.utils import resume_cali_model, get_train_samples

logger = logging.getLogger(__name__)

rescale = lambda x: (x + 1.) / 2.

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):


    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, shape, class_label, eta=1.0):
    ddim = DDIMSampler(model)

    n_samples_per_class = shape[0]
    ddim_steps = 20
    ddim_eta = 0.0
    scale = 3.0
    
    with model.ema_scope():
        uc = model.get_learned_conditioning(
            {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
            )
        print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
        xc = torch.tensor(n_samples_per_class*[class_label])
        c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
        
        samples_ddim, intermediates = ddim.sample(S=ddim_steps,
                                        conditioning=c,
                                        batch_size=n_samples_per_class,
                                        shape=[3, 64, 64],
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc, 
                                        eta=ddim_eta, 
                                        log_every_t=1)
    
    return samples_ddim, intermediates, c, uc


@torch.no_grad()
def convsample_dpm(model, steps, shape, eta=1.0):
    dpm = DPMSolverSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = dpm.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, class_label, vanilla=False, custom_steps=None, eta=1.0, dpm=False, logs=None):

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    # with model.ema_scope("Plotting"):
    t0 = time.time()
    if vanilla:
        sample, progrow = convsample(model, shape,
                                        make_prog_row=True)
    elif dpm:
        logger.info(f'Using DPM sampling with {custom_steps} sampling steps and eta={eta}')
        sample, intermediates = convsample_dpm(model,  steps=custom_steps, shape=shape,
                                                eta=eta)
    else:
        sample, intermediates, c, uc = convsample_ddim(model, shape=shape, class_label=class_label, eta=eta)
    
    xs = torch.stack(intermediates['x_inter'])[1:]
    ts = torch.stack(intermediates['ts'])
    c = c.unsqueeze(0).repeat(xs.shape[0], 1, 1, 1)
    uc = uc.unsqueeze(0).repeat(xs.shape[0], 1, 1, 1)

    if 'xs' in logs:
        logs['xs'] = torch.cat([logs['xs'], xs], dim=1)
        logs['ts'] = torch.cat([logs['ts'], ts], dim=1)
        logs['cs'] = torch.cat([logs['cs'], c], dim=1)
        logs['ucs'] = torch.cat([logs['ucs'], uc], dim=1)
    else:
        logs['xs'] = xs
        logs['ts'] = ts
        logs['cs'] = c
        logs['ucs'] = uc


def run(model, logdir, label, batch_size=50, vanilla=False, custom_steps=None, eta=None, 
    n_samples=50000, nplog=None, dpm=False, logs=None):
    if vanilla:
        logger.info(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        logger.info(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')


    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir,'*.png')))-1

    if model.cond_stage_model:
        all_images = []
        logger.info(f"Running class-conditional sampling for {n_samples} samples")
        for _ in trange(n_samples // batch_size, desc="Sampling Batches (conditional)"):
            make_convolutional_sample(model, batch_size=batch_size, class_label=label, 
                                             vanilla=vanilla, custom_steps=custom_steps,
                                             eta=eta, dpm=dpm, logs=logs)

def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume_base",
        type=str,
        nargs="?",
        help="load fp32 base model from logdir or checkpoint in logdir (will deprecate after direct quantized model loading implemented)",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=50000
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "--seed",
        type=int,
        # default=42,
        required=True,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fast dpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
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
        "--quant_mode", type=str, default="qdiff", 
        choices=["qdiff"], 
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
        "--a_sym", action="store_true",
        help="act quantizers use symmetric quantization (empirically helpful in some cases)"
    )
    parser.add_argument(
        "--a_min_max", action="store_true",
        help="act quantizers initialize with min-max (empirically helpful in some cases)"
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
        "--dpm", action="store_true",
        help="use dpm solver for sampling"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print out info like quantized model arch"
    )
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        logger.info(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = None
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None

    # fix random seed
    seed_everything(opt.seed)

    if not os.path.exists(opt.resume_base):
        raise ValueError("Cannot find {}".format(opt.resume_base))
    if os.path.isfile(opt.resume_base):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume_base.split('/')[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume_base.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume_base
    else:
        assert os.path.isdir(opt.resume_base), f"{opt.resume_base} is not a directory"
        logdir = opt.resume_base.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)

    logdir = os.path.join(logdir, "samples", now)
    os.makedirs(logdir)
    log_path = os.path.join(logdir, "run.log")
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
    print(config)

    logger.info(75 * "=")
    logger.info(f"Host {os.uname()[1]}")
    logger.info("logging to:")
    imglogdir = os.path.join(logdir, "img")
    numpylogdir = os.path.join(logdir, "numpy")

    os.makedirs(imglogdir)
    os.makedirs(numpylogdir)
    logger.info(logdir)
    logger.info(75 * "=")

    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    logger.info(f"global step: {global_step}")
    # logger.info("Switched to EMA weights")
    # model.model_ema.store(model.model.parameters())
    # model.model_ema.copy_to(model.model)

    # print(model.model)
    assert(not opt.cond)
    
    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, 'a+') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    if opt.verbose:
        print(sampling_conf)
        logger.info("first_stage_model")
        logger.info(model.first_stage_model)
        logger.info("UNet model")
        logger.info(model.model)

    logs = dict()

    for c in range(1000):
        run(model, imglogdir, label=c, eta=opt.eta,
            vanilla=opt.vanilla_sample,  n_samples=opt.n_samples, custom_steps=opt.custom_steps,
            batch_size=opt.batch_size, nplog=numpylogdir, dpm=opt.dpm, logs=logs)
    for k in logs.keys():
        print(logs[k].shape)

    torch.save(logs, os.path.join(logdir, "imagenet_sample10000_allst.pt"))

    logger.info("done.")
