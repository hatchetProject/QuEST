# QuEST
The official repository for **QuEST: Low-bit Diffusion Model Quantization via Efficient Selective Finetuning** [[ArXiv]](https://arxiv.org/abs/2402.03666)

## Update Log 
We would first like to thank your interest in our work. The current repository is under reconstruction for combining different settings. Using the current code is fine but may encounter unexpected bugs. Apologies for this inconvenience!

**(2024.2.28)** Reorganized the code structures.

**TODO**: Program verification

## Features
QuEST achieves state-of-the-art performance on mutiple high-resolution image generation tasks, including unconditional image generation, class-conditional image generation and text-to-image generation. We also achieve superior performance on full 4-bit (W4A4) generation.   
![Display1](imgs/fig1.png)
On ImageNet:
![Display2](imgs/imagenet_figs.png)
On Stable Diffusion v1.4:
![Display3](imgs/sd_images.png)

## Get Started
### Presquites
Make sure you have conda installed first, then:
```
git clone https://github.com/hatchetProject/QuEST.git
cd QuEST
conda env create -f environment.yml
conda activate quest
```
### Usage
1. For Latent Diffusion and Stable Diffusion experiments, first download relvant checkpoints following the instructions in the [latent-diffusion](https://github.com/CompVis/latent-diffusion/tree/main) and [stable-diffusion](https://github.com/CompVis/stable-diffusion#weights) repos from CompVis. We currently use sd-v1-4.ckpt for Stable Diffusion.
2. The calibration data for LSUN-Bedrooms/Churches and Stable Diffusion (COCO) can be downloaded from the [Q-Diffusion](https://github.com/Xiuyu-Li/q-diffusion/tree/master) repository. We will upload the calibration data for ImageNet soon.
3. Use the following commands to reproduce the models. If setting act_bit to 4, please change the 'channel_wise' argument to True in aq_params in the code. Though termed 'channelwise', it is token-wise actually and does not effect computation efficiency.
```
# LSUN-Bedrooms (LDM-4)
python sample_diffusion_ldm_bedroom.py -r models/ldm/lsun_beds256/model.ckpt -n 100 --batch_size 20 -c 200 -e 1.0  --seed 40 --ptq  --weight_bit <4 or 8> --quant_mode qdiff --cali_st 20 --cali_batch_size 32 --cali_n 256 --quant_act --act_bit <4 or 8> --a_sym --a_min_max --running_stat --cali_data_path <cali_data_path> -l <output_path>
# LSUN-Churches (LDM-8)
python scripts/sample_diffusion_ldm_church.py -r models/ldm/lsun_churches256/model.ckpt -n 50000 --batch_size 10 -c 400 -e 0.0 --seed 40 --ptq --weight_bit <4 or 8> --quant_mode qdiff --cali_st 20 --cali_batch_size 32 --cali_n 256 --quant_act --act_bit <4 or 8> --cali_data_path <cali_data_path> -l <output_path>
# ImageNet
python sample_diffusion_ldm_imagenet.py -r models/ldm/cin256-v2/model.ckpt -n 50 --batch_size 50 -c 20 -e 1.0  --seed 40 --ptq  --weight_bit <4 or 8> --quant_mode qdiff --cali_st 20 --cali_batch_size 32 --cali_n 256 --quant_act --act_bit <4 or 8> --a_sym --a_min_max --running_stat --cond --cali_data_path <cali_data_path> -l <output_path>
# Stable Diffusion
python txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms --cond --ptq --weight_bit <4 or 8> --quant_mode qdiff --quant_act --act_bit <4 or 8> --cali_st 25 --cali_batch_size 8 --cali_n 128 --no_grad_ckpt --split --running_stat --sm_abit 16 --cali_data_path <cali_data_path> --outdir <output_path>
```



## Acknowledgement
This project is heavily based on [LDM](https://github.com/CompVis/latent-diffusion/tree/main) and [Q-Diffusion](https://github.com/Xiuyu-Li/q-diffusion/tree/master).

## Citation
If you find this work helpful, please consider citing our paper:
```
@misc{wang2024quest,
      title={QuEST: Low-bit Diffusion Model Quantization via Efficient Selective Finetuning}, 
      author={Haoxuan Wang and Yuzhang Shang and Zhihang Yuan and Junyi Wu and Yan Yan},
      year={2024},
      eprint={2402.03666},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
