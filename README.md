# QuEST
The official repository for **QuEST: Low-bit Diffusion Model Quantization via Efficient Selective Finetuning** [[ArXiv]](https://arxiv.org/abs/2402.03666)

QuEST achieves state-of-the-art performance on mutiple high-resolution image generation tasks, including unconditional image generation, class-conditional image generation and text-to-image generation. We also achieve superior performance on full 4-bit (W4A4) generation.   
![Display1](imgs/fig1.png)
On ImageNet:
![Display2](imgs/imagenet_figs.png)
On Stable Diffusion v1.4:
![Display3](imgs/sd_images.png)

## Get Started
### Installation
Make sure you have conda installed first, then:
```
git clone
cd QuEST
conda env create -f environment.yml
conda activate quest
```





## Acknowledge
This project is heavily based on [LDM](https://github.com/CompVis/latent-diffusion/tree/main) and [Q-Diffusion](https://github.com/Xiuyu-Li/q-diffusion/tree/master).
