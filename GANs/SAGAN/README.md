# SAGAN Pytorch
Unofficial Pytorch implementation of SAGAN for ImageNet

CUDA=11.3 <br/>
pytorch 1.11.1+cu113

### Training
```bash
torchrun --nproc_per_node <number of GPUs> main --use_DDP
```
