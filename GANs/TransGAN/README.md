# TransGAN Pytorch
Unofficial Pytorch implementation of TransGAN. CUDAȯ�� Distributed DataParallel������ �����Ͽ���. �����ʹ� CelebA_HQ ���.

CUDA=11.3 <br/>
pytorch 1.11.1+cu113

### Training
```bash
torchrun --nproc_per_node <number of GPUs> train.py
```
