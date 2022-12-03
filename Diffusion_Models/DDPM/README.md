#### DDPM & DDIM

CelebA에 대해서 실험 & FID score 평가

##### Train
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 main.py
```

##### Evaluation
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 evaluate_FID.py
```
fid score 잴때, n_sample_timesteps와 eta를 조정하자.

n_sample_timesteps : 샘플링 횟수,

eta : DDIM 논문 eq. (16)에서 deterministic을 결정하는 하이퍼파라미터.
