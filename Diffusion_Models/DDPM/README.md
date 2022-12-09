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

CelebA-HQ기준
Official code : 53.5456
본 코드 eta1, n_timestep 50 : 54.9308
본 코드 eta1, n_timestep 50 : 40.120
본 코드 eta1, n_timestep 50 : 33.9795
