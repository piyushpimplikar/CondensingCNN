#!/usr/bin/env bash
cd /home/dlvc_team006/PDE_GlobalLayer_tensorflow
/home/dlvc_team006/miniconda3/envs/pde_venv/bin/python3 train.py --data "./data" --model Resnet --model_name Resnet --dataset CIFAR-10 -m 5 --n1 16 --n2 32 --n3 64 --n4 64 -b 128  -o  -e 300 --hp_optimization
/home/dlvc_team006/miniconda3/envs/pde_venv/bin/python3 train.py --data "./data"  --model Resnet-Global --model_name Resnet-Global --dataset CIFAR-10 -m 1 --K 5 --n1 16 --n2 32 --n3 64 --n4 64 -b 32 -wd 6e-5 --non_linear -e 300 --hp_optimization
/home/dlvc_team006/miniconda3/envs/pde_venv/bin/python3 train.py --data "./data" --no_diffusion --model Resnet-Global --model_name Resnet-Global_ND --dataset CIFAR-10 -m 1 --K 5 --n1 16 --n2 32 --n3 64 --n4 64 -b 32 -wd 6e-5 --non_linear -e 300 --hp_optimization
/home/dlvc_team006/miniconda3/envs/pde_venv/bin/python3 train.py --data "./data" --no_advection --model Resnet-Global --model_name Resnet-Global_NA --dataset CIFAR-10 -m 1 --K 5 --n1 16 --n2 32 --n3 64 --n4 64 -b 32 -wd 6e-5 --non_linear -e 300 --hp_optimization

/home/dlvc_team006/miniconda3/envs/pde_venv/bin/python3 train_inpainting.py --global_layer --hp_optimization
/home/dlvc_team006/miniconda3/envs/pde_venv/bin/python3 train_inpainting.py --hp_optimization

/home/dlvc_team006/miniconda3/envs/pde_venv/bin/python3 train.py --data "./data" --model Resnet --model_name Resnet --dataset CIFAR-100 -m 5 --n1 16 --n2 32 --n3 64 --n4 64 -b 128  -o   -e 300 --hp_optimization
/home/dlvc_team006/miniconda3/envs/pde_venv/bin/python3 train.py --data "./data"  --model Resnet-Global --model_name Resnet-Global --dataset CIFAR-100 -m 1 --K 5 --n1 16 --n2 32 --n3 64 --n4 64 -b 32 -wd 6e-5 --non_linear -e 300 --hp_optimization
/home/dlvc_team006/miniconda3/envs/pde_venv/bin/python3 train.py --data "./data" --no_diffusion --model Resnet-Global --model_name Resnet-Global_ND --dataset CIFAR-100 -m 1 --K 5 --n1 16 --n2 32 --n3 64 --n4 64 -b 32 -wd 6e-5 --non_linear -e 300 --hp_optimization
/home/dlvc_team006/miniconda3/envs/pde_venv/bin/python3 train.py --data "./data" --no_advection --model Resnet-Global --model_name Resnet-Global_NA --dataset CIFAR-100 -m 1 --K 5 --n1 16 --n2 32 --n3 64 --n4 64 -b 32 -wd 6e-5 --non_linear -e 300 --hp_optimization
