#!/bin/bash

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="." python evaluation/franka_kitchen/Eval/core/hydra_launcher.py \
        hydra/launcher=local \
        hydra/output=local \
        env="kitchen_knob1_on-v3" \
        camera="right_cap2" \
        pixel_based=true \
        embedding=ViT-Base \
        num_demos=25 \
        env_kwargs.load_path=clip-recincr4 \
        bc_kwargs.lr=1e-3 \
        bc_kwargs.finetune=true \
        job_name=recincr4_rcap_knob181\
        seed=181 \
        proprio=9
    
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="." python evaluation/franka_kitchen/Eval/core/hydra_launcher.py \
    hydra/launcher=local \
    hydra/output=local \
    env="kitchen_ldoor_open-v3" \
    camera="right_cap2" \
    pixel_based=true \
    embedding=ViT-Base \
    num_demos=25 \
    env_kwargs.load_path=clip-recincr \
    bc_kwargs.lr=1e-3 \
    bc_kwargs.finetune=true \
    job_name=recincr_rcap_ldoor181 \
    seed=181 \
    proprio=9

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="." python evaluation/franka_kitchen/Eval/core/hydra_launcher.py \
    hydra/launcher=local \
    hydra/output=local \
    env="kitchen_light_on-v3" \
    camera="right_cap2" \
    pixel_based=true \
    embedding=ViT-Base \
    num_demos=25 \
    env_kwargs.load_path=clip-recincr4 \
    bc_kwargs.lr=1e-3 \
    bc_kwargs.finetune=true \
    job_name=recincr4_rcap_light181 \
    seed=181 \
    proprio=9

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="." python evaluation/franka_kitchen/Eval/core/hydra_launcher.py \
    hydra/launcher=local \
    hydra/output=local \
    env="kitchen_micro_open-v3" \
    camera="right_cap2" \
    pixel_based=true \
    embedding=ViT-Base \
    num_demos=25 \
    env_kwargs.load_path=clip-recincr4 \
    bc_kwargs.lr=1e-3 \
    bc_kwargs.finetune=true \
    job_name=recincr4_rcap_micro182 \
    seed=182 \
    proprio=9

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="." python evaluation/franka_kitchen/Eval/core/hydra_launcher.py \
    hydra/launcher=local \
    hydra/output=local \
    env="kitchen_sdoor_open-v3" \
    camera="right_cap2" \
    pixel_based=true \
    embedding=ViT-Base \
    num_demos=25 \
    env_kwargs.load_path=clip-recincr4 \
    bc_kwargs.lr=1e-3 \
    bc_kwargs.finetune=true \
    job_name=recincr4_rcap_sdoor1 \
    seed=1 \
    proprio=9