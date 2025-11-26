#!/bin/bash

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="." python evaluation/franka_kitchen/Eval/core/hydra_launcher.py \
        hydra/launcher=local \
        hydra/output=local \
        env="kitchen_knob1_on-v3" \
        camera="left_cap2" \
        pixel_based=true \
        embedding=ViT-Base \
        num_demos=25 \
        env_kwargs.load_path=clip-recincr \
        bc_kwargs.lr=1.6e-3 \
        bc_kwargs.finetune=true \
        job_name=recincr_lcap_knob1\
        seed=1 \
        proprio=9
    
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="." python evaluation/franka_kitchen/Eval/core/hydra_launcher.py \
    hydra/launcher=local \
    hydra/output=local \
    env="kitchen_ldoor_open-v3" \
    camera="left_cap2" \
    pixel_based=true \
    embedding=ViT-Base \
    num_demos=25 \
    env_kwargs.load_path=clip-recincr \
    bc_kwargs.lr=1e-3 \
    bc_kwargs.finetune=true \
    job_name=recincr_lcap_ldoor181 \
    seed=181 \
    proprio=9

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="." python evaluation/franka_kitchen/Eval/core/hydra_launcher.py \
    hydra/launcher=local \
    hydra/output=local \
    env="kitchen_light_on-v3" \
    camera="left_cap2" \
    pixel_based=true \
    embedding=ViT-Base \
    num_demos=25 \
    env_kwargs.load_path=clip-recincr4 \
    bc_kwargs.lr=1e-3 \
    bc_kwargs.finetune=true \
    job_name=recincr4_lcap_light1 \
    seed=1 \
    proprio=9

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="." python evaluation/franka_kitchen/Eval/core/hydra_launcher.py \
    hydra/launcher=local \
    hydra/output=local \
    env="kitchen_micro_open-v3" \
    camera="left_cap2" \
    pixel_based=true \
    embedding=ViT-Base \
    num_demos=25 \
    env_kwargs.load_path=clip-drop \
    bc_kwargs.lr=1e-3 \
    bc_kwargs.finetune=true \
    job_name=drop_lcap_micro125 \
    seed=125 \
    proprio=9

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="." python evaluation/franka_kitchen/Eval/core/hydra_launcher.py \
    hydra/launcher=local \
    hydra/output=local \
    env="kitchen_sdoor_open-v3" \
    camera="left_cap2" \
    pixel_based=true \
    embedding=ViT-Base \
    num_demos=25 \
    env_kwargs.load_path=clip-recincr4 \
    bc_kwargs.lr=1e-3 \
    bc_kwargs.finetune=true \
    job_name=recincr4_lcap_sdoor1 \
    seed=1 \
    proprio=9