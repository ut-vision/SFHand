export PYTHONPATH="."
export PYOPENGL_PLATFORM=osmesa

# Recommendation: Training with 3 GPUs, batch size 256, learning rate 1e-4, 4 epochs.
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 main.py --config_file configs/config/clip_base.yml
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 main.py --config_file configs/config/clip_base_eval.yml --eval

