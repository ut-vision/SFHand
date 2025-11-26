export PYTHONPATH="/home/lruicong/embodi/StreamPOS"
export PYOPENGL_PLATFORM=osmesa

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 main.py --config_file configs/config/clip_base.yml
# CUDA_VISIBLE_DEVICES=0 torchrun --master_port=29512 --nproc_per_node=1 main.py --config_file configs/config/clip_base_eval.yml --eval

