# pipeline-grpo 

## Setup

```sh
conda create -n grpo python=3.10 -y
conda activate grpo
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Usage

```sh
accelerate launch --config_file accelerate-config.yaml main.py
```

## Speeding up training with VLLM

```sh
pip install vllm

# There is a bug in VLLM that requires this. https://github.com/vllm-project/vllm/issues/10300
export LD_LIBRARY_PATH=/home/baris/miniconda3/envs/grpo/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

accelerate launch --config_file accelerate-config.yaml main.py --use-vllm
```