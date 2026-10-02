# HY-World 1.5 (WorldPlay) Training Guideline

We provide a detailed guideline, including dataset structure and training scripts, to train the base model on your own dataset. Specifically, now we only support the autoregressive training with action control and memory design based on HunyuanVideo 1.5.

## Prepare the Dataset
The json file structure is presented as below:

```
{
    "latent": latents,  # video latent after vae encoding
    "prompt_embeds": prompt_embeds,   # text prompt encoding latent
    "image_cond": cond_latents,     # image latent
    "vision_states": vision_states,     # sigLip feature
    "prompt_mask": attention_mask,     # text prompt mask
    "byt5_text_states": byt5_embeddings,    # byt5 text prompt
    "byt5_text_mask": byt5_masks   # byt5 text prompt mask
}
```
Specifically, we pre-process the video, txt, and image latent to accelerate the training process. The dataset processing is similar to the illustration in [Fastvideo](https://github.com/hao-ai-lab/FastVideo).

## Install Dependencies
Same as the inference guideline, we suggest to install the dependencies:

```bash
conda create --name worldplay python=3.10 -y
conda activate worldplay
pip install -r requirements.txt
```

## Running Training Scripts
We provide a training command on a single node as below:

```
bash scripts/training/hyvideo15/run_ar_hunyuan_action_mem.sh
```
Specifically, to better understand our training framework, we provide a detailed illustration about our training parameters.

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--json_path` | str | Yes | - | Path to training dataset json file |
| `--train_time_shift` | int | Yes | 3 | Timestep scheduler shift |
| `--window_frames` | int | Yes | 24 | The window length of video latents |
| `--model_path` | str | Yes | - | Path to pretrained model directory |
| `--output_dir` | str | Yes | - | Output directory for model checkpoints |
| `--max_train_steps` | int | Yes | - | Training steps |
| `--train_batch_size` | int | Yes | 1 | Data number of per rank |
| `--train_sp_batch_size` | int | Yes | 1 | Data number of Sequence Parallel Group |
| `--gradient_accumulation_steps` | int | Yes | 1 | The number of gradient accumulation steps |
| `--num_gpus` | int | Yes | - | Total GPU number |
| `--sp_size` | int | Yes | 4 | Sequence Parallel Size (GPU number for each sp group) |
| `--tp_size` | int | Yes | 1 | Tensor Parallel Size, set as 1 |
| `--hsdp_replicate_dim` | int | Yes | - | how many times that sharded setup is replicated across the cluster, set to $NODES |
| `--hsdp_shard_dim` | int | Yes | - | how many GPUs the model states (weights/gradients) are sharded across (set to $NUM_GPUS) |
| `--cls_name` | str | Yes | - | Class name for transformer |
| `--ar_action_load_from_dir` | str | No | - | Load path for autoregressive model with action control, mainly for memory training |
| `--checkpointing_steps` | int | Yes | 500 | Checkpoint saving interval step |
| `--resume_from_checkpoint` | str | No | - | Resume checkpoint path, e.g., /your_path/to/checkpoints/checkpoint-1000 |


We provide a detailed calculation of total batch size as follows:
```
total_batch_size = world_size / training_args.sp_size * 
                   training_args.gradient_accumulation_steps * 
                   training_args.train_sp_batch_size
```
where `world_size` represents the total number of GPUs. For example, we have one 8*H100 GPU node, training_args.sp_size = 4 with others set as default, the total batch size will be 2.

## Practices

+ We suggest pre-processing the dataset and outputing the json form to accelerate the dataset loading procedure.
+ Larger batch size is strongly suggested. Usually, the total batch size at least should set to 32.