export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-}"   # Path to pretrained HunyuanVideo-1.5 model
NORMALIZED_MANIFEST="${NORMALIZED_MANIFEST:-}"   # Path to normalized origami manifest produced by origami_step_precompute.py
RAW_MANIFEST="${RAW_MANIFEST:-}"   # Path to raw origami manifest used for online feature encoding
ORIGAMI_FEATURE_SOURCE="${ORIGAMI_FEATURE_SOURCE:-precomputed}"
ORIGAMI_TARGET_FPS="${ORIGAMI_TARGET_FPS:-16}"
ORIGAMI_MEMORY_STEPS="${ORIGAMI_MEMORY_STEPS:-2}"
ORIGAMI_MEMORY_POLICY="${ORIGAMI_MEMORY_POLICY:-latest_k}"
ORIGAMI_MEMORY_BLEND="${ORIGAMI_MEMORY_BLEND:-0.35}"
ORIGAMI_MEMORY_MODE="${ORIGAMI_MEMORY_MODE:-mode2}"
LOAD_FROM_DIR="${LOAD_FROM_DIR:-}"   # Path to pretrained transformer directory
AR_ACTION_LOAD_FROM_DIR="${AR_ACTION_LOAD_FROM_DIR:-}"   # Optional: path to pretrained AR action model directory
OUTPUT_DIR="${OUTPUT_DIR:-}"   # Path to output directory
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
RESET="${RESET:-0}"
WANDB_KEY="${WANDB_KEY:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
TRACKER_PROJECT_NAME="${TRACKER_PROJECT_NAME:-origami_steps}"
LOG_STEPS="${LOG_STEPS:-10}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-500}"
TRAINING_STATE_CHECKPOINT_STEPS="${TRAINING_STATE_CHECKPOINT_STEPS:-0}"
WEIGHT_ONLY_CHECKPOINT_STEPS="${WEIGHT_ONLY_CHECKPOINT_STEPS:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
TRAIN_SP_BATCH_SIZE="${TRAIN_SP_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SAVE_LIMIT="${SAVE_LIMIT:-3}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200000}"
EPOCH="${EPOCH:-}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29612}"
SP_SIZE="${SP_SIZE:-}"

NUM_GPUS=$NPROC_PER_NODE

if [[ -z "$SP_SIZE" ]]; then
  # This is the stable baseline used by the original training launcher.
  SP_SIZE=4
fi

count_visible_devices() {
  local value="$1"
  python3 - "$value" <<'PY'
import sys
value = sys.argv[1]
items = [item.strip() for item in value.split(",") if item.strip()]
print(len(items))
PY
}

find_latest_training_state_checkpoint() {
  local output_dir="$1"
  find "$output_dir" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V | while read -r checkpoint_dir; do
    if [[ -f "$checkpoint_dir/training_state.pt" ]]; then
      printf "%s\n" "$checkpoint_dir"
    fi
  done | tail -n 1
}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${HIP_VISIBLE_DEVICES:-}" && -z "${ROCR_VISIBLE_DEVICES:-}" ]]; then
  DEFAULT_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC_PER_NODE - 1)))
  export CUDA_VISIBLE_DEVICES="$DEFAULT_VISIBLE_DEVICES"
  export HIP_VISIBLE_DEVICES="$DEFAULT_VISIBLE_DEVICES"
  export ROCR_VISIBLE_DEVICES="$DEFAULT_VISIBLE_DEVICES"
else
  ACTIVE_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$ACTIVE_VISIBLE_DEVICES}"
  export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$ACTIVE_VISIBLE_DEVICES}"
  export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-$ACTIVE_VISIBLE_DEVICES}"
fi

VISIBLE_DEVICE_COUNT=$(count_visible_devices "${HIP_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}}")
if (( VISIBLE_DEVICE_COUNT < NPROC_PER_NODE )); then
  echo "Visible device mismatch: NPROC_PER_NODE=$NPROC_PER_NODE but only $VISIBLE_DEVICE_COUNT visible device(s)." >&2
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" >&2
  echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-}" >&2
  echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-}" >&2
  echo "Unset the single-device env vars or set them to a full list such as 0,1,2,3,4,5,6,7 before launching." >&2
  exit 1
fi

if (( SP_SIZE < 1 || SP_SIZE > NPROC_PER_NODE )); then
  echo "SP_SIZE must be between 1 and NPROC_PER_NODE ($NPROC_PER_NODE); got $SP_SIZE." >&2
  exit 1
fi

if (((NNODES * NPROC_PER_NODE) % SP_SIZE != 0)); then
  echo "Total world size $((NNODES * NPROC_PER_NODE)) must be divisible by SP_SIZE=$SP_SIZE." >&2
  exit 1
fi

if (( SP_SIZE != 1 && SP_SIZE != 4 )); then
  echo "Origami training currently supports SP_SIZE=1 or SP_SIZE=4 only." >&2
  echo "Larger SP sizes can fail because the origami dataset allows short temporal sequences and the transformer chunks camera/action tensors by SP size." >&2
  exit 1
fi

if [[ -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH is required." >&2
  exit 1
fi

case "$ORIGAMI_FEATURE_SOURCE" in
  precomputed)
    if [[ -z "$NORMALIZED_MANIFEST" ]]; then
      echo "NORMALIZED_MANIFEST is required when ORIGAMI_FEATURE_SOURCE=precomputed." >&2
      exit 1
    fi
    ORIGAMI_MANIFEST="$NORMALIZED_MANIFEST"
    EFFECTIVE_DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
    EFFECTIVE_DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-}"
    ;;
  online)
    if [[ -z "$RAW_MANIFEST" ]]; then
      echo "RAW_MANIFEST is required when ORIGAMI_FEATURE_SOURCE=online." >&2
      exit 1
    fi
    ORIGAMI_MANIFEST="$RAW_MANIFEST"
    EFFECTIVE_DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
    EFFECTIVE_DATALOADER_PREFETCH_FACTOR=""
    ;;
  *)
    echo "Unsupported ORIGAMI_FEATURE_SOURCE=$ORIGAMI_FEATURE_SOURCE. Use precomputed or online." >&2
    exit 1
    ;;
esac

if [[ "$ORIGAMI_FEATURE_SOURCE" == "online" && "$EFFECTIVE_DATALOADER_NUM_WORKERS" != "0" ]]; then
  echo "ORIGAMI_FEATURE_SOURCE=online requires DATALOADER_NUM_WORKERS=0." >&2
  exit 1
fi

if [[ "$EFFECTIVE_DATALOADER_NUM_WORKERS" == "0" ]]; then
  EFFECTIVE_DATALOADER_PREFETCH_FACTOR=""
fi

if [[ -z "$LOAD_FROM_DIR" ]]; then
  echo "LOAD_FROM_DIR is required." >&2
  exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  echo "OUTPUT_DIR is required." >&2
  exit 1
fi

if [[ "$RESUME_FROM_CHECKPOINT" == "latest" ]]; then
  RESUME_FROM_CHECKPOINT="$(find_latest_training_state_checkpoint "$OUTPUT_DIR")"
  if [[ -z "$RESUME_FROM_CHECKPOINT" ]]; then
    echo "No checkpoint with training_state.pt found under $OUTPUT_DIR." >&2
    exit 1
  fi
fi

EXACT_RESUME_MODE=0
if [[ "$TRAINING_STATE_CHECKPOINT_STEPS" != "0" || -n "$RESUME_FROM_CHECKPOINT" ]]; then
  EXACT_RESUME_MODE=1
fi

if [[ "$EXACT_RESUME_MODE" == "1" && "$EFFECTIVE_DATALOADER_NUM_WORKERS" != "0" ]]; then
  echo "Exact origami resume requires DATALOADER_NUM_WORKERS=0 when using TRAINING_STATE_CHECKPOINT_STEPS or RESUME_FROM_CHECKPOINT." >&2
  exit 1
fi

if [[ -n "$RESUME_FROM_CHECKPOINT" && ! -f "$RESUME_FROM_CHECKPOINT/training_state.pt" ]]; then
  echo "RESUME_FROM_CHECKPOINT=$RESUME_FROM_CHECKPOINT does not contain training_state.pt." >&2
  exit 1
fi

training_args=(
  --data-path $ORIGAMI_MANIFEST
  --json_path $ORIGAMI_MANIFEST
  --dataset_type origami_steps
  --origami-feature-source $ORIGAMI_FEATURE_SOURCE
  --origami-target-fps $ORIGAMI_TARGET_FPS
  --origami-memory-steps $ORIGAMI_MEMORY_STEPS
  --origami-memory-policy $ORIGAMI_MEMORY_POLICY
  --origami-memory-blend $ORIGAMI_MEMORY_BLEND
  --origami-memory-mode $ORIGAMI_MEMORY_MODE
  --causal
  --action
  --i2v_rate 1.0
  --train_time_shift 5.0
  --window_frames 24
  --train_batch_size $TRAIN_BATCH_SIZE
  --train_sp_batch_size $TRAIN_SP_BATCH_SIZE
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS
  --num_latent_t 9
  --num_height 480
  --num_width 832
  --num_frames 77
  --enable_gradient_checkpointing_type "full"
  --seed 3208
  --log-steps $LOG_STEPS
  --weighting_scheme "logit_normal"
  --logit_mean 0.0
  --logit_std 1.0
  --output_dir $OUTPUT_DIR
)

if [[ -n "$EPOCH" ]]; then
  training_args+=(--num-train-epochs "$EPOCH")
else
  training_args+=(--max-train-steps "$MAX_TRAIN_STEPS")
fi

if [[ -n "$WANDB_KEY" ]]; then
  training_args+=(--wandb_key "$WANDB_KEY")
fi

if [[ -n "$WANDB_ENTITY" ]]; then
  training_args+=(--wandb_entity "$WANDB_ENTITY")
fi

if [[ -n "$TRACKER_PROJECT_NAME" ]]; then
  training_args+=(--tracker_project_name "$TRACKER_PROJECT_NAME")
fi

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  training_args+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

if [[ "$RESET" == "1" ]]; then
  training_args+=(--resume-reset-dataset true)
fi

parallel_args=(
  --num_gpus $((NNODES * NPROC_PER_NODE))
  --sp_size $SP_SIZE
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim $((NNODES * NPROC_PER_NODE))
)

model_args=(
  --cls_name "HunyuanTransformer3DARActionModel"
  --load_from_dir $LOAD_FROM_DIR
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
)

if [[ -n "$AR_ACTION_LOAD_FROM_DIR" ]]; then
  model_args+=(--ar_action_load_from_dir "$AR_ACTION_LOAD_FROM_DIR")
fi

dataset_args=(
  --dataloader_num_workers $EFFECTIVE_DATALOADER_NUM_WORKERS
)
if [[ -n "$EFFECTIVE_DATALOADER_PREFETCH_FACTOR" ]]; then
  dataset_args+=(--dataloader-prefetch-factor "$EFFECTIVE_DATALOADER_PREFETCH_FACTOR")
fi

optimizer_args=(
  --learning_rate 3e-5
  --lr_warmup_steps 100
  --lr_scheduler cosine
  --mixed_precision "bf16"
  --checkpointing-steps $CHECKPOINT_STEPS
  --training-state-checkpointing-steps $TRAINING_STATE_CHECKPOINT_STEPS
  --weight-only-checkpointing-steps $WEIGHT_ONLY_CHECKPOINT_STEPS
  --weight_decay 1e-4
  --max_grad_norm 1.0
)


miscellaneous_args=(
  --inference_mode False
  --checkpoints-total-limit $SAVE_LIMIT
  --training_cfg_rate 0.1
  --multi_phased_distill_schedule "4000-1"
  --not_apply_cfg_solver
  --dit_precision "fp32"
  --num_euler_timesteps 50
  --ema_start_step 0
)

echo "Distributed config: NNODES=$NNODES NPROC_PER_NODE=$NPROC_PER_NODE NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "Parallel topology: SP_SIZE=$SP_SIZE HSDP_REPLICATE_DIM=1 HSDP_SHARD_DIM=$((NNODES * NPROC_PER_NODE))"
echo "Origami feature source: $ORIGAMI_FEATURE_SOURCE"
echo "Origami manifest: $ORIGAMI_MANIFEST"
echo "Origami vision memory: steps=$ORIGAMI_MEMORY_STEPS policy=$ORIGAMI_MEMORY_POLICY blend=$ORIGAMI_MEMORY_BLEND mode=$ORIGAMI_MEMORY_MODE"
echo "Origami dataloader: workers=$EFFECTIVE_DATALOADER_NUM_WORKERS prefetch_factor=${EFFECTIVE_DATALOADER_PREFETCH_FACTOR:-unset}"
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  echo "Resume checkpoint: $RESUME_FROM_CHECKPOINT"
fi
if [[ -n "$EPOCH" ]]; then
  echo "Training schedule: epochs=$EPOCH checkpoint_steps=$CHECKPOINT_STEPS save_limit=$SAVE_LIMIT"
else
  echo "Training schedule: max_train_steps=$MAX_TRAIN_STEPS checkpoint_steps=$CHECKPOINT_STEPS save_limit=$SAVE_LIMIT"
fi
echo "Visible devices: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-}"

torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$NPROC_PER_NODE \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        trainer/training/ar_hunyuan_w_mem_training_pipeline.py \
        "${parallel_args[@]}" \
        "${model_args[@]}" \
        "${dataset_args[@]}" \
        "${training_args[@]}" \
        "${optimizer_args[@]}" \
        "${miscellaneous_args[@]}"
