#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

CONFIG_FILE="${UNISKILL_CONFIG_FILE:-$SCRIPT_DIR/config.yaml}"
if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="$(pwd)/$CONFIG_FILE"
fi
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Missing training configuration: $CONFIG_FILE" >&2
    exit 2
fi
CONFIG_EXPORTS="$(python3 "$SCRIPT_DIR/load_config.py" "$CONFIG_FILE")" || exit 2
eval "$CONFIG_EXPORTS"

if [[ ! -d uniskill || ! -d verl ]]; then
    echo "Run this script from the verl-agent repository root." >&2
    exit 2
fi

ENGINE=vllm
if [[ $# -gt 0 && "$1" != *=* ]]; then
    ENGINE=$1
    shift
fi

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH in scripts/config.yaml or the environment}"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:?Set EMBEDDING_MODEL_PATH in scripts/config.yaml or the environment}"
: "${UNISKILL_CRITIC_API_URL:?Set UNISKILL_CRITIC_API_URL}"
: "${UNISKILL_CRITIC_API_KEY:?Set UNISKILL_CRITIC_API_KEY}"

RUN_NAME="${RUN_NAME:-alfworld_main}"
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "RUN_NAME must contain only letters, digits, '.', '_' or '-'." >&2
    exit 2
fi
EXPERIMENT_NAME="uniskill_qwen2.5_7b_${RUN_NAME}"
DATA_DIR="${DATA_DIR:-uniskill/artifacts/data/${RUN_NAME}}"

export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR=uniskill/artifacts/wandb
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

train_data_size=16
val_data_size=128
group_size=8
num_cpus_per_env_worker=0.2

uniskill_output_dir=uniskill/artifacts/records/alfworld/${RUN_NAME}
uniskill_skill_dir=uniskill/artifacts/runtime_skills/alfworld/${RUN_NAME}
uniskill_embedding_dir=uniskill/artifacts/skill_embedding/alfworld/${RUN_NAME}
uniskill_checkpoint_dir=uniskill/artifacts/checkpoints/${RUN_NAME}

for run_dir in "$uniskill_output_dir" "$uniskill_skill_dir" "$uniskill_checkpoint_dir"; do
    if [[ -e "$run_dir" ]]; then
        echo "Run output already exists: $run_dir. Choose a fresh RUN_NAME." >&2
        exit 2
    fi
done

lambda_proposal=0.5
proposal_entropy_coeff=0.0
proposal_kl_loss_coef=0.02
proposal_full_response_loss=true
proposal_assistant_prefill='<action>'
proposal_composite_reward=true
r_align_reward_clip=0.05
operation_support_min_probability=0.10
operation_support_coeff=0.01
invalid_penalty=-0.05
r_align_warmup_steps=3
min_anchor_coverage=0.8
proposal_max_response_length=1024
best_checkpoint_min_success_rate=0.85
critic_config_path=uniskill/critic/config.yaml
critic_concurrency=4
critic_rate_limit_backoff_seconds=60.0

debug_enabled=false
debug_samples=0
debug_action_steps=0
debug_token_ids=false

python3 -m examples.data_preprocess.prepare_uniskill_placeholder \
    --mode 'text' \
    --local_dir "$DATA_DIR" \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m uniskill.main_ppo \
    hydra.run.dir=uniskill/artifacts/hydra/${RUN_NAME} \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    +algorithm.credit_assignment=true \
    +algorithm.step_gamma=0.95 \
    algorithm.filter_groups.enable=false \
    algorithm.use_kl_in_reward=false \
    data.train_files="$DATA_DIR/text/train.parquet" \
    data.val_files="$DATA_DIR/text/test.parquet" \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=16384 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    data.return_raw_chat=true \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_fused_kernels=false \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.use_invalid_action_penalty=true \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    +actor_rollout_ref.actor.replace_strict_format_invalid_score=true \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.multi_turn.enable=false \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=17408 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.free_cache_engine=false \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.history_length=2 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    +uniskill.output_dir=$uniskill_output_dir \
    +uniskill.skill_dir=$uniskill_skill_dir \
    +uniskill.embedding_dir=$uniskill_embedding_dir \
    +uniskill.seed=0 \
    +uniskill.lambda_proposal=$lambda_proposal \
    +uniskill.proposal_entropy_coeff=$proposal_entropy_coeff \
    +uniskill.proposal_kl_loss_coef=$proposal_kl_loss_coef \
    +uniskill.proposal_full_response_loss=$proposal_full_response_loss \
    "+uniskill.proposal_assistant_prefill='$proposal_assistant_prefill'" \
    +uniskill.proposal_composite_reward=$proposal_composite_reward \
    +uniskill.r_align_reward_clip=$r_align_reward_clip \
    +uniskill.operation_support_min_probability=$operation_support_min_probability \
    +uniskill.operation_support_coeff=$operation_support_coeff \
    +uniskill.invalid_penalty=$invalid_penalty \
    +uniskill.r_align_warmup_steps=$r_align_warmup_steps \
    +uniskill.min_anchor_coverage=$min_anchor_coverage \
    +uniskill.proposal_max_prompt_length=16384 \
    +uniskill.proposal_max_response_length=$proposal_max_response_length \
    +uniskill.best_checkpoint_min_success_rate=$best_checkpoint_min_success_rate \
    +uniskill.debug.enabled=$debug_enabled \
    +uniskill.debug.samples_per_batch=$debug_samples \
    +uniskill.debug.max_action_steps=$debug_action_steps \
    +uniskill.debug.show_token_ids=$debug_token_ids \
    +uniskill.retrieval.method=embedding \
    +uniskill.retrieval.start_step=2 \
    +uniskill.retrieval.embedding_model_path="$EMBEDDING_MODEL_PATH" \
    +uniskill.critic.config_path=$critic_config_path \
    +uniskill.critic.max_attempts=5 \
    +uniskill.critic.parse_retry_attempts=1 \
    +uniskill.critic.concurrency=$critic_concurrency \
    +uniskill.critic.initial_backoff_seconds=1.0 \
    +uniskill.critic.max_backoff_seconds=30.0 \
    +uniskill.critic.jitter_seconds=0.5 \
    +uniskill.critic.rate_limit_backoff_seconds=$critic_rate_limit_backoff_seconds \
    trainer.ray_wait_register_center_timeout=900 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='uniskill_alfworld' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$uniskill_checkpoint_dir \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.resume_mode=disable \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=250 \
    trainer.val_before_train=False \
    "$@"
