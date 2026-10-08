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

if [[ ! -d uniskill || ! -d verl || ! -d agent_system ]]; then
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
: "${JAVA_HOME:?Set JAVA_HOME to the JVM installation used by WebShop}"
RUN_NAME="${RUN_NAME:-webshop_main}"
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "RUN_NAME must contain only letters, digits, '.', '_' or '-'." >&2
    exit 2
fi
DATA_DIR="${DATA_DIR:-uniskill/artifacts/data/${RUN_NAME}}"

for run_dir in "uniskill/artifacts/records/webshop/${RUN_NAME}" \
               "uniskill/artifacts/runtime_skills/webshop/${RUN_NAME}" \
               "uniskill/artifacts/checkpoints/${RUN_NAME}"; do
    if [[ -e "$run_dir" ]]; then
        echo "Run output already exists: $run_dir. Choose a fresh RUN_NAME." >&2
        exit 2
    fi
done

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export TOKENIZERS_PARALLELISM=false
export JVM_PATH="${JVM_PATH:-$JAVA_HOME/lib/server/libjvm.so}"
export LD_LIBRARY_PATH=$JAVA_HOME/lib/server:${LD_LIBRARY_PATH:-}
export JAVA_TOOL_OPTIONS=-XX:ActiveProcessorCount=1
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR=uniskill/artifacts/wandb
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAY_num_server_call_thread=1

mkdir -p "$WANDB_DIR"

python3 -m examples.data_preprocess.prepare_uniskill_placeholder \
    --mode text \
    --local_dir "$DATA_DIR" \
    --train_data_size 16 \
    --val_data_size 128

python3 -m uniskill.main_ppo \
    hydra.run.dir=uniskill/artifacts/hydra/${RUN_NAME} \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    +algorithm.credit_assignment=true \
    +algorithm.step_gamma=0.95 \
    algorithm.filter_groups.enable=false \
    algorithm.use_kl_in_reward=false \
    data.train_files=$DATA_DIR/text/train.parquet \
    data.val_files=$DATA_DIR/text/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=128 \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
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
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.history_length=2 \
    env.rollout.n=8 \
    env.resources_per_worker.num_cpus=0.1 \
    env.webshop.use_small=true \
    env.webshop.human_goals=false \
    +uniskill.output_dir=uniskill/artifacts/records/webshop/${RUN_NAME} \
    +uniskill.skill_dir=uniskill/artifacts/runtime_skills/webshop/${RUN_NAME} \
    +uniskill.embedding_dir=uniskill/artifacts/skill_embedding/webshop/${RUN_NAME} \
    +uniskill.seed=0 \
    +uniskill.webshop.strict_actor_credit_start_step=31 \
    +uniskill.webshop.selected_options_state=true \
    +uniskill.webshop.seed_initial_skill=false \
    +uniskill.webshop.native_prompt_char_limit=50000 \
    +uniskill.webshop.warmup_samples=8 \
    +uniskill.r_align_warmup_steps=30 \
    +uniskill.proposal_warmup_lambda=0.1 \
    +uniskill.lambda_proposal=0.5 \
    +uniskill.proposal_full_response_loss=true \
    +uniskill.proposal_composite_reward=true \
    +uniskill.r_align_reward_clip=0.05 \
    +uniskill.proposal_entropy_coeff=0.0 \
    +uniskill.proposal_kl_loss_coef=0.02 \
    +uniskill.operation_support_min_probability=0.1 \
    +uniskill.operation_support_coeff=0.01 \
    +uniskill.invalid_penalty=-0.05 \
    +uniskill.min_anchor_coverage=0.8 \
    +uniskill.proposal_max_prompt_length=16384 \
    +uniskill.proposal_max_response_length=1024 \
    +uniskill.best_checkpoint_min_success_rate=0.80 \
    +uniskill.retrieval.method=embedding \
    +uniskill.retrieval.start_step=31 \
    +uniskill.retrieval.embedding_min_score=0.55 \
    +uniskill.retrieval.embedding_model_path="$EMBEDDING_MODEL_PATH" \
    +uniskill.critic.config_path=uniskill/critic/config.yaml \
    +uniskill.critic.concurrency=4 \
    +uniskill.critic.rate_limit_backoff_seconds=60.0 \
    +uniskill.debug.enabled=false \
    +uniskill.debug.samples_per_batch=0 \
    +uniskill.debug.max_action_steps=0 \
    +uniskill.debug.show_token_ids=false \
    +ray_init.log_to_driver=true \
    +ray_init.runtime_env.worker_process_setup_hook=uniskill.native_webshop_runtime.setup_worker_logging \
    trainer.ray_wait_register_center_timeout=900 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=uniskill_webshop \
    trainer.experiment_name=qwen2.5_7b_${RUN_NAME} \
    trainer.default_local_dir=uniskill/artifacts/checkpoints/${RUN_NAME} \
    trainer.rollout_data_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.resume_mode=disable \
    trainer.save_freq=-1 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.test_freq=5 \
    trainer.total_epochs=250 \
    trainer.val_before_train=true \
    "$@"
