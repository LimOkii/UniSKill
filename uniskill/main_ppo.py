# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""UniSkill PPO training entry point."""

import os
from functools import partial

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from uniskill.trainer.ray_trainer import UniSkillRayPPOTrainer
from uniskill.config import UniSkillSettings


@hydra.main(config_path="../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        # Forward WebShop worker output to the launcher's stderr stream.
        if "webshop" in str(config.env.env_name).lower():
            import sys

            sys.stdout = sys.stderr

        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        uniskill_settings = UniSkillSettings.from_config(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        envs, val_envs = make_uniskill_envs(config)

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import CriticWorker
            from uniskill.workers import UniSkillActorRolloutRefWorker

            actor_rollout_cls = UniSkillActorRolloutRefWorker
            ActorRolloutRefWorker = UniSkillActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError("UniSkill supports FSDP/FSDP2 only")

        from uniskill.trainer.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError(
                    f"Unsupported reward model strategy: {config.reward_model.strategy}"
                )
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == "episode":
            from agent_system.reward_manager import EpisodeRewardManager

            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError(f"Unsupported reward manager: {reward_manager_name}")

        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        assert config.actor_rollout_ref.rollout.n == 1, (
            "actor_rollout_ref.rollout.n must be 1; use env.rollout.n "
            "to set the number of trajectories per task"
        )

        from uniskill.multi_turn_rollout import TrajectoryCollector

        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = UniSkillRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
            uniskill_settings=uniskill_settings,
        )
        trainer.init_workers()
        trainer.fit()


def make_uniskill_envs(config):
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    if "webshop" in config.env.env_name.lower():
        return _make_webshop_envs(config)
    if "alfworld" not in config.env.env_name.lower():
        raise ValueError(f"Unsupported UniSkill environment: {config.env.env_name}")

    from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
    from uniskill.environments.env_manager import AlfWorldEnvironmentManager
    from uniskill.skillbank import AlfWorldSkillBank

    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if config.env.env_name in ["alfworld/AlfredThorEnv", "alfworld/AlfredTWEnv"]:
        import agent_system.environments.env_manager as native_env_manager

        alf_config_path = os.path.join(
            os.path.dirname(native_env_manager.__file__),
            "env_package/alfworld/configs/config_tw.yaml",
        )
    else:
        raise ValueError(f"Unsupported environment: {config.env.env_name}")

    env_kwargs = {
        "eval_dataset": config.env.alfworld.eval_dataset,
    }
    envs = build_alfworld_envs(
        alf_config_path,
        config.env.seed,
        config.data.train_batch_size,
        group_n,
        is_train=True,
        env_kwargs=env_kwargs,
        resources_per_worker=resources_per_worker,
    )
    val_envs = build_alfworld_envs(
        alf_config_path,
        config.env.seed + 1000,
        config.data.val_batch_size,
        1,
        is_train=False,
        env_kwargs=env_kwargs,
        resources_per_worker=resources_per_worker,
    )

    projection_f = partial(alfworld_projection)
    uniskill_config = config.get("uniskill", {})
    retrieval_config = uniskill_config.get("retrieval", {})
    shared_skillbank = AlfWorldSkillBank(
        skill_dir=uniskill_config.get("skill_dir"),
        retrieval_start_step=int(retrieval_config.get("start_step", 2)),
        retrieval_method=retrieval_config.get("method", "lexical"),
        embedding_model_path=retrieval_config.get("embedding_model_path"),
        embedding_dir=uniskill_config.get("embedding_dir"),
    )
    return (
        AlfWorldEnvironmentManager(envs, projection_f, config, skillbank=shared_skillbank),
        AlfWorldEnvironmentManager(val_envs, projection_f, config, skillbank=shared_skillbank),
    )


def _make_webshop_envs(config):
    import time

    from agent_system.environments.env_package.webshop import (
        build_webshop_envs,
        webshop_projection,
    )
    from uniskill.environments.webshop_native_env_manager import (
        NativeActorWebshopEnvironmentManager as WebshopEnvironmentManager,
    )
    from uniskill.skillbank import WebshopSkillBank
    from uniskill.webshop_state import install_webshop_state_worker

    install_webshop_state_worker()

    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(
        config.env.resources_per_worker, resolve=True
    )
    webshop_config = config.env.webshop
    default_data_dir = os.path.join(
        os.path.dirname(__file__),
        "../agent_system/environments/env_package/webshop/webshop/data",
    )
    data_dir = os.path.abspath(
        str(
            webshop_config.get("data_dir")
            or os.environ.get("WEBSHOP_DATA_DIR")
            or default_data_dir
        )
    )
    suffix = "_1000" if bool(webshop_config.use_small) else ""
    file_path = os.path.join(data_dir, f"items_shuffle{suffix}.json")
    attr_path = os.path.join(data_dir, f"items_ins_v2{suffix}.json")
    # The vendored WebShop engine loads this file from its own data directory
    # even when synthetic goals are selected.
    human_attr_path = os.path.join(default_data_dir, "items_human_ins.json")
    missing = [
        path
        for path in (file_path, attr_path, human_attr_path)
        if not os.path.isfile(path)
    ]
    if missing:
        raise FileNotFoundError(
            "WebShop data files are missing. Set WEBSHOP_DATA_DIR or "
            "+env.webshop.data_dir to the directory containing them: "
            + ", ".join(missing)
        )

    env_kwargs = {
        "observation_mode": "text",
        "num_products": None,
        "human_goals": bool(webshop_config.human_goals),
        "file_path": file_path,
        "attr_path": attr_path,
    }
    envs = build_webshop_envs(
        seed=config.env.seed,
        env_num=config.data.train_batch_size,
        group_n=group_n,
        is_train=True,
        env_kwargs=env_kwargs,
        resources_per_worker=resources_per_worker,
    )
    val_envs = build_webshop_envs(
        seed=config.env.seed + 1000,
        env_num=config.data.val_batch_size,
        group_n=1,
        is_train=False,
        env_kwargs=env_kwargs,
        resources_per_worker=resources_per_worker,
    )

    uniskill_config = config.get("uniskill", {})
    retrieval_config = uniskill_config.get("retrieval", {})
    webshop_uniskill_config = uniskill_config.get("webshop", {})
    shared_skillbank = WebshopSkillBank(
        skill_dir=uniskill_config.get("skill_dir"),
        retrieval_start_step=int(retrieval_config.get("start_step", 2)),
        retrieval_method=retrieval_config.get("method", "lexical"),
        embedding_model_path=retrieval_config.get("embedding_model_path"),
        embedding_dir=uniskill_config.get("embedding_dir"),
        embedding_min_score=float(
            retrieval_config.get("embedding_min_score", 0.55)
        ),
        seed_initial_skill=bool(
            webshop_uniskill_config.get("seed_initial_skill", False)
        ),
    )
    projection_f = partial(webshop_projection)
    if (
        config.trainer.resume_mode == "disable"
        and shared_skillbank._read_skills("skills.json", "webshop")
    ):
        raise ValueError("WebShop fresh runs require an empty, independent skill_dir")
    time.sleep(
        (config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1
    )
    return (
        WebshopEnvironmentManager(envs, projection_f, config, skillbank=shared_skillbank),
        WebshopEnvironmentManager(val_envs, projection_f, config, skillbank=shared_skillbank),
    )


if __name__ == "__main__":
    main()
