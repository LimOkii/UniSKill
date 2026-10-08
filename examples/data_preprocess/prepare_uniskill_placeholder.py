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
"""
Generate local placeholder parquet files for verl-agent environment training.

The environment supplies the actual task observations during rollout; these
records only provide dataloader shape, modality, and batch count.
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


def resolve_train_size(args) -> int:
    if args.target_train_steps is None:
        return args.train_data_size

    if args.train_batch_size is None:
        raise ValueError("--train_batch_size is required when --target_train_steps is set")
    if args.total_epochs is None:
        raise ValueError("--total_epochs is required when --target_train_steps is set")
    if args.train_batch_size <= 0:
        raise ValueError("--train_batch_size must be positive")
    if args.total_epochs <= 0:
        raise ValueError("--total_epochs must be positive")
    if args.target_train_steps <= 0:
        raise ValueError("--target_train_steps must be positive")
    if args.target_train_steps % args.total_epochs != 0:
        raise ValueError(
            "--target_train_steps must be divisible by --total_epochs for exact alignment: "
            f"{args.target_train_steps} % {args.total_epochs} != 0"
        )

    steps_per_epoch = args.target_train_steps // args.total_epochs
    return steps_per_epoch * args.train_batch_size


def make_placeholder_dataset(*, split: str, size: int, mode: str) -> datasets.Dataset:
    if size <= 0:
        raise ValueError(f"{split} size must be positive")

    if mode != "text":
        raise NotImplementedError("UniSkill placeholder generation currently supports --mode text only")

    records = []
    for idx in range(size):
        records.append(
            {
                "data_source": mode,
                "prompt": [
                    {
                        "role": "user",
                        "content": "",
                    }
                ],
                "ability": "agent",
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "placeholder": True,
                },
            }
        )
    return datasets.Dataset.from_list(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="text", choices=["text"])
    parser.add_argument("--local_dir", default="~/data/verl-agent/")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--train_data_size", default=256, type=int)
    parser.add_argument("--val_data_size", default=256, type=int)
    parser.add_argument("--target_train_steps", default=None, type=int)
    parser.add_argument("--train_batch_size", default=None, type=int)
    parser.add_argument("--total_epochs", default=None, type=int)
    args = parser.parse_args()

    train_size = resolve_train_size(args)
    val_size = args.val_data_size

    local_dir = os.path.join(os.path.expanduser(args.local_dir), args.mode)
    os.makedirs(local_dir, exist_ok=True)

    train_dataset = make_placeholder_dataset(split="train", size=train_size, mode=args.mode)
    test_dataset = make_placeholder_dataset(split="test", size=val_size, mode=args.mode)

    train_path = os.path.join(local_dir, "train.parquet")
    test_path = os.path.join(local_dir, "test.parquet")
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"Generated UniSkill placeholder data for mode={args.mode}")
    print(f"train rows: {train_size} -> {train_path}")
    print(f"val rows: {val_size} -> {test_path}")
    if args.train_batch_size:
        steps_per_epoch = train_size // args.train_batch_size
        dropped_rows = train_size % args.train_batch_size
        print(f"expected train dataloader steps per epoch: {steps_per_epoch}")
        print(f"rows dropped by drop_last=True: {dropped_rows}")
        if args.total_epochs:
            print(f"expected total training steps: {steps_per_epoch * args.total_epochs}")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
