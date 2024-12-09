#!/usr/bin/env python

import argparse
import json
import os

import _jsonnet
import attr
from ratsql.commands import preprocess, train, infer, eval
import torch
import datetime

@attr.s
class PreprocessConfig:
    config = attr.ib()
    config_args = attr.ib()


@attr.s
class TrainConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()


@attr.s
class InferConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    beam_size = attr.ib()
    output = attr.ib()
    step = attr.ib()
    mode = attr.ib()
    use_heuristic = attr.ib(default=False)
    limit = attr.ib(default=None)
    output_history = attr.ib(default=False)


@attr.s
class EvalConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    inferred = attr.ib()
    output = attr.ib()
    use_original_eval = attr.ib()


def setup_ddp(local_rank, world_size):
    if local_rank != -1:
        assert world_size > 1
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', 
            rank = local_rank,
            world_size = world_size,
            timeout=datetime.timedelta(minutes=60))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help="preprocess/train/eval", choices=["preprocess", "train", "train_LM", "eval", "debug"])
    parser.add_argument('exp_config_file', help="jsonnet file for experiments")
    parser.add_argument('--model_config_args', help="optional overrides for model config args")
    parser.add_argument('--logdir', help="optional override for logdir")
    parser.add_argument('--fp16', action='store_true', help='Flag for using 16-bit float precision')
    args = parser.parse_args()
    
    args.local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else -1
    args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1

    exp_config = json.loads(_jsonnet.evaluate_file(args.exp_config_file))
    model_config_file = exp_config["model_config"]
    if "model_config_args" in exp_config:
        model_config_args = exp_config["model_config_args"]
        if args.model_config_args is not None:
            model_config_args_json = _jsonnet.evaluate_snippet("", args.model_config_args)
            model_config_args.update(json.loads(model_config_args_json))
        model_config_args = json.dumps(model_config_args)
    elif args.model_config_args is not None:
        model_config_args = _jsonnet.evaluate_snippet("", args.model_config_args)
    else:
        model_config_args = None

    logdir = args.logdir or exp_config["logdir"]

    if args.mode == "preprocess":
        preprocess_config = PreprocessConfig(model_config_file, model_config_args)
        preprocess.main(preprocess_config)
    elif args.mode in ["train", "train_LM"]:
        print(f"local_rank:{args.local_rank} (world_size:{args.world_size})")
        setup_ddp(args.local_rank, args.world_size)
        train_config = TrainConfig(model_config_file,
                                   model_config_args, logdir)
        train_config.local_rank = args.local_rank
        train_config.world_size = args.world_size
        mode = "TEXT2SQL" if args.mode == 'train' else 'PLM'
        train.main(train_config, mode=mode)
    elif args.mode in ["eval", "debug"]:
        for step in exp_config["eval_steps"]:
            mode = 'infer' if args.mode == 'eval' else 'debug'
            infer_output_path = f"{exp_config['eval_output']}/{exp_config['eval_name']}-step_{step}-{mode}.jsonl"
            infer_config = InferConfig(
                model_config_file,
                model_config_args,
                logdir,
                exp_config["eval_section"],
                exp_config["eval_beam_size"],
                infer_output_path,
                step,
                mode=mode,
                use_heuristic=exp_config["eval_use_heuristic"]
            )
            infer.main(infer_config)

            if args.mode == 'eval':
                eval_output_path = f"{exp_config['eval_output']}/{exp_config['eval_name']}-step_{step}-eval.json"
                eval_config = EvalConfig(
                    model_config_file,
                    model_config_args,
                    logdir,
                    exp_config["eval_section"],
                    infer_output_path,
                    eval_output_path,
                    exp_config['use_original_eval'],
                )
                eval.main(eval_config)
    elif args.mode in ["infer"]:
        raise NotImplementedError("Not implemented yet")


if __name__ == "__main__":
    main()
