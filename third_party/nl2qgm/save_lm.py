import logging
from pathlib import Path
import argparse
from transformers import AutoTokenizer, AutoConfig, AutoModel


def main(args):
    lm_name = args.lm_name
    save_path = args.save_path

    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    config = AutoConfig.from_pretrained(lm_name)
    model = AutoModel.from_pretrained(lm_name)

    tokenizer.save_pretrained(save_path)
    config.save_pretrained(save_path)
    model.save_pretrained(save_path)


if __name__ == "__main__":

    def _get_arguments():
        parser = argparse.ArgumentParser()

        parser.add_argument("--lm_name", type=Path, default="markussagen/xlm-roberta-longformer-base-4096")
        parser.add_argument("--save_path", type=Path, default="language_models/markussagen/xlm-roberta-longformer-base-4096")
        
        return parser.parse_args()

    args = _get_arguments()
    logging.basicConfig(level=logging.INFO)

    main(args)
