import functools
from typing import *

import hkkang_utils.pattern as pattern_utils
import redis
import requests
import torch
from auto_gptq import AutoGPTQForCausalLM
from text_generation import Client
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# Text generation API Address
TEXT_GENERATION_API_IP = "141.223.199.149"
TEXT_GENERATION_API_PORT = "8080"
TEXT_GENERATION_API_ADDR = f"http://{TEXT_GENERATION_API_IP}:{TEXT_GENERATION_API_PORT}"

# Initialize redis
rd = redis.StrictRedis(host="141.223.199.149", port=6379, db=2)


class LLM(metaclass=pattern_utils.SingletonABCMetaWithArgs):
    def __init__(
        self,
        model_name: str = "text_generation/fangloveskari/ORCA_LLaMA_70B_QLoRA",
        precision_num: Optional[int] = None,
        max_new_token: int = 200,
        use_cache: bool = True,
        override_cache: bool = False,
        text_api_addr: str = TEXT_GENERATION_API_ADDR,
    ):
        super().__init__()
        self.model_name = model_name
        self.precision_num = 4 if precision_num is None else precision_num
        self.max_new_token = max_new_token
        self.use_cache = use_cache
        self.override_cache = override_cache
        self.text_api_addr = text_api_addr
        self.__post_init__()

    def __post_init__(self) -> None:
        # Check if model is available
        assert self.model_name in [
            "text_generation/fangloveskari/ORCA_LLaMA_70B_QLoRA",
        ]
        if "text_generation" in self.model_name.lower():
            self.model = Client(self.text_api_addr, timeout=120)
        else:
            if "gptq" in self.model_name.lower():
                self.model = AutoGPTQForCausalLM.from_quantized(
                    self.model_name,
                    use_safetensors=True,
                    trust_remote_code=False,
                    device_map="auto",
                    use_triton=False,
                    quantize_config=None,
                    inject_fused_attention=False,
                )
            else:
                # Prepare model loader
                if self.precision_num == 8:
                    from_pretrained = functools.partial(
                        AutoModelForCausalLM.from_pretrained, load_in_8bit=True
                    )
                elif self.precision_num == 4:
                    from_pretrained = functools.partial(
                        AutoModelForCausalLM.from_pretrained, load_in_4bit=True
                    )
                else:
                    from_pretrained = AutoModelForCausalLM.from_pretrained
                # Load model
                self.model = from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    rope_scaling={
                        "type": "dynamic",
                        "factor": 2,
                    },  # allows handling of longer inputs
                )
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, use_fast=True
            )
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.unk_token

            # Optimize speed
            self.model = self.model.to_bettertransformer()
            self.model.eval()

    def __call__(
        self,
        instruction_prompt_or_prompts: Union[str, List[str]],
        user_prompt_or_user_prompts: Union[str, List[str]],
        output_prefix_or_prefixes: Optional[Union[str, List[str]]] = None,
        stop_token: str = None,
    ) -> Union[str, List[str]]:
        if isinstance(instruction_prompt_or_prompts, str):
            # Decide which generate function to use
            if self.use_cache:
                generate_func = self.generate_with_cache
            else:
                generate_func = self.generate
            return generate_func(
                instruction_prompt=instruction_prompt_or_prompts,
                user_prompt=user_prompt_or_user_prompts,
                output_prefix=output_prefix_or_prefixes,
                stop_token=stop_token,
            )
        elif isinstance(instruction_prompt_or_prompts, list):
            if "text_generation" in self.model_name.lower():
                raise ValueError(
                    "Batch generation is not supported for model: {self.model_name}"
                )
            # Decide which generate function to use
            if self.use_cache:
                generate_func = self.batch_generate_with_cache
            else:
                generate_func = self.batch_generate
            return generate_func(
                instruction_prompts=instruction_prompt_or_prompts,
                user_prompts=user_prompt_or_user_prompts,
                output_prefixes=output_prefix_or_prefixes,
                stop_token=stop_token,
            )
        else:
            raise ValueError(
                f"Invalid type for instruction_prompt_or_prompts: {type(instruction_prompt_or_prompts)}"
            )

    # Helper methods to format input

    def format_input(
        self, instruction: str, prompt: str, output_prefix: str = None
    ) -> str:
        if any(
            [
                key in self.model_name.lower()
                for key in ["platypus", "orca", "posicube", "tuned"]
            ]
        ):
            return self.format_for_platypus(instruction, prompt, output_prefix)
        elif (
            "upstage" in self.model_name.lower()
            or "sheep-duck" in self.model_name.lower()
        ):
            return self.format_for_upstage_llama(instruction, prompt, output_prefix)
        elif "airoboros" in self.model_name.lower():
            return self.format_for_airboros(instruction, prompt, output_prefix)
        elif "gptq" in self.model_name.lower():
            return self.format_for_gptq(instruction, prompt, output_prefix)
        raise ValueError(f"Invalid model name: {self.model_name}")

    def format_for_upstage_llama(
        self, instruction: str, prompt: str, output_prefix: str
    ) -> str:
        output_prefix = output_prefix if output_prefix else ""
        return f"### System:\n{instruction}\n\n### User:\n{prompt}\n\n### Assistant:\n{output_prefix}"

    def format_for_platypus(
        self, instruction: str, prompt: str, output_prefix: str
    ) -> str:
        output_prefix = output_prefix if output_prefix else ""
        # Create prompt delimiter
        if instruction.endswith("\n\n"):
            prompt_delimiter = ""
        elif instruction.endswith("\n"):
            prompt_delimiter = "\n"
        else:
            prompt_delimiter = "\n\n"
        # Create response delimiter
        if prompt.endswith("\n\n"):
            response_delimiter = ""
        elif prompt.endswith("\n"):
            response_delimiter = "\n"
        else:
            response_delimiter = "\n\n"
        return f"### Instruction:\n{instruction}{prompt_delimiter}{prompt}{response_delimiter}### Response:\n{output_prefix}"

    def format_for_airboros(
        self, instruction: str, prompt: str, output_prefix: str
    ) -> str:
        output_prefix = output_prefix if output_prefix else ""
        return f"SYSTEM: {instruction}\nUSER: {prompt}\nASSISTANT: {output_prefix}"

    def format_for_gptq(self, instruction: str, prompt: str, output_prefix: str) -> str:
        if instruction:
            instruction = f"Instruction: {instruction}\n"
        output_prefix = output_prefix if output_prefix else ""
        return f"### User:{instruction}{prompt}\n### Assistant:{output_prefix}"

    def format_for_togethercomputer(self, instruction: str, prompt: str) -> str:
        return f"[INST]{instruction}\n{prompt}\n[/INST]\n\n"

    # Helper methods to parse output
    def parse_output(self, formatted_prompt: str, output: str) -> str:
        if output.startswith(formatted_prompt):
            return output[len(formatted_prompt) :]
        return output

    def start_chat(self) -> None:
        while True:
            instruction = input("Enter instruction: ").strip()
            prompt = input("Enter prompt: ").strip()
            print(self(instruction, prompt))

    def create_redis_key(
        self,
        instruction_prompt: str,
        user_prompt: str,
        output_prefix: str = None,
        stop_token: str = None,
    ) -> str:
        key = (
            f"{self.model_name}_{self.precision_num}_{instruction_prompt}_{user_prompt}"
        )
        if output_prefix:
            key += f"_{output_prefix}"
        if stop_token:
            key += f"_{stop_token}"
        return key

    def generate_with_cache(
        self,
        instruction_prompt: str,
        user_prompt: str,
        output_prefix: str = None,
        stop_token: str = None,
    ) -> str:
        rd_key = self.create_redis_key(
            instruction_prompt, user_prompt, output_prefix, stop_token
        )
        if not rd.exists(rd_key) or self.override_cache:
            result = self.generate(
                instruction_prompt, user_prompt, output_prefix, stop_token
            )
            rd.set(rd_key, result)
        return rd.get(rd_key).decode("utf-8")

    def batch_generate_with_cache(
        self,
        instruction_prompts: List[str],
        user_prompts: List[str],
        output_prefixes: List[str],
        stop_token: str = None,
    ) -> List[str]:
        # generate redis keys
        redis_keys = [
            self.create_redis_key(i, u, o, stop_token)
            for i, u, o in zip(instruction_prompts, user_prompts, output_prefixes)
        ]
        # Variable to store results
        results = {i: "" for i in range(len(redis_keys))}
        # Get cached results
        for idx, redis_key in enumerate(redis_keys):
            if rd.exists(redis_key):
                results[idx] = rd.get(redis_key).decode("utf-8")

        # Create new mini batch
        target_indices = []
        target_redis_keys = []
        target_instruction_prompts = []
        target_user_prompts = []
        target_output_prefixes = []
        for idx, redis_key in enumerate(redis_keys):
            if not results[idx]:
                target_indices.append(idx)
                target_redis_keys.append(redis_key)
                target_instruction_prompts.append(instruction_prompts[idx])
                target_user_prompts.append(user_prompts[idx])
                target_output_prefixes.append(
                    output_prefixes[idx] if output_prefixes else None
                )
        if target_indices:
            target_results = self.batch_generate(
                instruction_prompts=target_instruction_prompts,
                user_prompts=target_user_prompts,
                output_prefixes=target_output_prefixes,
                stop_token=stop_token,
            )
            ## Map key to results
            for target_idx, target_result in zip(target_indices, target_results):
                results[target_idx] = target_result

        # Save to redis
        for idx, output_text in results.items():
            redis_key = redis_keys[idx]
            rd.set(redis_key, output_text)
        # Return results
        return list(results.values())

    def generate(
        self,
        instruction_prompt: str,
        user_prompt: str,
        output_prefix: Optional[str] = None,
        stop_token: str = None,
    ) -> str:
        if "text_generation" in self.model_name.lower():
            return self.generate_through_text_generation(
                instruction_prompt=instruction_prompt,
                user_prompt=user_prompt,
                output_prefix=output_prefix,
                stop_token=stop_token,
            )
        # Generate tokens
        formatted_promt = self.format_input(
            instruction_prompt, user_prompt, output_prefix
        )
        # Tokenize
        inputs = self.tokenizer(formatted_promt, return_tensors="pt").to(
            self.model.device
        )
        streamer = TextStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                streamer=streamer,
                use_cache=True,
                max_new_tokens=self.max_new_token,
                do_sample=False,
                stop_token=stop_token,
            )
            output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # Parse output
        parsed_output_text = self.parse_output(formatted_promt, output_text)
        return parsed_output_text

    def batch_generate(
        self,
        instruction_prompts: List[str],
        user_prompts: List[str],
        output_prefixes: List[str],
    ) -> List[str]:
        ## Format input
        formatted_prompts = [
            self.format_input(i, u, o)
            for i, u, o in zip(instruction_prompts, user_prompts, output_prefixes)
        ]
        ## Tokenize
        inputs = self.tokenizer(
            formatted_prompts, padding=True, return_tensors="pt"
        ).to(self.model.device)

        ## Generate
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=self.max_new_token,
                do_sample=False,
            )
            ## Decode
            output_texts = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        parsed_results = [
            self.parse_output(f, o) for f, o in zip(formatted_prompts, output_texts)
        ]
        return parsed_results

    def generate_through_text_generation(
        self,
        instruction_prompt: str,
        user_prompt: str,
        output_prefix: str = None,
        stop_token: str = None,
    ):
        # Format input
        formatted_promt = self.format_input(
            instruction_prompt, user_prompt, output_prefix
        )
        # Generate
        print('formatted_promt', formatted_promt)
        response_list = requests.post(
                        "http://141.223.199.10:30000/generate",
                        json={
                            "text": [formatted_promt],
                            "sampling_params": {
                                "max_new_tokens": self.max_new_token,
                                "temperature": 0.0,
                            }
                        },
                        timeout=None,
                    ).json()
        response = response_list[0]["text"]
        print('response', response)
        return response


class LLMAPI(metaclass=pattern_utils.SingletonABCMetaWithArgs):
    def __init__(
        self,
        use_text_generation: bool = True,
        use_cache: bool = True,
        override_cache: bool = False,
        api_addr: Optional[str] = None,
        model_name: Optional[str] = None,
        precision: Optional[int] = None,
    ):
        self.use_text_generation = use_text_generation
        self.use_cache = use_cache
        self.override_cache = override_cache
        self.api_addr = TEXT_GENERATION_API_ADDR if api_addr is None else api_addr
        self.model_name = model_name
        self.precision = precision
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.use_text_generation:
            if self.model_name:
                llm_class = functools.partial(LLM, model_name=self.model_name)
            else:
                llm_class = LLM
            self.llm_model = llm_class(
                use_cache=self.use_cache,
                override_cache=self.override_cache,
                text_api_addr=self.api_addr,
                precision_num=self.precision,
            )

    def __call__(
        self,
        instruction_prompt_or_prompts: Union[str, List[str]],
        user_prompt_or_user_prompts: Union[str, List[str]],
        output_prefix_or_prefixes: Optional[Union[str, List[str]]] = None,
    ) -> Union[str, List[str]]:
        if isinstance(instruction_prompt_or_prompts, str):
            return self.generate(
                instruction_prompt=instruction_prompt_or_prompts,
                user_prompt=user_prompt_or_user_prompts,
                output_prefix=output_prefix_or_prefixes,
            )
        elif isinstance(instruction_prompt_or_prompts, list):
            return self.batch_generate(
                instruction_prompts=instruction_prompt_or_prompts,
                user_prompts=user_prompt_or_user_prompts,
                output_prefixes=output_prefix_or_prefixes,
            )
        else:
            raise ValueError(
                f"Invalid type for instruction_prompt_or_prompts: {type(instruction_prompt_or_prompts)}"
            )

    def start_chat(self) -> None:
        while True:
            instruction = input("Enter instruction: ").strip()
            prompt = input("Enter prompt: ").strip()
            print(self(instruction, prompt))

    def generate(
        self,
        instruction_prompt: str,
        user_prompt: str,
        output_prefix: str = None,
        stop_token: str = None,
    ) -> str:
        if self.use_text_generation:
            response = self.llm_model(
                instruction_prompt, user_prompt, output_prefix, stop_token
            )
        else:
            try:
                data = requests.post(
                    self.api_addr,
                    json={
                        "instructionPrompt": instruction_prompt,
                        "userPrompt": user_prompt,
                        "outputPrefix": output_prefix,
                    },
                    timeout=None,
                ).json()
            except Exception as e:
                raise RuntimeError(f"Error when requesting llm inference: {e}")
            response = data["response"]
        return response

    def batch_generate(
        self,
        instruction_prompts: List[str],
        user_prompts: List[str],
        output_prefixes: List[str] = None,
        stop_token: str = None,
    ) -> List[str]:
        if self.use_text_generation:
            if self.model_name:
                llm_class = functools.partial(LLM, model_name=self.model_name)
            else:
                llm_class = LLM
            llm_model = llm_class(
                use_cache=self.use_cache,
                override_cache=self.override_cache,
                text_api_addr=self.api_addr,
            )
            responses = llm_model(
                instruction_prompts, user_prompts, output_prefixes, stop_token
            )
        else:
            try:
                data = requests.post(
                    self.api_addr,
                    json={
                        "instructionPrompt": instruction_prompts,
                        "userPrompt": user_prompts,
                        "outputPrefix": output_prefixes,
                    },
                    timeout=None,
                ).json()
            except Exception as e:
                raise RuntimeError(f"Error when requesting llm inference: {e}")
            responses = data["response"]
        return responses


if __name__ == "__main__":
    llm = LLMAPI()

    response = llm.generate(
        instruction_prompt=self.instruction,
        user_prompt=user_prompt,
        output_prefix=output_prefix,
    ).strip()
