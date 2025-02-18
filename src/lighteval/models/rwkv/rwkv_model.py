import time
from dataclasses import dataclass, fields
from typing import Any, Type
from typing import Optional

import itertools
from lighteval.models.abstract_model import LightevalModel, ModelInfo
from lighteval.data import GenerativeTaskDataset
from lighteval.utils.utils import EnvConfig, as_list
from lighteval.models.model_input import GenerationParameters
import logging
from lighteval.models.model_output import GenerativeResponse
from lighteval.tasks.requests import GreedyUntilRequest
import ray
from more_itertools import distribute
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

from tqdm import tqdm


def convert_types(data_obj: Any, cls: Type) -> None:
    """
    处理RWKVModelConfig对象中的字段值，尝试将它们转换为预期的数据类型。

    :param data_obj: 已经实例化的 RWKVModelConfig 对象
    :param cls: 数据类类型（例如 RWKVModelConfig）
    """
    schema = {field.name: field.type for field in fields(cls)}

    for key, expected_type in schema.items():
        raw_value = getattr(data_obj, key)
        if raw_value is not None:
            try:
                # 尝试将输入值转换为预期类型
                converted_value = expected_type(raw_value)
                setattr(data_obj, key, converted_value)
            except (ValueError, TypeError) as e:
                print(f"Warning: Cannot convert '{key}' to {expected_type.__name__}. Error: {e}")
                # 设置为None或其他默认值
                setattr(data_obj, key, None)
        else:
            # 如果键对应的值为None，可以选择设置为其他默认值
            setattr(data_obj, key, None)


@dataclass
class RWKVModelConfig:
    model: str
    max_model_length: int = 32768
    max_new_tokens: int = 2048
    seed: int = 2024
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 10
    use_chat_template: bool = False
    add_special_tokens: Optional[list] = None
    ctx_len: int = 256
    num_gpus: int = 1
    batch_size: int = 8
    generation_parameters: GenerationParameters = None  # sampling parameters to use for generation

    def __post_init__(self):
        # 在对象初始化后调用 convert_types 进行类型转换
        convert_types(self, self.__class__)


class RWKVModel(LightevalModel):
    def __init__(self,
                 config: RWKVModelConfig,
                 env_config: EnvConfig,
                 ):
        self.rwkv_config = config
        self.model, self._tokenizer = self.init_model()
        self.use_chat_template = config.use_chat_template
        self._add_special_tokens = config.add_special_tokens if config.add_special_tokens is not None else False
        self._max_length = int(config.max_model_length) if config.max_model_length is not None else None
        self.model_info = ModelInfo(model_name=config.model, model_sha="")

    def init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.rwkv_config.model, trust_remote_code=True)
        if self.rwkv_config.num_gpus > 1:
            return None, tokenizer

        model = AutoModelForCausalLM.from_pretrained(self.rwkv_config.model, trust_remote_code=True).cuda()
        return model, tokenizer

    def cleanup(self):
        pass

    def model_info(self):
        return self.model_info

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def add_special_tokens(self):
        return self._add_special_tokens

    @property
    def max_length(self) -> int:
        return self._max_length

    def loglikelihood(self):
        pass

    def loglikelihood_rolling(self):
        pass

    def loglikelihood_single_token(self):
        pass

    def greedy_until(
            self,
            requests: list[GreedyUntilRequest],
            override_bs: Optional[int] = None,
    ) -> list[GenerativeResponse]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            requests (list[Request]): list of requests containing the context and ending conditions.
            override_bs (int, optional): Override the batch size for generation. Defaults to None.

        Returns:
            list[GenerateReturn]: list of generated responses.
        """

        for request in requests:
            # request.stop_sequence = as_list(request.stop_sequence) + [self.tokenizer.eos_token]
            request.stop_sequence = as_list(request.stop_sequence)
            request.tokenized_context = self.tok_encode(request.context)

        dataset = GenerativeTaskDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
        results = []

        for _ in tqdm(
                dataset.splits_start_end_iterator(),
                total=dataset.num_dataset_splits,
                desc="Splits",
                position=0,
                disable=False,  # self.disable_tqdm,
        ):
            # context = [c.context for c in dataset]
            # left truncate the inputs to the maximum length
            samle_param = {"do_sample": self.rwkv_config.do_sample,
                           "temperature": self.rwkv_config.temperature,
                           "top_p": self.rwkv_config.top_p,
                           "max_new_tokens": self.rwkv_config.max_new_tokens,
                           "top_k": self.rwkv_config.top_k
                           }
            outputs = self._generate(requests, **samle_param)
            for result in outputs:
                cur_response = GenerativeResponse(
                    result=result)
                results.append(cur_response)

        return dataset.get_original_order(results)

    def pad_token(self, request, max_length):
        batch_tokenized = []
        for idx in range(len(request)):
            tokenized_context = request[idx].tokenized_context
            pad_length = max_length - len(tokenized_context)
            pad_tokenized = [self.rwkv_config.pad_token_id] * pad_length + tokenized_context
            batch_tokenized.append(pad_tokenized)
        return torch.tensor(batch_tokenized, dtype=torch.long)

    def batch_request(self, requests: list):
        batch_request = []
        for idx in range(0, len(requests), self.rwkv_config.batch_size):
            batch_req = requests[idx:idx + self.rwkv_config.batch_size]
            max_length = max([len(_.tokenized_context) for _ in batch_req])
            batch_request.append(self.pad_token(batch_req, max_length))
        return batch_request

    def _generate(self, requests,
                  **kwargs):
        if self.rwkv_config.num_gpus > 1:
            @ray.remote(num_gpus=1)
            def run_inference_one_model(requests, device, **kwargs):
                model = AutoModelForCausalLM.from_pretrained(self.rwkv_config.model, trust_remote_code=True).cuda()
                results = []
                with tqdm(total=len(batch_requests), postfix={'Throughput': 'N/A'}) as pbar:
                    for batch_input_id in requests:
                        start_time = time.time()
                        input_length = [len(_) for _ in batch_input_id]
                        generate_tokens = model.generate(batch_input_id.cuda(), **kwargs).cpu()
                        for idx in range(len(generate_tokens)):
                            out = generate_tokens[idx][input_length[idx]:]
                            text = self.tokenizer.batch_decode(out, skip_special_tokens=True)
                            results.append(text)
                        bs = len(input_length)
                        throughput = bs * self.rwkv_config.max_new_tokens / (end_time - start_time)
                        pbar.set_postfix({'Throughput': f'{throughput:.2f} token/s'})
                        pbar.update(1)
                return results

            # dispatch requests to all self.data_parallel_size workers, in interleaved fashion
            # interleaved important to balance context lengths across workers
            requests = [list(x) for x in distribute(self.rwkv_config.num_gpus, requests)]
            inputs = [self.batch_request(req) for req in requests]
            device_info = [f"cuda: {idx}" for idx in range(len(inputs))]
            object_refs = [run_inference_one_model.remote(inputs[idx], device_info[idx], **kwargs) for idx in
                           range(len(inputs))]
            results = ray.get(object_refs)
            # Invoke ray.shutdown() to prevent hang-ups if subsequent calls required.
            ray.shutdown()
            # flatten results
            outputs = [
                x
                for x in itertools.chain.from_iterable(itertools.zip_longest(*[list(x) for x in results]))
                if x is not None
            ]
        else:
            outputs = []
            batch_requests = self.batch_request(requests)
            with tqdm(total=len(batch_requests), postfix={'Throughput': 'N/A'}) as pbar:
                for req in batch_requests:
                    start_time = time.time()
                    input_length = [len(_) for _ in req]
                    generate_tokens = self.model.generate(req.cuda(), **kwargs).cpu()
                    for idx in range(len(generate_tokens)):
                        out = generate_tokens[idx][input_length[idx]:]
                        text = self.tokenizer.decode(out, skip_special_tokens=True)
                        outputs.append(text)
                    end_time = time.time()
                    bs = len(input_length)
                    throughput = bs * self.rwkv_config.max_new_tokens / (end_time - start_time)
                    pbar.set_postfix({'Throughput': f'{throughput:.2f} token/s'})
                    pbar.update(1)
        return outputs
