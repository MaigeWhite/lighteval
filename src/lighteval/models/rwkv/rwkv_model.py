from dataclasses import dataclass, fields
from typing import Any, Type
from typing import Optional

import itertools
from lighteval.models.abstract_model import LightevalModel, ModelInfo
from lighteval.data import GenerativeTaskDataset
from lighteval.utils.utils import EnvConfig, as_list
from lighteval.models.model_input import GenerationParameters
from tqdm import tqdm
import logging
from lighteval.models.model_output import GenerativeResponse
from lighteval.tasks.requests import GreedyUntilRequest
import ray
from more_itertools import distribute
from .model import RWKV
from .rwkv_tokenizer import TRIE_TOKENIZER
import torch


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
    load_model: str
    max_model_length: int = 32768
    max_new_tokens: int = 16384
    seed: int = 2024
    temperature: float = 1.0
    top_p: float = 0.0
    top_k: int = 10
    alpha_frequency: float = 0.5
    alpha_presence: float = 0.5
    alpha_decay: float = 0.996
    use_chat_template: bool = False
    add_special_tokens: bool = False
    ctx_len: int = 256
    num_gpus: int = 1
    strategy: str = "cuda fp16"
    n_layer: int = 24
    n_embd: int = 2048
    dim_att: int = 0
    dim_ffn: int = 0
    pre_ffn: int = 0
    grad_cp: int = 0
    head_size_a: int = 64
    head_size_divisor: int = 8
    vocab_size: int = 65536
    dropout: int = 0
    weight_decay: int = 0
    weight_decay_final: int = -1
    do_sample: bool = False
    batch_size: int = 32
    stop_token_idx: int = 261
    device: str = "gpu"
    pad_token_id: int = 0
    generation_parameters: GenerationParameters = None  # sampling parameters to use for generation

    def __post_init__(self):
        # 在对象初始化后调用 convert_types 进行类型转换
        convert_types(self, self.__class__)
        if self.dim_att <= 0:
            self.dim_att = self.n_embd
        if self.dim_ffn <= 0:
            self.dim_ffn = int((self.n_embd * 3.5) // 32 * 32)  # default = 3.5x emb size


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
        tokenizer = TRIE_TOKENIZER("rwkv_vocab_v20230424.txt")
        if self.rwkv_config.num_gpus > 1:
            return None, tokenizer

        model = RWKV(self.rwkv_config).half()
        if self.rwkv_config.load_model:
            model.load_state_dict(torch.load(self.rwkv_config.load_model, map_location="cpu"),
                                  strict=False)
        model.cuda()
        return model, tokenizer

    # Tokenization utils
    def tok_encode(self, str_to_encode: str | list[str], add_special_tokens: Optional[bool] = None):
        if isinstance(str_to_encode, str):
            return self.tokenizer.encode(str_to_encode)
        else:
            raise TypeError(f"only support str to encode but got{type(str_to_encode)}")

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
            samle_param = (self.rwkv_config.do_sample,
                           self.rwkv_config.temperature,
                           self.rwkv_config.top_p,
                           self.rwkv_config.max_new_tokens)
            outputs = self._generate(requests, *samle_param, stop_token_idx=self.rwkv_config.stop_token_idx)
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
                  *args,
                  stop_token_idx=261):
        if self.rwkv_config.num_gpus > 1:
            @ray.remote(num_gpus=1)
            def run_inference_one_model(requests, *args, stop_token_idx=261):
                model = RWKV(self.rwkv_config).half()
                model.load_state_dict(torch.load(self.rwkv_config.load_model, map_location="cpu"), strict=False)
                model.cuda()
                results = []
                for batch_input_id in requests:
                    generate_token, _, _ = model.generate(batch_input_id.cuda(), *args, stop_token_idx=stop_token_idx)
                    for gen in generate_token:
                        gen_text = self.tokenizer.decode(gen)
                        results.append(gen_text)
                return results

            # dispatch requests to all self.data_parallel_size workers, in interleaved fashion
            # interleaved important to balance context lengths across workers
            requests = [list(x) for x in distribute(self.rwkv_config.num_gpus, requests)]
            inputs = [self.batch_request(req) for req in requests]
            object_refs = [run_inference_one_model.remote(x, *args, stop_token_idx=self.rwkv_config.stop_token_idx) for
                           x in inputs]
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
            for req in tqdm(batch_requests):
                generate_token, _, _ = self.model.generate(req.cuda(), *args, stop_token_idx=stop_token_idx)
                for gen in generate_token:
                    outputs.append(self.tokenizer.decode(gen))
        return outputs
