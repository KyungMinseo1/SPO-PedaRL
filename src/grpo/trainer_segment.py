# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

##########################################################################################
# Slightly modified from official TRL Library.
# - Changed the `_generate_and_score_completions` function to work with our vLLM server.
# - Added `_compute_assistant_mask` function to mask user turns for loss computation.
##########################################################################################


import torch

_orig_load = torch.load

# Fixes an error related to model loading.
def _unsafe_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)


torch.load = _unsafe_load

import os
import gc
import shutil, os, re
import time
import wandb
import deepspeed
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union, Tuple, Dict, List

import datasets
import torch
import transformers
from accelerate.utils import (
    gather,
    gather_object,
    is_peft_model,
    set_seed,
)
from peft import get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available

from trl.data_utils import maybe_apply_chat_template
from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
from trl.models import prepare_deepspeed
from trl.extras.profiling import profiling_decorator
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import (
    pad,
    selective_log_softmax,
)
from vllm import RequestOutput
from src.utils.utils import incremental_state_dict
from src.vllm.client import sample_conversations, sample_nodes, wait_batch
from src.utils.shared_memory import create_shared_state_dict, get_shareable_version
from src.utils.utils import init_logger, _ForwardRedirection
from src.grpo.config_segment import ClassroomSPOConfig

logger = init_logger()


RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(
                self.num_samples, generator=self.generator
            ).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [
            indexes[i : i + self.batch_size]
            for i in range(0, len(indexes), self.batch_size)
        ]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class RepeatRandomSampler(RepeatSampler):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "RepeatRandomSampler is deprecated and will be removed in version 0.18. Use RepeatSampler instead.",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)


def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.

    Example:
        >>> x = torch.arange(12).reshape(6, 2)
        >>> y = torch.arange(6).reshape(6, 1)
        >>> tensor_dict = {"x": x, "y": y}
        >>> split_tensor_dict(tensor_dict, 3)
        [
            {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]])},
            {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]])},
            {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]])}
        ]
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    return [
        {
            key: (
                tensor[i * chunk_size : (i + 1) * chunk_size]
                if tensor is not None
                else None
            )
            for key, tensor in tensor_dict.items()
        }
        for i in range(num_chunks)
    ]


class ClassroomSPOTrainer(Trainer):

    _tag_names = ["trl", "spo"]

    def __init__(
        self,
        model: str,
        # reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: ClassroomSPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
    ):
        ################################################################################
        # Checks
        ################################################################################
        if not isinstance(model, str):
            raise ValueError(
                "The `model` argument must be a string representing the model name. "
                f"Got {type(model)} instead."
            )
        processing_class = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        # We need that eos_token_id != pad_token_id. If pad_token_id is None we set it to the last token of the vocab.
        if (
            processing_class.pad_token_id is None
            or processing_class.pad_token_id == processing_class.eos_token_id
        ):
            processing_class.pad_token_id = processing_class.vocab_size - 1
            processing_class.pad_token = processing_class.convert_ids_to_tokens(
                processing_class.pad_token_id
            )
            logger.warning(
                f"Setting pad_token_id to {processing_class.pad_token_id} ({processing_class.pad_token}) "
                "because it was None or equal to eos_token_id."
            )

        self.server_port = args.vllm_server_port
        self.use_experimental_shared_memory = args.use_experimental_shared_memory

        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = ClassroomSPOConfig(f"{model_name}-GRPO")

        ################################################################################
        # Model loading
        ################################################################################
        self.model_name_or_path = model

        # Loading the policy model
        model_init_kwargs = args.model_init_kwargs or {}

        torch_dtype = model_init_kwargs.get("torch_dtype")
        if (
            isinstance(torch_dtype, torch.dtype)
            or torch_dtype == "auto"
            or torch_dtype is None
        ):
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
            model_init_kwargs["torch_dtype"] = torch_dtype
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )

        # Disable caching if gradient checkpointing is enabled (not supported)
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, **model_init_kwargs
        )

        if args.peft_config is not None:
            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
            model = get_peft_model(model, args.peft_config)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Loading the reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        else:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path, **model_init_kwargs
            )

        ################################################################################
        # Final setups
        ################################################################################
        # self.reward_funcs = reward_funcs

        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length

        self.num_generations = args.num_generations
        
        self.temperature = args.temperature

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = (
            args.epsilon_high if args.epsilon_high is not None else args.epsilon
        )
        self.epsilon = args.epsilon
        self.use_liger_loss = args.use_liger_loss
        if self.use_liger_loss:
            self._forward_redirection = _ForwardRedirection()

        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle.
        self._step = 0

        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list)}

        self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
            beta=self.beta,
            epsilon_low=self.epsilon_low,
            epsilon_high=self.epsilon_high,
            temperature=self.temperature,
            use_ref_model=self.ref_model is not None,
        )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.num_processes = self.accelerator.num_processes
        self.num_nodes = int(os.getenv("NNODES", "1"))
        self.node_id = self.accelerator.process_index // torch.cuda.device_count()
        logger.info(
            f"Node ID: {self.node_id}, Process ID: {self.accelerator.process_index}, Number of processes: {self.num_processes}"
        )

        # This means number of completions per process.
        self.global_batch_size = (
            self.args.per_device_train_batch_size
            * self.num_processes
            * self.args.gradient_accumulation_steps
        )

        # We need it for the global_batch_size to be divisible by the number of generations.
        if self.global_batch_size < self.num_generations:
            raise ValueError(
                f"Global batch size {self.global_batch_size} must be >= num_generations {self.num_generations}."
            )
        self.number_of_problems_per_batch = (
            self.global_batch_size // self.num_generations
        )

        set_seed(args.seed, device_specific=True)

        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "answer"]

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        dataloader_params = {
            "batch_size": self.number_of_problems_per_batch // self.num_processes,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self) -> Sampler:
        return RepeatSampler(
            self.train_dataset,
            mini_repeat_count=1,
            batch_size=self.number_of_problems_per_batch // self.num_processes,
            repeat_count=self.num_iterations * self.args.gradient_accumulation_steps,
            shuffle=True,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(
        self, model: PreTrainedModel, args: GRPOConfig
    ) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs
            or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    ####################################################################################
    # Map node-level advantages to token-level advantages (TreeNode 기반)
    ####################################################################################
    def _map_node_advantages_to_tokens(
        self,
        completion_ids: torch.Tensor,  # (B, L)
        node_advantages_list: List[List[Dict]],  # List[List[node_turn_pair_adv_dict]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Node별 advantage를 토큰 레벨로 매핑 (Context 없음! 각 node 독립!)
        
        각 sequence는 하나의 TreeNode의 turn pairs만 포함:
        - max_group_size개의 turn pairs (예: 3개)
        - 각 turn에 해당 turn의 combined_advantage 적용
        
        Args:
            completion_ids: (batch_size, seq_len)
            node_advantages_list: 각 node의 turn pair advantages (3개)
        
        Returns:
            token_advantages: (batch_size, seq_len) - 각 토큰의 advantage
        """
        batch_size, seq_len = completion_ids.shape
        device = completion_ids.device
        
        # 결과 텐서 초기화
        token_advantages = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
        padding_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Chat template tokens
        if "llama" in self.model_name_or_path.lower():
            start_header_tok = self.processing_class.encode(
                "<|start_header_id|>", add_special_tokens=False
            )[0]
            end_header_tok = self.processing_class.encode(
                "<|end_header_id|>", add_special_tokens=False
            )[0]
            assistant_tok = self.processing_class.encode(
                "assistant", add_special_tokens=False
            )[0]
            eot_tok = self.processing_class.encode(
                "<|eot_id|>", add_special_tokens=False
            )[0]
        else:
            start_token = self.processing_class.apply_chat_template(
                [{"role": "system", "content": ""}]
            )[0]
            assistant_token = self.processing_class.encode("assistant")[0]
            eos_token = self.processing_class.eos_token_id
        
        # 각 시퀀스에 대해 처리
        for batch_idx in range(batch_size):
            seq = completion_ids[batch_idx].tolist()
            node_turn_pairs = node_advantages_list[batch_idx]

            if any(t.get("is_padding", False) for t in node_turn_pairs):
                token_advantages[batch_idx, :] = 0.0  # Making Dummy Features
                padding_mask[batch_idx] = True
                continue
            
            if not node_turn_pairs:
                continue
            
            # Assistant 턴의 시작/끝 위치 찾기
            if "llama" in self.model_name_or_path.lower():
                assistant_turn_spans = self._find_assistant_turns_llama(
                    seq, start_header_tok, end_header_tok, assistant_tok, eot_tok
                )
            else:
                assistant_turn_spans = self._find_assistant_turns_qwen(
                    seq, start_token, assistant_token, eos_token
                )
            
            # Context 없음! 노드의 turn pairs와 정확히 매칭되어야 함
            num_node_turns = len(node_turn_pairs) # Number of turns in one node
            num_assistant_turns = len(assistant_turn_spans) # Number of assistant turns in the node sequence
            
            if num_assistant_turns != num_node_turns:
                logger.warning(
                    f"Batch {batch_idx}: Assistant turns ({num_assistant_turns}) "
                    f"!= node turns ({num_node_turns}). Using fallback."
                )
                # Fallback
                if node_turn_pairs:
                    token_advantages[batch_idx, :] = node_turn_pairs[0]['combined_advantage']
                continue
            
            # 각 assistant 턴에 해당하는 advantage 할당
            for turn_idx, (start_pos, end_pos) in enumerate(assistant_turn_spans):
                if turn_idx < len(node_turn_pairs):
                    adv = node_turn_pairs[turn_idx]['combined_advantage']
                    token_advantages[batch_idx, start_pos:end_pos] = adv
        
        return token_advantages, padding_mask
    
    def _find_assistant_turns_llama(self, seq, start_header_tok, end_header_tok, assistant_tok, eot_tok):
        """Llama-3 형식의 assistant 턴 찾기"""
        spans = []
        inside_msg = False
        inside_asst = False
        current_start = None
        
        for i, tok in enumerate(seq):
            if tok == start_header_tok:
                inside_msg = True
                inside_asst = False
            elif inside_msg and tok == assistant_tok:
                inside_asst = True
            elif tok == end_header_tok and inside_asst:
                # Assistant content 시작
                current_start = i + 1
            elif tok == eot_tok and inside_asst and current_start is not None:
                # Assistant content 끝
                spans.append((current_start, i + 1))  # eot_tok 포함
                inside_msg = False
                inside_asst = False
                current_start = None
        
        return spans
    
    def _find_assistant_turns_qwen(self, seq, start_token, assistant_token, eos_token):
        """Qwen 형식의 assistant 턴 찾기"""
        spans = []
        inside_msg = False
        inside_asst = False
        current_start = None
        
        for i, tok in enumerate(seq):
            if tok == start_token:
                inside_msg = True
                inside_asst = False
            elif inside_msg and tok == assistant_token:
                inside_asst = True
                # Assistant content는 다음 토큰부터 시작
                # 보통 assistant 토큰 뒤에 몇 개의 special token이 더 있음
                # 간단히 +3 정도로 skip (실제로는 더 정교하게 해야 할 수도)
                current_start = i + 3
            elif tok == eos_token and inside_asst and current_start is not None:
                # Assistant content 끝
                spans.append((current_start, i + 1))
                inside_msg = False
                inside_asst = False
                current_start = None
        
        return spans

    ####################################################################################
    # Mask user turns, used for multi-turn RL Training
    ####################################################################################
    def _compute_assistant_mask(self, input_ids):

        if "llama" in self.model_name_or_path.lower():
            # Llama-3 chat-template tokens
            start_header_tok = self.processing_class.encode(
                "<|start_header_id|>", add_special_tokens=False
            )[0]
            end_header_tok = self.processing_class.encode(
                "<|end_header_id|>", add_special_tokens=False
            )[0]
            assistant_tok = self.processing_class.encode(
                "assistant", add_special_tokens=False
            )[0]
            eot_tok = self.processing_class.encode(
                "<|eot_id|>", add_special_tokens=False
            )[0]

            def compute_mask_for_sequence(seq):
                mask = []
                inside_msg = inside_asst = False
                tokens_in = 0
                for tok in seq:
                    if tok == start_header_tok:
                        inside_msg = True
                        inside_asst = False
                        tokens_in = 0
                        mask.append(0)
                    elif inside_msg and tok == assistant_tok:
                        inside_asst = True
                        tokens_in = 0
                        mask.append(0)
                    elif tok == end_header_tok:
                        mask.append(0)
                    elif tok == eot_tok:
                        mask.append(1 if inside_asst else 0)
                        inside_msg = inside_asst = False
                    elif inside_msg and inside_asst:
                        mask.append(1 if tokens_in >= 3 else 0)
                        tokens_in += 1
                    else:
                        mask.append(0)
                logger.debug(f"Number of assistant tokens in sequence: {tokens_in}")
                return mask

            if input_ids.dim() == 1:
                return torch.tensor(
                    compute_mask_for_sequence(input_ids.tolist()),
                    device=input_ids.device,
                )
            elif input_ids.dim() == 2:
                return torch.tensor(
                    [compute_mask_for_sequence(seq.tolist()) for seq in input_ids],
                    device=input_ids.device,
                )
            else:
                raise ValueError(
                    "Unsupported input_ids dimensions: expected 1D or 2D tensor."
                )
        else:
            # Get the special tokens.
            start_token = self.processing_class.apply_chat_template(
                [{"role": "system", "content": ""}]
            )[0]
            system_token = self.processing_class.encode("system")[0]
            user_token = self.processing_class.encode("user")[0]
            assistant_token = self.processing_class.encode("assistant")[0]
            eos_token = self.processing_class.eos_token_id

            def compute_mask_for_sequence(seq):
                """
                Given a 1D list (or tensor converted to list) of tokens,
                splits it into messages based on the start token and computes the mask.
                """
                mask = []
                inside_assistant_message = False
                inside_a_message = False
                tokens_in = 0
                for token in seq:
                    if token == start_token:
                        inside_a_message = True
                        mask.append(0)
                    elif inside_a_message and token == assistant_token:
                        inside_assistant_message = True
                        tokens_in = 0
                        mask.append(0)
                    elif token == eos_token:
                        mask.append(1 if inside_assistant_message else 0)
                        inside_a_message = False
                        inside_assistant_message = False
                    elif inside_a_message and inside_assistant_message:
                        mask.append(1 if tokens_in >= 3 else 0)
                        tokens_in += 1
                    else:
                        mask.append(0)

                assert len(mask) == len(seq)
                return mask

            # If input_ids is a 1D tensor (a single sequence).
            if input_ids.dim() == 1:
                # Convert to a list of ints if needed.
                seq = input_ids.tolist()
                mask = compute_mask_for_sequence(seq)
                return torch.tensor(mask, device=input_ids.device)

            # If input_ids is a 2D tensor (batch of sequences).
            elif input_ids.dim() == 2:
                batch_masks = []
                # Process each sequence in the batch independently.
                for seq_tensor in input_ids:
                    seq = seq_tensor.tolist()
                    mask = compute_mask_for_sequence(seq)
                    batch_masks.append(mask)
                # Convert the list of masks into a tensor.
                return torch.tensor(batch_masks, device=input_ids.device)

            else:
                raise ValueError(
                    "Unsupported input_ids dimensions: expected 1D or 2D tensor."
                )

    @profiling_decorator
    def _get_last_hidden_state(
        self, model, input_ids, attention_mask, logits_to_keep=None
    ):
        # unwrap the model to access the model.model
        unwrapped_model = self.accelerator.unwrap_model(model)
        last_hidden_state = unwrapped_model.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[
                :, -logits_to_keep:, :
            ]  # (B, logits_to_keep, H)
        return last_hidden_state

    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(
        self, model, input_ids, attention_mask, logits_to_keep, batch_size=None
    ) -> torch.Tensor:
        batch_size = batch_size or input_ids.size(
            0
        )  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]

            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                logits_to_keep=logits_to_keep + 1,
            ).logits
            logits = logits[
                :, :-1, :
            ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -logits_to_keep:]
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            logps = selective_log_softmax(
                logits, input_ids_batch
            )  # compute logprobs for the input tokens
            all_logps.append(logps)
        return torch.cat(all_logps, dim=0)

    @profiling_decorator
    def _prepare_inputs(
        self, accumulated_local_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:

        generate_every = self.args.gradient_accumulation_steps * self.num_iterations
        if self._step % generate_every == 0 or self._buffered_inputs is None:
            # self._buffered_inputs=None can occur when resuming from a checkpoint
            accumulated_local_batch = self._generate_and_score_completions(
                accumulated_local_batch
            )
            self._buffered_inputs = split_tensor_dict(
                accumulated_local_batch, self.args.gradient_accumulation_steps
            )
        inputs = self._buffered_inputs[
            self._step % self.args.gradient_accumulation_steps
        ]
        self._step += 1

        return inputs

    def _save_only_model(self, output_dir: str):
        """
        Save only the model, then delete any other checkpoint-* folders
        under the same `policy` directory so that only the latest remains.
        """
        CHECKPOINT_RE = re.compile(r"^checkpoint-\d+$")
        if self.accelerator.is_main_process:

            # Save current model
            os.makedirs(output_dir, exist_ok=True)
            # Prune older checkpoints
            policy_root = os.path.dirname(output_dir)  # …/policy
            for d in os.listdir(policy_root):
                if CHECKPOINT_RE.match(d):
                    shutil.rmtree(os.path.join(policy_root, d), ignore_errors=True)

        self.accelerator.wait_for_everyone()
        self.save_model(output_dir, _internal_call=True)

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        inputs = gather_object(inputs)
        # sort inputs by prompt
        inputs = sorted(inputs, key=lambda x: str(x))
        prompts = [x["prompt"] for x in inputs]
        answers = [x["answer"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        prompt_inputs = self.processing_class(
            prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        ############################################################################################################
        # vLLM Classroom generation
        # ############################################################################################################

        state_dict = None
        meta_info = None
        meta_info_shared = {}

        # If we are using shared memory.
        if self.use_experimental_shared_memory:
            state_dict = incremental_state_dict(self.model)
            if self.accelerator.is_local_main_process:
                logger.info("Creating shared memory")

                meta_info = create_shared_state_dict(state_dict)
                meta_info_shared = get_shareable_version(meta_info)

            # Free up memory
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Shared memory created")

        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()
        # We also sometimes save the model every self.args.save_policy_to_disk_every_n_steps steps.
        # We do this to avoid a bug with vllm when changing weights at tensor parallel=4.
        if self.state.global_step % self.args.save_policy_to_disk_every_n_steps == 0:
            logger.info(f"Saving the model to {self.args.output_dir}")
            self._save_only_model(
                os.path.join(
                    self.args.output_dir,
                    "policy",
                    f"checkpoint-{self.state.global_step}",
                )
            )
            self.accelerator.wait_for_everyone()
            logger.info(f"Model saved to {self.args.output_dir}")

        all_prompts_text = prompts_text
        for _ in range(5):
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(1)
        
        # We only generate on main processes
        if self.accelerator.is_local_main_process:
            logger.info(f"Entered with {len(all_prompts_text)} prompts")
            
            # TreeNode 기반: 중복 없음! 그대로 사용
            ordered_set_of_prompts = all_prompts_text
            local_answers = answers

            num_prompts_per_node = len(ordered_set_of_prompts) // self.num_nodes

            # We get our slice of the prompts
            start_slice = self.node_id * num_prompts_per_node
            end_slice = (self.node_id + 1) * num_prompts_per_node
            ordered_set_of_prompts = ordered_set_of_prompts[start_slice:end_slice]
            local_answers = local_answers[start_slice:end_slice]
            
            # TreeNode 기반: answers도 중복 없음
            real_answers = local_answers

            logger.info(
                f"Generating completions for {len(ordered_set_of_prompts)} unique problems, with {self.num_generations} generations each and answers {real_answers}"
            )

            # all_completion_ids: Per Node generation request outputs(sequence_text(tokenized), token_ids)
            # all_node_advantages: Per Node advantage list(doubled list for each turn pair per node([[turn1, turn2], [turn3, turn4]]), with combined advantage)
            all_completion_ids, all_node_advantages, all_node_rewards = sample_nodes(
                ordered_set_of_prompts,
                answers=real_answers,
                meta=meta_info_shared,
                server_port=self.server_port,
                tokenizer=self.processing_class,
            )
            logger.info(f"Generated {len(all_completion_ids)} nodes from {len(ordered_set_of_prompts)} problems")

            expected_count = len(ordered_set_of_prompts) * self.num_generations
            actual_count = len(all_completion_ids)

            # Actual count might differ from expected count due to early stopping in conversations.
            if actual_count != expected_count:
                logger.warning(
                    f"Expected {expected_count} outputs, but got {actual_count}. This might lead to issues in training."
                )
                if actual_count > expected_count:
                    logger.info(f"Truncating to {expected_count} conversations")
                    all_outputs = all_completion_ids[:expected_count]
                    all_node_advantages = all_node_advantages[:expected_count]
                elif actual_count < expected_count:
                    logger.info(f"Padding to {expected_count} nodes")
                    padding_count = expected_count - actual_count
                    dummy_output = all_completion_ids[-1]
                    dummy_advantage = [
                        {
                            "accuracy_advantage": 0.0,
                            "pedagogical_advantage": 0.0,
                            "length_advantage": 0.0,
                            "end_of_conversation_advantage": 0.0,
                            "combined_advantage": 0.0,
                            "is_padding": True,
                        }
                    ]
                    all_completion_ids.extend([dummy_output] * padding_count)
                    all_node_advantages.extend([dummy_advantage] * padding_count)

            if self.use_experimental_shared_memory:
                logger.info("Closing the shared memory")

                for meta in meta_info.values():
                    shm = meta["_shm_obj"]
                    shm.close()
                    shm.unlink()
        else:
            all_completion_ids = []
            all_node_advantages = []
            all_node_rewards = []
            if self.use_experimental_shared_memory:
                logger.info("Deleting state_dict memory")
                try:
                    for meta in meta_info.values():
                        shm = meta["_shm_obj"]
                        shm.close()
                        shm.unlink()
                except:
                    pass
                gc.collect()
                torch.cuda.empty_cache()
            time.sleep(2 * 60)
            logger.info("Waiting for the batch to be ready")
            wait_batch(
                server_port=self.server_port,
            )

        self.accelerator.wait_for_everyone()
        if self.use_experimental_shared_memory:
            logger.info("Closing the shared memory")
            try:
                for meta in meta_info.values():
                    shm = meta["_shm_obj"]
                    shm.close()
                    shm.unlink()
            except Exception as e:
                pass
            logger.info("Shared memory closed")
        self.accelerator.wait_for_everyone()
        all_completion_ids_ = gather_object(all_completion_ids)
        all_completion_ids = all_completion_ids_
        all_node_advantages_ = gather_object(all_node_advantages)
        all_node_advantages = all_node_advantages_
        all_node_rewards_ = gather_object(all_node_rewards)
        all_node_rewards = all_node_rewards_

        step_size = len(all_completion_ids) // self.num_processes
        start_slice = self.accelerator.process_index * step_size

        logger.info(f"Start slice: {start_slice}, Step size: {step_size}")

        node_advantages = all_node_advantages[start_slice : start_slice + step_size]
        completion_ids = all_completion_ids[start_slice : start_slice + step_size]
        node_rewards = all_node_rewards[start_slice : start_slice + step_size]

        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(
            completion_ids, padding_value=self.processing_class.pad_token_id
        )
        # prompt_completion_ids: tensored token ids for generated sequences
        prompt_completion_ids = completion_ids

        # TODO: Technically problematic if pad_token_id is equal to eos_token_id.
        completion_mask = (completion_ids != self.processing_class.pad_token_id).int()
        attention_mask = completion_mask
        logits_to_keep = completion_ids.size(1) - 1

        batch_size = self.args.per_device_train_batch_size

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size,
                    )

        # NEW: Mapping per node advantages to token level advantages
        logger.info("Mapping node-level advantages to token-level advantages")
        token_level_advantages, padding_mask = self._map_node_advantages_to_tokens(
            completion_ids,
            node_advantages  # each node's turn pair advantages
        )  # (batch_size, seq_len)
        
        # Assistant mask
        assistant_mask = self._compute_assistant_mask(completion_ids).float()  # (B, L)
        assistant_mask[padding_mask] = 0.0 # Making padded dummy sequences
        
        # Completion mask
        completion_mask_float = (completion_ids != self.processing_class.pad_token_id).float()
        
        # Adding masks to per token advantages
        masked_token_advantages = token_level_advantages * assistant_mask * completion_mask_float
        
        # Each sequence's mean advantage (per-sequence advantage for logging)
        active_token_counts = (assistant_mask * completion_mask_float).sum(dim=1).clamp(min=1)
        sequence_advantages = masked_token_advantages.sum(dim=1) / active_token_counts
        
        logger.info(
            f"Token-level advantages: mean={sequence_advantages.mean():.3f}, "
            f"std={sequence_advantages.std():.3f}, "
            f"min={sequence_advantages.min():.3f}, max={sequence_advantages.max():.3f}"
        )

        # Normalization(optional, group-wise)
        if hasattr(self.args, 'normalize_tree_advantages') and self.args.normalize_tree_advantages:
            # Group-wise normalization
            adv_grouped_mean = sequence_advantages.view(-1, self.num_generations).mean(dim=1)
            adv_grouped_mean = adv_grouped_mean.repeat_interleave(self.num_generations, dim=0)
            sequence_advantages = sequence_advantages - adv_grouped_mean

            token_level_advantages = token_level_advantages - adv_grouped_mean.unsqueeze(1)

            if self.args.scale_rewards:
                adv_grouped_std = sequence_advantages.view(-1, self.num_generations).std(dim=1)
                adv_grouped_std = adv_grouped_std.repeat_interleave(self.num_generations, dim=0)
                sequence_advantages = sequence_advantages / (adv_grouped_std + 1e-4)

                token_level_advantages = token_level_advantages / (adv_grouped_std.unsqueeze(1) + 1e-4)

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1))
            .float()
            .mean()
            .item()
        )
        self._metrics[mode]["completion_length"].append(completion_length)

        # Per-reward advantage logging (turn-pair level breakdown)
        all_accuracy_rewards = []
        all_pedagogical_rewards = []
        all_length_rewards = []
        all_end_of_conv_rewards = []
        all_accuracy_advs = []
        all_pedagogical_advs = []
        all_length_advs = []
        all_end_of_conv_advs = []
        all_combined_advs = []

        for node_turn_pairs in node_advantages:
            for turn in node_turn_pairs:
                if turn.get("is_padding", False):
                    continue
                all_pedagogical_rewards.append(turn.get("pedagogical_reward", 0.0))
                all_length_rewards.append(turn.get("length_reward", 0.0))
                all_accuracy_advs.append(turn.get("accuracy_advantage", 0.0))
                all_pedagogical_advs.append(turn.get("pedagogical_advantage", 0.0))
                all_length_advs.append(turn.get("length_advantage", 0.0))
                all_end_of_conv_advs.append(turn.get("end_of_conversation_advantage", 0.0))
                all_combined_advs.append(turn.get("combined_advantage", 0.0))

        for node_reward in node_rewards:
            acc = node_reward.get("accuracy_reward")
            eoc = node_reward.get("end_of_conversation_reward")
            if acc is not None:
                all_accuracy_rewards.append(acc)
            if eoc is not None:
                all_end_of_conv_rewards.append(eoc)

        def _safe_mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        self._metrics[mode]["accuracy_reward_mean"].append(_safe_mean(all_accuracy_rewards))
        self._metrics[mode]["pedagogical_reward_mean"].append(_safe_mean(all_pedagogical_rewards))
        self._metrics[mode]["length_reward_mean"].append(_safe_mean(all_length_rewards))
        self._metrics[mode]["end_of_conv_reward_mean"].append(_safe_mean(all_end_of_conv_rewards))
        self._metrics[mode]["accuracy_advantage_mean"].append(_safe_mean(all_accuracy_advs))
        self._metrics[mode]["pedagogical_advantage_mean"].append(_safe_mean(all_pedagogical_advs))
        self._metrics[mode]["length_advantage_mean"].append(_safe_mean(all_length_advs))
        self._metrics[mode]["end_of_conv_advantage_mean"].append(_safe_mean(all_end_of_conv_advs))
        self._metrics[mode]["combined_advantage_mean"].append(_safe_mean(all_combined_advs))

        # Ratio of tokens actually receiving a non-zero advantage signal
        nonzero_ratio = (token_level_advantages.abs() > 1e-6).float().sum() / token_level_advantages.numel()
        self._metrics[mode]["nonzero_advantage_ratio"].append(
            self.accelerator.gather_for_metrics(nonzero_ratio).mean().item()
        )

        logger.info(f"Returning completion_ids shape: {completion_ids.shape}")
        logger.info(f"Returning token_advantages shape: {token_level_advantages.shape}")
        logger.info(f"gradient_accumulation_steps: {self.args.gradient_accumulation_steps}")

        return {
            "padding_mask": padding_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "token_advantages": token_level_advantages,  # NEW: (B, L) - 토큰별 advantage
            "sequence_advantages": sequence_advantages,  # NEW: (B,) - 시퀀스별 평균 (로깅용)
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }

    def _compute_token_level_grpo_loss(
        self,
        per_token_logps: torch.Tensor,  # (B, L)
        old_per_token_logps: torch.Tensor,  # (B, L)
        ref_per_token_logps: Optional[torch.Tensor],  # (B, L)
        token_advantages: torch.Tensor,  # (B, L)
        completion_mask: torch.Tensor,  # (B, L)
        assistant_mask: torch.Tensor,  # (B, L)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Token-level advantage를 사용한 GRPO loss 계산
        
        Args:
            per_token_logps: 현재 policy의 per-token log probabilities
            old_per_token_logps: 이전 policy의 per-token log probabilities
            ref_per_token_logps: Reference model의 per-token log probabilities
            token_advantages: Token-level advantages (각 토큰마다 다름!)
            completion_mask: Padding mask
            assistant_mask: Assistant turn mask
            
        Returns:
            loss: scalar loss
            metrics: dict of metric tensors
        """
        # Policy ratio
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon_high)
        
        # Token-level policy loss
        per_token_loss1 = coef_1 * token_advantages
        per_token_loss2 = coef_2 * token_advantages
        per_token_policy_loss = -torch.min(per_token_loss1, per_token_loss2)
        
        # KL divergence (if using reference model)
        if self.beta != 0.0 and ref_per_token_logps is not None:
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )
            per_token_loss = per_token_policy_loss + self.beta * per_token_kl
        else:
            per_token_loss = per_token_policy_loss
            per_token_kl = None
        
        # Apply masks
        active_mask = completion_mask * assistant_mask
        masked_loss = per_token_loss * active_mask
        
        # Average over active tokens
        loss = masked_loss.sum() / active_mask.sum().clamp(min=1)
        
        # Metrics
        metrics = {}
        
        if per_token_kl is not None:
            metrics['kl'] = (per_token_kl * active_mask).sum() / active_mask.sum().clamp(min=1)
        
        # Clip ratio
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        metrics['clip_ratio'] = (is_clipped * active_mask).sum() / active_mask.sum().clamp(min=1)
        
        return loss, metrics

    def compute_old_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )

        # input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        input_ids = completion_ids  # New: we only need the completions
        # attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        # attention_mask = completion_mask # New: we only need the completions
        attention_mask = (completion_ids != self.processing_class.pad_token_id).int()
        # we remove one token from completion_mask at the start.
        completion_mask = completion_mask[:, 1:]
        logits_to_keep = (
            completion_ids.size(1) - 1
        )  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep,
            self.args.per_device_train_batch_size
        )

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        # NEW: Use token-level advantages!
        token_advantages = inputs["token_advantages"][:, 1:]  # (B, L) -> (B, L-1) to match logits
        
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = (
            inputs["old_per_token_logps"]
            if self.num_iterations > 1
            else per_token_logps.detach()
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        
        # NEW: 각 토큰이 자신의 advantage를 사용!
        per_token_loss1 = coef_1 * token_advantages  # (B, L) * (B, L)
        per_token_loss2 = coef_2 * token_advantages  # (B, L) * (B, L)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # We mask with assistant loss too.
        assistant_mask = self._compute_assistant_mask(completion_ids).int()[:, 1:]
        padding_mask = inputs.get("padding_mask", None)
        if padding_mask is not None:
            assistant_mask[padding_mask] = 0

        loss = (per_token_loss * completion_mask * assistant_mask).sum() / (
            assistant_mask * completion_mask
        ).sum()
        # loss = (per_token_loss * completion_mask).sum() / (completion_mask).sum()

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (
                (per_token_kl * completion_mask * assistant_mask).sum(dim=1) / completion_mask.sum(dim=1)
            ).mean()
            self._metrics[mode]["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).mean().item()
            )

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )
        
        # NEW: Log sequence-level advantages for monitoring
        sequence_advantages = inputs["sequence_advantages"]
        self._metrics[mode]["advantage_mean"].append(
            self.accelerator.gather_for_metrics(sequence_advantages.mean()).mean().item()
        )
        self._metrics[mode]["advantage_std"].append(
            self.accelerator.gather_for_metrics(sequence_advantages.std()).mean().item()
        )
        
        torch.cuda.empty_cache()
        return loss

    @profiling_decorator
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        torch.cuda.empty_cache()

        # NEW: Always use token-level loss (not Liger)
        if not self.use_liger_loss:
            return self.compute_old_loss(
                model, inputs, return_outputs, num_items_in_batch
            )
        
        # Token-level GRPO loss with Liger-like optimizations
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_ids = completion_ids

        attention_mask = (completion_ids != self.processing_class.pad_token_id).int()
        completion_mask = completion_mask[:, 1:]  # Remove first token
        logits_to_keep = completion_ids.size(1) - 1

        # We mask with assistant loss too.
        assistant_mask = self._compute_assistant_mask(completion_ids).int()[:, 1:]
        padding_mask = inputs.get("padding_mask", None)
        if padding_mask is not None:
            assistant_mask[padding_mask] = 0

        # Get per-token log probabilities
        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep,
            self.args.per_device_train_batch_size
        )
        
        # Get reference log probabilities
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
        else:
            ref_per_token_logps = None
        
        # Get old log probabilities
        old_per_token_logps = (
            inputs["old_per_token_logps"]
            if self.num_iterations > 1
            else per_token_logps.detach()
        )
        
        # NEW: Get token-level advantages
        token_advantages = inputs["token_advantages"][:, 1:]  # (B, L)
        
        # Compute token-level GRPO loss
        loss, metrics = self._compute_token_level_grpo_loss(
            per_token_logps=per_token_logps,
            old_per_token_logps=old_per_token_logps,
            ref_per_token_logps=ref_per_token_logps,
            token_advantages=token_advantages,
            completion_mask=completion_mask,
            assistant_mask=assistant_mask,
        )
        
        # Log metrics
        mode = "train" if self.model.training else "eval"
        
        if 'kl' in metrics:
            self._metrics[mode]["kl"].append(
                self.accelerator.gather_for_metrics(metrics['kl']).mean().item()
            )
        
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(metrics['clip_ratio']).mean().item()
        )
        
        # NEW: Log advantages
        sequence_advantages = inputs["sequence_advantages"]
        self._metrics[mode]["advantage_mean"].append(
            self.accelerator.gather_for_metrics(sequence_advantages.mean()).mean().item()
        )
        self._metrics[mode]["advantage_std"].append(
            self.accelerator.gather_for_metrics(sequence_advantages.std()).mean().item()
        )
        
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics[mode].items()
        }  # average the metrics

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()