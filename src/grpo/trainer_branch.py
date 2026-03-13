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
# trainer_branch.py
#
# Branch-aware multi-turn RL trainer.
#
# Key differences from trainer_segment.py:
#   - Uses client_branch.sample_conversations_branch  (full conversations, not tree nodes)
#   - Uses RepeatSampler with num_generations so each problem gets N full rollouts
#   - GRPO advantage normalisation already performed server-side in compute_all_advantages_flat:
#       • accuracy / end_of_conversation / think(node)  →  per-problem GRPO
#       • pedagogical / length / think(turn)            →  per-(problem, turn-pos) GRPO
#   - normalize_tree_advantages flag optionally re-normalises in-trainer (global)
##########################################################################################

import torch

_orig_load = torch.load


def _unsafe_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)


torch.load = _unsafe_load

import os
import gc
import shutil
import re
import time
import wandb
import random
import deepspeed
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union, Tuple, Dict, List

import datasets
import torch
import transformers
from accelerate.utils import gather_object, is_peft_model, set_seed
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
from trl.trainer.utils import pad, selective_log_softmax

from src.utils.utils import incremental_state_dict
from src.vllm.client_branch import sample_conversations_branch, wait_batch
from src.utils.shared_memory import create_shared_state_dict, get_shareable_version
from src.utils.utils import init_logger, _ForwardRedirection
from src.grpo.config_branch import ClassroomBranchConfig

logger = init_logger()

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

class ClassroomBranchTrainer(Trainer):
    """
    GRPO trainer for the branch-based pedagogical RL setup.

    Compared to ClassroomSPOTrainer (trainer_segment.py):
      • Uses RepeatSampler (num_generations rollouts per problem)
      • Calls sample_conversations_branch → full conversation sequences
      • Per-turn token-level advantages from server-computed GRPO groups
    """

    _tag_names = ["trl", "grpo-branch"]

    def __init__(
        self,
        model: str,
        args: ClassroomBranchConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer],
            Optional[torch.optim.lr_scheduler.LambdaLR],
        ] = (None, None),
    ):
        if not isinstance(model, str):
            raise ValueError(f"`model` must be a string; got {type(model)}")

        processing_class = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if (
            processing_class.pad_token_id is None
            or processing_class.pad_token_id == processing_class.eos_token_id
        ):
            processing_class.pad_token_id = processing_class.vocab_size - 1
            processing_class.pad_token = processing_class.convert_ids_to_tokens(
                processing_class.pad_token_id
            )

        self.server_port = args.vllm_server_port
        self.use_experimental_shared_memory = args.use_experimental_shared_memory

        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = ClassroomBranchConfig(f"{model_name}-Branch_GRPO")

        self.model_name_or_path = model

        model_init_kwargs = args.model_init_kwargs or {}
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, str) and torch_dtype != "auto":
            torch_dtype = getattr(torch, torch_dtype)
            model_init_kwargs["torch_dtype"] = torch_dtype

        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
        model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **model_init_kwargs)

        if args.peft_config is not None:
            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=args.gradient_checkpointing
                )
            model = get_peft_model(model, args.peft_config)

        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        self.beta = args.beta
        self.ref_model = (
            None
            if self.beta == 0.0
            else AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **model_init_kwargs)
        )

        def data_collator(features):
            return features

        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.temperature = args.temperature

        self.num_iterations = args.num_iterations
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
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

        self._metrics = {"train": defaultdict(list), "etc": defaultdict(list)}

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

        self.global_batch_size = (
            self.args.per_device_train_batch_size
            * self.num_processes
            * self.args.gradient_accumulation_steps
        )

        # We need it for the global_batch_size to be divisible by the number of generations.
        if self.global_batch_size % self.num_generations != 0:
            raise ValueError(
                f"Global batch size {self.global_batch_size} is not divisible by the number of generations {self.num_generations}."
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

        self._kl_save_thresholds = self.args.kl_save_thresholds
        self._kl_threshold_idx = 0

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "answer", "solve_rates"]

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )
        dataloader_params = {
            "batch_size": self._train_batch_size
            * self.args.gradient_accumulation_steps,  # < this is the change
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

        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.global_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.gradient_accumulation_steps,
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

    # ──────────────────────────────────────────────────────────────────────
    # Token-level advantage mapping  (identical to trainer_segment.py)
    # ──────────────────────────────────────────────────────────────────────

    def _find_assistant_turns_llama(self, seq, start_header_tok, end_header_tok, assistant_tok, eot_tok):
        spans = []
        inside_asst = inside_msg = False
        current_start = None
        for i, tok in enumerate(seq):
            if tok == start_header_tok:
                inside_msg = True
                inside_asst = False
            elif inside_msg and tok == assistant_tok:
                inside_asst = True
            elif tok == end_header_tok and inside_asst:
                current_start = i + 1
            elif tok == eot_tok and inside_asst and current_start is not None:
                spans.append((current_start, i + 1))
                inside_msg = inside_asst = False
                current_start = None
        return spans

    def _find_assistant_turns_qwen(self, seq, start_token, assistant_token, eos_token):
        spans = []
        inside_asst = inside_msg = False
        current_start = None
        newline_toks = set(
            self.processing_class.encode("\n", add_special_tokens=False)
        )
        for i, tok in enumerate(seq):
            if tok == start_token:
                inside_msg = True
                inside_asst = False
            elif inside_msg and tok == assistant_token:
                inside_asst = True
                j = i + 1
                while j < len(seq) and seq[j] in newline_toks:
                    j += 1
                current_start = j
            elif tok == eos_token and inside_asst and current_start is not None:
                spans.append((current_start, i + 1))
                inside_msg = inside_asst = False
                current_start = None
        return spans

    def _map_node_advantages_to_tokens_disentangled(
        self,
        completion_ids: torch.Tensor,          # (B, L)
        node_advantages_list: List[List[Dict]],
        reward_list: List[str],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Map per-turn-pair advantages to token-level tensors.

        Each element of node_advantages_list corresponds to one conversation's
        list of main-turn TurnPair dicts (already filtered by client_branch).
        The number of assistant turns in the token sequence must match the
        number of TurnPair dicts.
        """
        batch_size, seq_len = completion_ids.shape
        device = completion_ids.device

        advantage_key_map = {
            "accuracy":              "accuracy_advantage",
            "pedagogical_alignment": "pedagogical_advantage",
            "pedagogical_stage_alignment": "pedagogical_stage_alignment_advantage",
            "think":                 "think_advantage",
            "length":                "length_advantage",
            "end_of_conversation":   "end_of_conversation_advantage",
        }

        token_advantages_per_reward = {
            r: torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
            for r in reward_list
        }
        padding_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if "llama" in self.model_name_or_path.lower():
            sh = self.processing_class.encode("<|start_header_id|>", add_special_tokens=False)[0]
            eh = self.processing_class.encode("<|end_header_id|>", add_special_tokens=False)[0]
            at = self.processing_class.encode("assistant", add_special_tokens=False)[0]
            et = self.processing_class.encode("<|eot_id|>", add_special_tokens=False)[0]
        else:
            st = self.processing_class.apply_chat_template([{"role": "system", "content": ""}])[0]
            at = self.processing_class.encode("assistant")[0]
            et = self.processing_class.eos_token_id

        for b in range(batch_size):
            seq = completion_ids[b].tolist()
            turn_pairs = node_advantages_list[b]

            if any(t.get("is_padding", False) for t in turn_pairs):
                padding_mask[b] = True
                continue

            if "llama" in self.model_name_or_path.lower():
                spans = self._find_assistant_turns_llama(seq, sh, eh, at, et)
            else:
                spans = self._find_assistant_turns_qwen(seq, st, at, et)

            if len(spans) != len(turn_pairs):
                logger.warning(
                    f"Batch {b}: assistant turns ({len(spans)}) != turn pairs ({len(turn_pairs)}). Skipping."
                )
                continue

            for turn_idx, (start_pos, end_pos) in enumerate(spans):
                tp = turn_pairs[turn_idx]
                for reward in reward_list:
                    key = advantage_key_map[reward]
                    adv = tp.get(key, 0.0) or 0.0
                    token_advantages_per_reward[reward][b, start_pos:end_pos] = adv

        return token_advantages_per_reward, padding_mask

    # ──────────────────────────────────────────────────────────────────────
    # Assistant mask
    # ──────────────────────────────────────────────────────────────────────

    def _compute_assistant_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        if "llama" in self.model_name_or_path.lower():
            sh = self.processing_class.encode("<|start_header_id|>", add_special_tokens=False)[0]
            eh = self.processing_class.encode("<|end_header_id|>", add_special_tokens=False)[0]
            at = self.processing_class.encode("assistant", add_special_tokens=False)[0]
            et = self.processing_class.encode("<|eot_id|>", add_special_tokens=False)[0]

            def _mask(seq):
                mask, inside_msg, inside_asst, tokens_in = [], False, False, 0
                for tok in seq:
                    if tok == sh:
                        inside_msg, inside_asst, tokens_in = True, False, 0
                        mask.append(0)
                    elif inside_msg and tok == at:
                        inside_asst, tokens_in = True, 0
                        mask.append(0)
                    elif tok == eh:
                        mask.append(0)
                    elif tok == et:
                        mask.append(1 if inside_asst else 0)
                        inside_msg = inside_asst = False
                    elif inside_msg and inside_asst:
                        mask.append(1 if tokens_in >= 3 else 0)
                        tokens_in += 1
                    else:
                        mask.append(0)
                return mask

        else:
            st = self.processing_class.apply_chat_template([{"role": "system", "content": ""}])[0]
            at = self.processing_class.encode("assistant")[0]
            eos = self.processing_class.eos_token_id
            eos_set = {eos} if not isinstance(eos, list) else set(eos)

            newline_toks = set(
                self.processing_class.encode("\n", add_special_tokens=False)
            )

            def _mask(seq):
                mask, inside_msg, inside_asst, skip_until = [], False, False, -1
                for i, tok in enumerate(seq):
                    if tok == st:
                        inside_msg = True
                        mask.append(0)
                    elif inside_msg and tok == at:
                        inside_asst = True
                        j = i + 1
                        while j < len(seq) and seq[j] in newline_toks:
                            j += 1
                        skip_until = j
                        mask.append(0)
                    elif tok in eos_set:
                        mask.append(1 if inside_asst else 0)
                        inside_msg = inside_asst = False
                        skip_until = -1
                    elif inside_msg and inside_asst:
                        mask.append(1 if i >= skip_until else 0)
                    else:
                        mask.append(0)
                assert len(mask) == len(seq)
                return mask

        if input_ids.dim() == 1:
            return torch.tensor(_mask(input_ids.tolist()), device=input_ids.device)
        return torch.tensor(
            [_mask(row.tolist()) for row in input_ids],
            device=input_ids.device,
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
        inputs = sorted(inputs, key=lambda x: str(x))
        answers = [x["answer"] for x in inputs]
        solve_rates = [x["solve_rates"] for x in inputs]
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
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

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
        
        for _ in range(5):
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(1)

        # We only generate on main processes
        if self.accelerator.is_local_main_process:
            all_prompts = prompts_text
            logger.info(f"Entered with {len(all_prompts)} prompts")
            unique_prompts = all_prompts[:: self.num_generations]
            unique_answers = answers[:: self.num_generations]
            unique_solve_rates = solve_rates[:: self.num_generations]

            num_per_node = len(unique_prompts) // self.num_nodes
            start = self.node_id * num_per_node
            end = (self.node_id + 1) * num_per_node
            node_prompts = unique_prompts[start:end]
            node_answers = unique_answers[start:end]
            node_solve_rates = unique_solve_rates[start:end]
            expanded_problem_idxs = [
                start + i
                for i in range(len(node_prompts))
                for _ in range(self.num_generations)
            ]

            # Expand: send each problem num_generations times so the server
            # gets N rollouts per problem for grouping in compute_all_advantages_flat
            expanded_problems, expanded_answers, expanded_solve_rates = [], [], []
            for p, a, s in zip(node_prompts, node_answers, node_solve_rates):
                expanded_problems.extend([p] * self.num_generations)
                expanded_answers.extend([str(a)] * self.num_generations)
                expanded_solve_rates.extend([s] * self.num_generations)

            logger.info(
                f"Generating completions for {len(unique_prompts)} unique problems, with {self.num_generations} generations each and answers {unique_answers}"
            )

            all_completion_ids, all_turn_pair_advs, all_trajectory_rewards, all_problem_idx, all_accuracy_logs = (
                sample_conversations_branch(
                    problems=expanded_problems,
                    problem_idxs=expanded_problem_idxs,
                    answers=expanded_answers,
                    solve_rates=expanded_solve_rates,
                    meta=meta_info_shared,
                    server_port=self.server_port,
                    tokenizer=self.processing_class,
                )
            )

            if self.use_experimental_shared_memory and meta_info:
                for meta in meta_info.values():
                    meta["_shm_obj"].close()
                    meta["_shm_obj"].unlink()
        else:
            all_completion_ids, all_turn_pair_advs, all_trajectory_rewards, all_problem_idx, all_accuracy_logs = (
                [], [], [], [], []
            )
            if self.use_experimental_shared_memory and meta_info:
                try:
                    for meta in meta_info.values():
                        meta["_shm_obj"].close()
                        meta["_shm_obj"].unlink()
                except Exception:
                    pass
                gc.collect()
                torch.cuda.empty_cache()
            time.sleep(2 * 60)
            wait_batch(server_port=self.server_port)

        self.accelerator.wait_for_everyone()

        if self.use_experimental_shared_memory:
            try:
                for meta in meta_info.values():
                    meta["_shm_obj"].close()
                    meta["_shm_obj"].unlink()
            except Exception:
                pass

        self.accelerator.wait_for_everyone()

        # Gather across processes
        all_completion_ids = gather_object(all_completion_ids)
        all_turn_pair_advs = gather_object(all_turn_pair_advs)
        all_trajectory_rewards = gather_object(all_trajectory_rewards)
        all_problem_idx = gather_object(all_problem_idx)
        all_accuracy_logs = gather_object(all_accuracy_logs)

        # Distribute across processes
        all_pids = sorted(set(all_problem_idx))
        num_problems = len(all_pids)
        probs_per_proc = num_problems // self.num_processes
        start_pid = self.accelerator.process_index * probs_per_proc
        end_pid = (
            (self.accelerator.process_index + 1) * probs_per_proc
            if self.accelerator.process_index < self.num_processes - 1
            else num_problems
        )
        my_pids = set(all_pids[start_pid:end_pid])
        my_idx = [i for i, pid in enumerate(all_problem_idx) if pid in my_pids]

        completion_ids = [all_completion_ids[i] for i in my_idx]
        turn_pair_advs = [all_turn_pair_advs[i] for i in my_idx]
        trajectory_rewards = [all_trajectory_rewards[i] for i in my_idx]
        problem_idx = [all_problem_idx[i] for i in my_idx]
        accuracy_logs = [all_accuracy_logs[i] for i in my_idx]

        completion_ids = pad(
            [torch.tensor(ids, device=device) for ids in completion_ids],
            padding_value=self.processing_class.pad_token_id,
        )
        completion_mask = (completion_ids != self.processing_class.pad_token_id).int()
        attention_mask = completion_mask

        # Token-level advantages
        token_advantages_per_reward, padding_mask = (
            self._map_node_advantages_to_tokens_disentangled(
                completion_ids, turn_pair_advs, self.args.reward_list
            )
        )

        problem_to_indices = defaultdict(list)
        for i, pid in enumerate(problem_idx):
            problem_to_indices[pid].append(i)

        # Optional in-trainer normalisation
        if self.args.normalize_tree_advantages:
            assistant_mask_bool = self._compute_assistant_mask(completion_ids).bool()
            assistant_mask_bool[padding_mask] = False

            token_level_advantages = torch.zeros(completion_ids.shape, dtype=torch.float32, device=device)
            normalized_adv_means = {}

            for reward, weight in zip(self.args.reward_list, self.args.reward_weights):
                adv = token_advantages_per_reward[reward].clone()
                active_vals = adv[assistant_mask_bool & ~padding_mask.unsqueeze(1)]
                if active_vals.numel() >= 2:
                    mean_g = active_vals.mean()
                    std_g = active_vals.std().clamp(min=1e-4)
                    adv = torch.where(assistant_mask_bool, (adv - mean_g) / std_g, adv)
                normalized_adv_means[reward] = adv[assistant_mask_bool].mean().item() if assistant_mask_bool.any() else 0.0
                token_level_advantages += weight * adv

            assistant_mask = assistant_mask_bool.float()
        else:
            token_level_advantages = sum(
                token_advantages_per_reward[r] * w
                for r, w in zip(self.args.reward_list, self.args.reward_weights)
            )
            assistant_mask = self._compute_assistant_mask(completion_ids).float()
            assistant_mask[padding_mask] = 0.0

        # Log-probs
        logits_to_keep = completion_ids.size(1) - 1
        bs = self.args.per_device_train_batch_size

        with torch.no_grad():
            old_per_token_logps = (
                self._get_per_token_logps(self.model, completion_ids, attention_mask, logits_to_keep, bs)
                if self.num_iterations > 1
                else None
            )
            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, completion_ids, attention_mask, logits_to_keep, bs
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, completion_ids, attention_mask, logits_to_keep, bs
                    )

        # Sequence-level advantage for logging
        comp_mask_f = (completion_ids != self.processing_class.pad_token_id).float()
        masked_adv = token_level_advantages * assistant_mask * comp_mask_f
        active_tok = (assistant_mask * comp_mask_f).sum(dim=1).clamp(min=1)
        sequence_advantages = masked_adv.sum(dim=1) / active_tok

        logger.info(
            f"[branch] advantages: mean={sequence_advantages.mean():.3f}, "
            f"std={sequence_advantages.std():.3f}, "
            f"min={sequence_advantages.min():.3f}, max={sequence_advantages.max():.3f}"
        )

        # Metrics
        mode = "eval" if self.control.should_evaluate else "train"

        self._metrics[mode]["completion_length"].append(
            self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        )

        # Per-turn-pair reward/advantage breakdown
        buckets: Dict[str, list] = {
            k: [] for k in [
                "accuracy_rewards", "eoc_rewards", "pedagogical_stage_rewards", 
                "think_rewards", "pedagogical_rewards", "length_rewards",
                "accuracy_advs", "eoc_advs", "pedagogical_stage_advs",
                "think_advs", "pedagogical_advs", "length_advs",
                # NOTE: combined_advantage is never stored on TurnPair;
                # combined_A_avg is computed from sequence_advantages below.
            ]
        }

        for turn_pair_list in turn_pair_advs:
            for turn_pair in turn_pair_list:
                if turn_pair.get("is_padding"):
                    continue
                buckets["pedagogical_rewards"].append(turn_pair.get("pedagogical_reward") or 0.0)
                buckets["length_rewards"].append(turn_pair.get("length_reward") or 0.0)
                if self.args.is_think_turn_reward:
                    buckets["think_rewards"].append(turn_pair.get("think_reward") or 0.0)
                buckets["accuracy_advs"].append(turn_pair.get("accuracy_advantage") or 0.0)
                buckets["eoc_advs"].append(turn_pair.get("end_of_conversation_advantage") or 0.0)
                buckets["pedagogical_stage_advs"].append(turn_pair.get("pedagogical_stage_alignment_advantage") or 0.0)
                buckets["pedagogical_advs"].append(turn_pair.get("pedagogical_advantage") or 0.0)
                buckets["length_advs"].append(turn_pair.get("length_advantage") or 0.0)
                buckets["think_advs"].append(turn_pair.get("think_advantage") or 0.0)

        for nr in trajectory_rewards:
            acc = nr.get("accuracy_reward")
            eoc = nr.get("end_of_conversation_reward")
            psr = nr.get("pedagogical_stage_alignment_reward")
            tnk = nr.get("think_reward")
            if acc is not None:
                buckets["accuracy_rewards"].append(acc)
            if eoc is not None:
                buckets["eoc_rewards"].append(eoc)
            if psr is not None:
                buckets["pedagogical_stage_rewards"].append(psr)
            if tnk is not None and not self.args.is_think_turn_reward:
                buckets["think_rewards"].append(tnk)

        def _m(lst):
            return sum(lst) / len(lst) if lst else 0.0

        reward_name_map = {
            "accuracy":              ("accuracy_rewards",    "accuracy_advs"),
            "end_of_conversation":   ("eoc_rewards",         "eoc_advs"),
            "pedagogical_alignment": ("pedagogical_rewards", "pedagogical_advs"),
            "pedagogical_stage_alignment": ("pedagogical_stage_rewards", "pedagogical_stage_advs"),
            "length":                ("length_rewards",      "length_advs"),
            "think":                 ("think_rewards",       "think_advs"),
        }
        for reward in self.args.reward_list:
            rk, ak = reward_name_map[reward]
            self._metrics[mode][f"{reward}_R_avg"].append(_m(buckets[rk]))
            if self.args.normalize_tree_advantages:
                self._metrics[mode][f"{reward}_A_avg"].append(
                    normalized_adv_means.get(reward, 0.0)
                )
            else:
                self._metrics[mode][f"{reward}_A_avg"].append(_m(buckets[ak]))

        # combined_A_avg: always derived from the weighted token-level advantages
        self._metrics[mode]["combined_A_avg"].append(
            self.accelerator.gather_for_metrics(sequence_advantages.mean()).float().mean().item()
        )

        # etc. metrics for rewards not in reward_list
        all_rewards = {"accuracy", "end_of_conversation", "pedagogical_alignment", "pedagogical_stage_alignment", "length", "think"}
        for reward in all_rewards - set(self.args.reward_list):
            rk, ak = reward_name_map[reward]
            self._metrics["etc"][f"{reward}_R_avg"].append(_m(buckets[rk]))
            self._metrics["etc"][f"{reward}_A_avg"].append(_m(buckets[ak]))

        nonzero_ratio = (token_level_advantages.abs() > 1e-6).float().sum() / token_level_advantages.numel()
        self._metrics[mode]["nonzero_A_ratio"].append(
            self.accelerator.gather_for_metrics(nonzero_ratio).mean().item()
        )

        accuracy_log_buckets: Dict[str, list] = {
            k: [] for k in ["prior_accuracy", "post_accuracy", "delta_accuracy"]
        }

        for al in accuracy_logs:
            for k in accuracy_log_buckets:
                accuracy_log_buckets[k].append(al.get(k, 0.0))

        for k, v in accuracy_log_buckets.items():
            self._metrics[mode][k].append(
                self.accelerator.gather_for_metrics(
                    torch.tensor(v, device=device).float()
                ).mean().item()
            )

        return {
            "padding_mask":         padding_mask,
            "completion_ids":       completion_ids,
            "completion_mask":      completion_mask,
            "token_advantages":     token_level_advantages,
            "sequence_advantages":  sequence_advantages,
            "old_per_token_logps":  old_per_token_logps,
            "ref_per_token_logps":  ref_per_token_logps,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Loss
    # ──────────────────────────────────────────────────────────────────────

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("ClassroomBranchTrainer does not support return_outputs")
        torch.cuda.empty_cache()

        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        attention_mask = (completion_ids != self.processing_class.pad_token_id).int()
        completion_mask = completion_mask[:, 1:]
        logits_to_keep = completion_ids.size(1) - 1

        assistant_mask = self._compute_assistant_mask(completion_ids).int()[:, 1:]
        padding_mask = inputs.get("padding_mask")
        if padding_mask is not None:
            assistant_mask[padding_mask] = 0

        per_token_logps = self._get_per_token_logps(
            model, completion_ids, attention_mask, logits_to_keep,
            self.args.per_device_train_batch_size,
        )

        old_per_token_logps = (
            inputs["old_per_token_logps"]
            if self.num_iterations > 1
            else per_token_logps.detach()
        )
        ref_per_token_logps = inputs.get("ref_per_token_logps")

        token_advantages = inputs["token_advantages"][:, 1:]

        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss = -torch.min(coef_1 * token_advantages, coef_2 * token_advantages)

        if self.beta != 0.0 and ref_per_token_logps is not None:
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )
            per_token_loss = per_token_loss + self.beta * per_token_kl
        else:
            per_token_kl = None

        active = completion_mask * assistant_mask
        loss = (per_token_loss * active).sum() / active.sum().clamp(min=1)

        mode = "eval" if self.control.should_evaluate else "train"
        if per_token_kl is not None:
            kl_mean = (per_token_kl * active).sum() / active.sum().clamp(min=1)
            kl_val = self.accelerator.gather_for_metrics(kl_mean).mean().item()
            self._metrics[mode]["kl"].append(kl_val)

            # If KL divergence exceeds thresholds, save the model checkpoint for analysis.
            while (
                self._kl_threshold_idx < len(self._kl_save_thresholds)
                and kl_val >= self._kl_save_thresholds[self._kl_threshold_idx]
            ):
                threshold = self._kl_save_thresholds[self._kl_threshold_idx]
                policy_root = os.path.join(self.args.output_dir, "policy")
                if self.accelerator.is_main_process and os.path.isdir(policy_root):
                    ckpts = sorted([
                        d for d in os.listdir(policy_root)
                        if re.match(r"^checkpoint-\d+$", d)
                    ], key=lambda d: int(d.split("-")[1]))
                    if ckpts:
                        os.makedirs(os.path.join(self.args.output_dir, "kl_triggered"), exist_ok=True)
                        latest = os.path.join(policy_root, ckpts[-1])
                        dst = os.path.join(
                            self.args.output_dir,
                            "kl_triggered",
                            f"kl{threshold:.2f}_step{self.state.global_step}",
                        )
                        logger.warning(
                            f"[KL trigger] kl={kl_val:.4f} >= {threshold:.2f}, "
                            f"copying {latest} → {dst}"
                        )
                        shutil.copytree(latest, dst, dirs_exist_ok=True)
                self._kl_threshold_idx += 1

        is_clipped = (coef_1 * token_advantages < coef_2 * token_advantages).float()
        clip_ratio = (is_clipped * active).sum() / active.sum().clamp(min=1)
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )

        seq_adv = inputs["sequence_advantages"]
        self._metrics[mode]["advantage_mean"].append(
            self.accelerator.gather_for_metrics(seq_adv.mean()).mean().item()
        )
        self._metrics[mode]["advantage_std"].append(
            self.accelerator.gather_for_metrics(seq_adv.std()).mean().item()
        )

        torch.cuda.empty_cache()
        return loss

    # ──────────────────────────────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────────────────────────────

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {k: sum(v) / len(v) for k, v in self._metrics[mode].items()}
        if self._metrics["etc"]:
            metrics.update(
                {f"{k}(etc)": sum(v) / len(v) for k, v in self._metrics["etc"].items()}
            )
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics[mode].clear()
        self._metrics["etc"].clear()
