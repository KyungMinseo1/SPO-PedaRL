from functools import lru_cache
import re
import gc
import torch
import time
import json
import pandas as pd
from tqdm import tqdm
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from random import choice
from jinja2 import Template
from pydantic import BaseModel
from transformers import AutoTokenizer
from vllm import PoolingOutput, SamplingParams, RequestOutput
from config.classification_config import (
    ClassifierConfig,
    GenerationConfig,
)
from src.vllm.data_parallel_vllm import ParallelvLLMInference, InferenceTask
from src.inference_providers.open_router_inference import OpenRouterInference
from src.inference_providers.gemini_api_inference import GeminiInference
import logging

logger = logging.getLogger(__name__)

class ClassifiedStage(Enum):
    STAGE_11 = "1-1"
    STAGE_12 = "1-2"
    STAGE_13 = "1-3"
    STAGE_14 = "1-4"
    STAGE_21 = "2-1"
    STAGE_22 = "2-2"
    STAGE_23 = "2-3"
    STAGE_24 = "2-4"
    STAGE_31 = "3-1"
    STAGE_32 = "3-2"
    STAGE_33 = "3-3"
    STAGE_41 = "4-1"
    STAGE_42 = "4-2"
    STAGE_51 = "5-1"

class ClassificationResponse(BaseModel):
    reasoning: str
    stage: ClassifiedStage
    confidence: str

@lru_cache(maxsize=1000)
def read_template(path: str) -> Template:
    return Template(open(path).read())


@lru_cache(maxsize=1)
def get_tokenizer(tokenizer_to_use: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(tokenizer_to_use)

class Classification:
    def __init__(
        self,
        classifier_config: ClassifierConfig,
        generation_config: GenerationConfig,
        log_file_path: Optional[str] = None,
    ):
        self.classifier_config = classifier_config
        self.generation_config = generation_config
        self.tokenizer = get_tokenizer(generation_config.tokenizer_to_use)

        if self.classifier_config.use_openrouter:
            self.classifier_model = OpenRouterInference(
                self.classifier_config.model_name_or_path
            )
        elif self.classifier_config.use_gemini:
             self.classifier_model = GeminiInference(
                self.classifier_config.model_name_or_path
            )
        else:
            self.classifier_model = ParallelvLLMInference(
                model_path=self.classifier_config.model_name_or_path,
                gpus_per_instance=self.classifier_config.vllm.number_of_gpus_per_instance,
                gpu_memory_utilization=self.classifier_config.vllm.gpu_memory_utilization,
                max_model_len=self.classifier_config.vllm.max_length,
                max_num_seqs=self.classifier_config.vllm.max_num_seqs,
                load_and_unload=self.classifier_config.vllm.load_and_unload,
                max_number_of_instances=self.classifier_config.vllm.max_number_of_instances,
                enable_sleep_mode=self.classifier_config.vllm.enable_sleep_mode,
                bits_and_bytes=self.classifier_config.vllm.bits_and_bytes,
                use_awq=self.classifier_config.vllm.use_awq,
                from_0=self.classifier_config.vllm.from_0,
                use_v0=self.classifier_config.vllm.use_v0,
                enforce_eager=self.classifier_config.vllm.enforce_eager,
                logging_enabled=log_file_path != None,
                log_file_path=log_file_path,
            )
        
        self.sampling_params_classifier = SamplingParams(
            temperature=self.classifier_config.vllm.temperature,
            top_k=self.classifier_config.vllm.top_k,
            top_p=self.classifier_config.vllm.top_p,
            max_tokens=self.generation_config.max_tokens_per_turn,
            # logits_processors=(
            #     [force_thinking_processor] if generation_cfg.force_thinking else []
            # ),
        )

        self.classification_results_set = []

    def _hide_thinking(self, content: str):
        return re.sub(r"<think>.*?</think>", "", content, flags=re.S).replace(
            "<end_of_conversation>", ""
        )

    def _get_hidden_conversation(self, turn_segment: List[dict]) -> List[dict]:
        conversation = []
        for message in turn_segment:
            conversation.append(
                {
                    "role": message["role"],
                    "content": self._hide_thinking(message["content"]),
                }
            )
        return conversation
    
    def sample_classification_results(
        self,
        all_turn_segment: List[List[dict]],
    ) -> List[dict]:
        classification_prompts = []
        for turn_segment in all_turn_segment:
            classification_prompts.append([
                [
                    {
                        "role": "user",
                        "content": Template(
                            open(
                                self.generation_config.classification_prompt_path
                            ).read()
                        ).render(conversation=self._get_hidden_conversation(turn_segment)),
                    }
                ],
                turn_segment
            ])
        prompts = [msg for msg, _ in classification_prompts]
        classification_results = []
        responses = self.classifier_model.run_batch(prompts, self.sampling_params_classifier)

        failed = []
        for (prompt, turn_segment), response in zip(classification_prompts, responses):
            for output in response.outputs:
                try:
                    out_text = output.text[
                        output.text.find("{") : output.text.rfind("}") + 1
                    ].replace("\\", "")
                    decision = ClassificationResponse(
                        **json.loads(out_text, strict=False)
                    )
                    classification_results.append(
                        {
                            "turn_segment": turn_segment,
                            "reasoning": decision.reasoning,
                            "class": decision.stage.value,
                            "confidence": decision.confidence,
                        }
                    )
                except Exception as e:
                    failed.append([prompt, turn_segment])
                    logger.warning(f"Failed to parse classification response: Retrying next")
                    continue

        logger.info(f"Classification completed with {len(failed)} failures out of {len(classification_prompts)} samples.")
        retry_prompts = [msg for msg, _ in failed]
        responses_retry = self.classifier_model.run_batch(retry_prompts, self.sampling_params_classifier)
        for (prompt, turn_segment), response in zip(failed, responses_retry):
            for output in response.outputs:
                try:
                    out_text = output.text[
                        output.text.find("{") : output.text.rfind("}") + 1
                    ].replace("\\", "")
                    decision = ClassificationResponse(
                        **json.loads(out_text, strict=False)
                    )
                    classification_results.append(
                        {
                            "turn_segment": turn_segment,
                            "reasoning": decision.reasoning,
                            "class": decision.stage.value,
                            "confidence": decision.confidence,
                        }
                    )
                except Exception as e:
                    logger.error(f"Failed to parse classification response on retry: {e}. Skipping this sample.")
                    continue
        
        self.classification_results_set.append(classification_results)

        return classification_results

    def to_pd_latest(self):
        return pd.DataFrame(self.classification_results_set[-1])