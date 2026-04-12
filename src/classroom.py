#####################################################################
# Main Classroom Logic Class. Here is where the rollouts are created.
#####################################################################

from functools import lru_cache
import re
import gc
import torch
import time
import json
import pandas as pd
from tqdm import tqdm
from enum import Enum
from typing import Any, Dict, List, Optional
from random import choice
from jinja2 import Template
from pydantic import BaseModel
from transformers import AutoTokenizer
from vllm import SamplingParams
from config.train_rl_model import (
    StudentModelConfig,
    TeacherModelConfig,
    JudgeModelConfig,
    RewardModelConfig,
    GenerationConfig,
)
from src.vllm.data_parallel_vllm import ParallelvLLMInference, InferenceTask
from src.utils.utils import check_equal, extract_answer
from src.inference_providers.open_router_inference import OpenRouterInference
from src.inference_providers.gemini_api_inference import GeminiInference
import logging

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    START = 0
    TEACHER_TURN = 1
    STUDENT_TURN = 2
    JUDGE_TURN = 4
    GENERATE_SOLUTION = 5
    REWARD_TURN = 6
    END = 7


class ConversationType(Enum):
    GUIDED = 0
    ATTEMPTED = 1


class JudgeDecision(Enum):
    OK = "OK"
    REJECT = "REJECT"


class JudgeResponse(BaseModel):
    reasoning: str
    decision: JudgeDecision


@lru_cache(maxsize=1000)
def read_template(path: str) -> Template:
    return Template(open(path).read())


@lru_cache(maxsize=1)
def get_tokenizer(tokenizer_to_use: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(tokenizer_to_use)


class Conversation:
    def __init__(
        self,
        problem_idx: int,
        problem: str,
        answer: str,
        solve_rate: float,
        generation_cfg: GenerationConfig,
        forced_type: Optional[ConversationType] = None,
        forced_student_name: Optional[str] = None,
    ):
        self.problem = problem
        self.problem_idx = problem_idx
        self.answer = answer
        self.generation_cfg = generation_cfg
        self.conversation: List[dict] = []
        self.state = ConversationState.START
        self.solve_rate = solve_rate
        self.conversation_id = ""

        self.teacher_turns = 0
        self.student_turns = 0

        problem_hash = hash(problem)
        self.type: ConversationType = (
            ConversationType.ATTEMPTED if forced_type is None else forced_type
        )
        self.student_name = (
            generation_cfg.student_names[problem_hash % len(generation_cfg.student_names)]
            if forced_student_name is None
            else forced_student_name
        )

        self.student_persona = list(
            generation_cfg.student_personas_prompts_paths.keys()
        )[
            problem_hash
            % len(list(generation_cfg.student_personas_prompts_paths.keys()))
        ]
        self.system_prompt_student = read_template(
            generation_cfg.student_personas_prompts_paths[self.student_persona]
        ).render(student_name=self.student_name, problem=problem)
        self.system_prompt_teacher = read_template(
            generation_cfg.teacher_prompt_path
        ).render(
            student_name=self.student_name,
            problem=problem,
            include_thinking=generation_cfg.use_thinking,
        )
        self.system_prompt_student_attempt = read_template(
            generation_cfg.student_initial_attempt_prompt_path
        ).render(problem=problem)
        self.initial_attempt_wrapper = read_template(
            generation_cfg.initial_attempt_wrapper_prompt_path
        )
        self.student_final_prompt = read_template(
            generation_cfg.student_final_prompt_path
        ).render()
        self.student_attempt = read_template(
            generation_cfg.student_attempt_prompt_path
        ).render(problem=problem)

        self.judge_decisions: Dict[str, list[JudgeResponse]] = {}
        self.turn_judge_decisions: Dict[int, Dict[str, List[JudgeResponse]]] = {}
        self.turn_pedagogical_alignment: Dict[int, float] = {}
        self.turn_rewards: Dict[int, Optional[float]] = {}

        self.total_teacher_turns = 0
        self.participating_teacher_turns = 0
        self.first_reject_turn: Optional[int] = None
        self.cutoff_message_index: Optional[int] = None

        self.solutions: list[str] = []
        self.rewards: list[float] = []
        self.aggregated_turn_reward: Optional[float] = None

        self.tokenizer = get_tokenizer(generation_cfg.tokenizer_to_use)

        self.initial_attempts = []
        self.initial_rewards = []

        self.failed_judges = False

        if (
            "teacher_message" in open(generation_cfg.teacher_prompt_path).read()
            and "teacher_message" in open(generation_cfg.teacher_prompt_path).read()
        ):
            self.system_prompt_teacher = read_template(
                generation_cfg.teacher_prompt_path
            ).render()
            start_user_message = read_template(
                generation_cfg.teacher_prompt_path
            ).render(problem=problem, user_message=True)
            teacher_start_message = read_template(
                generation_cfg.teacher_prompt_path
            ).render(teacher_message=True)
            self.conversation.append({"role": "student", "content": start_user_message})
            self.conversation.append(
                {"role": "teacher", "content": teacher_start_message}
            )
            self.teacher_turns = 1
            self.state = ConversationState.STUDENT_TURN

    @classmethod
    def from_dataframe(
        cls, row: Any, generation_cfg: GenerationConfig
    ) -> "Conversation":
        answer = row.get("Answer", "")

        forced_type = None
        type_val = row.get("Type")
        if isinstance(type_val, str) and type_val in ConversationType.__members__:
            forced_type = ConversationType[type_val]

        instance = cls(
            problem_idx=row.get("Problem Idx", -1),
            problem=row["Problem"],
            answer=answer,
            solve_rate=row.get("Solve Rate", 0.0),
            generation_cfg=generation_cfg,
            forced_type=forced_type,
            forced_student_name=row.get("Student Name"),
        )

        conv_data = row.get("Conversation", [])
        if isinstance(conv_data, str):
            try:
                conv_data = eval(conv_data)
            except Exception as e:
                raise ValueError(f"Failed to load 'Conversation' field: {e}")
        instance.conversation = conv_data

        state_val = row.get("State")
        if isinstance(state_val, str) and state_val in ConversationState.__members__:
            instance.state = ConversationState[state_val]

        instance.student_persona = row.get("Student Persona", instance.student_persona)

        jd_data = row.get("Judge Decisions", {})
        if isinstance(jd_data, str):
            try:
                jd_data = eval(jd_data)
            except Exception as e:
                raise ValueError(f"Failed to load 'Judge Decisions': {e}")

        converted = {}
        for key, decisions in jd_data.items():
            if isinstance(decisions, str):
                try:
                    decisions = eval(decisions)
                except Exception as e:
                    raise ValueError(
                        f"Failed to load judge decisions for key {key}: {e}"
                    )
            converted[key] = [
                JudgeResponse(
                    reasoning=d["reasoning"], decision=JudgeDecision[d["decision"]]
                )
                for d in decisions
            ]
        instance.judge_decisions = converted

        solutions = row.get("Solutions", [])
        if isinstance(solutions, str):
            try:
                solutions = eval(solutions)
            except Exception as e:
                raise ValueError(f"Failed to load 'Solutions': {e}")
        instance.solutions = solutions

        rewards = row.get("Rewards", [])
        if isinstance(rewards, str):
            try:
                rewards = eval(rewards)
            except Exception as e:
                raise ValueError(f"Failed to load 'Rewards': {e}")
        instance.rewards = rewards

        initial_attempts = row.get("Initial Attempts", [])
        if isinstance(initial_attempts, str):
            try:
                initial_attempts = eval(initial_attempts)
            except Exception as e:
                raise ValueError(f"Failed to load 'Initial Attempts': {e}")
        instance.initial_attempts = initial_attempts

        initial_rewards = row.get("Initial Rewards", [])
        if isinstance(initial_rewards, str):
            try:
                initial_rewards = eval(initial_rewards)
            except Exception as e:
                raise ValueError(f"Failed to load 'Initial Rewards': {e}")
        instance.initial_rewards = initial_rewards

        return instance

    def get_student_no_tutor_attempt(self):
        return [{"role": "user", "content": self.student_attempt}]

    def start_conversation(self):
        if self.state != ConversationState.START:
            return

        if self.type == ConversationType.GUIDED:
            self.state = ConversationState.TEACHER_TURN
        else:
            self.state = ConversationState.STUDENT_TURN

    def _exceeded_max_tokens(self):
        return (
            sum(
                [
                    len(self.tokenizer.encode(message["content"]))
                    for message in self.conversation
                ]
            )
            > self.generation_cfg.max_tokens_in_conversation
        )

    def _hide_thinking(self, content: str):
        return re.sub(r"<think>.*?</think>", "", content, flags=re.S).replace(
            "<end_of_conversation>", ""
        )

    def _hide_messages(self, messages: List[dict]):
        return [
            {
                "role": message["role"],
                "content": self._hide_thinking(message["content"]),
            }
            for message in messages
        ]

    def _get_hidden_conversation(self):
        return self._hide_messages(self.conversation)

    def _get_conversation_from_teacher_perspective(self, messages: Optional[List[dict]] = None):
        if messages is None:
            messages = self.conversation
        conversation = []
        for message in messages:
            if message["role"] == "teacher":
                conversation.append({"role": "assistant", "content": message["content"]})
            else:
                conversation.append({"role": "user", "content": message["content"]})
        return conversation

    def _get_conversation_from_student_perspective(self):
        conversation = []
        for message in self.conversation:
            if message["role"] == "student":
                conversation.append(
                    {
                        "role": "assistant",
                        "content": self._hide_thinking(message["content"]),
                    }
                )
            else:
                conversation.append(
                    {"role": "user", "content": self._hide_thinking(message["content"])}
                )
        return conversation

    def _get_teacher_message_indices(self) -> List[int]:
        return [
            idx for idx, message in enumerate(self.conversation)
            if message["role"] == "teacher"
        ]

    def _get_hidden_conversation_until_turn(self, teacher_turn_idx: int, context_turns: int):
        """
        Getting part of the conversation for n turns.
        """
        teacher_indices = self._get_teacher_message_indices()
        if teacher_turn_idx >= len(teacher_indices):
            return []

        end_idx = teacher_indices[teacher_turn_idx]
        sub_messages = self.conversation[: end_idx + 1]

        if context_turns is None or context_turns <= 0:
            return self._hide_messages(sub_messages)

        local_teacher_indices = [
            idx for idx, message in enumerate(sub_messages)
            if message["role"] == "teacher"
        ]
        if len(local_teacher_indices) <= context_turns:
            return self._hide_messages(sub_messages)

        start_teacher_local_idx = local_teacher_indices[-context_turns]
        start_idx = start_teacher_local_idx

        while start_idx > 0 and sub_messages[start_idx - 1]["role"] == "student":
            start_idx -= 1

        return self._hide_messages(sub_messages[start_idx:])

    def get_turn_judge_prompt(self, rule_path: str, teacher_turn_idx: int, context_turns: int):
        conversation = self._get_hidden_conversation_until_turn(
            teacher_turn_idx, context_turns
        )
        return [
            {
                "role": "user",
                "content": Template(open(rule_path).read()).render(
                    conversation=conversation,
                    n_turns=(
                        min(context_turns, teacher_turn_idx + 1)
                        if context_turns and context_turns > 0
                        else teacher_turn_idx + 1
                    ),
                ),
            }
        ]

    def get_conversation(self):
        if self.state == ConversationState.TEACHER_TURN:
            return [
                {"role": "system", "content": self.system_prompt_teacher}
            ] + self._get_conversation_from_teacher_perspective()

        if self.state == ConversationState.STUDENT_TURN:
            if self.type == ConversationType.ATTEMPTED and len(self.conversation) == 0:
                return [{"role": "system", "content": self.system_prompt_student_attempt}]
            return [
                {"role": "system", "content": self.system_prompt_student}
            ] + self._get_conversation_from_student_perspective()

        if self.state == ConversationState.GENERATE_SOLUTION:
            conversation = [
                {"role": "system", "content": self.system_prompt_student}
            ] + self._get_conversation_from_student_perspective()
            conversation.append({"role": "user", "content": self.student_final_prompt})
            return conversation

        return []

    def add_message(self, content: str):
        if self.state == ConversationState.TEACHER_TURN:
            self.conversation.append({"role": "teacher", "content": content})
            self.teacher_turns += 1
            self.state = ConversationState.STUDENT_TURN
            if (
                len(self.conversation) >= self.generation_cfg.max_turns
                or "<end_of_conversation>" in content
            ):
                self.state = ConversationState.JUDGE_TURN

        elif self.state == ConversationState.STUDENT_TURN:
            if self.type == ConversationType.ATTEMPTED and len(self.conversation) == 0:
                self.conversation.append(
                    {
                        "role": "student",
                        "content": self.initial_attempt_wrapper.render(attempt=content),
                    }
                )
                self.student_turns += 1
                self.state = ConversationState.TEACHER_TURN
            else:
                self.conversation.append({"role": "student", "content": content})
                self.student_turns += 1
                self.state = ConversationState.TEACHER_TURN

        if self._exceeded_max_tokens():
            self.state = ConversationState.JUDGE_TURN

        if (
            self.generation_cfg.number_judge_attempts == 0
            and self.state == ConversationState.JUDGE_TURN
        ):
            self.state = ConversationState.GENERATE_SOLUTION

    def add_turn_judge_decisions(
        self, teacher_turn_idx: int, rule_name: str, decisions: List[JudgeResponse]
    ):
        self.turn_judge_decisions.setdefault(teacher_turn_idx, {})
        self.turn_judge_decisions[teacher_turn_idx][rule_name] = decisions
        self.judge_decisions.setdefault(rule_name, []).extend(decisions)

    def finalize_turn_judges(self):
        teacher_indices = self._get_teacher_message_indices()
        self.total_teacher_turns = len(teacher_indices)

        if self.total_teacher_turns == 0:
            self.participating_teacher_turns = 0
            self.first_reject_turn = None
            self.cutoff_message_index = None
            self.failed_judges = False
            return

        ped_by_turn = {}
        for turn_idx in range(self.total_teacher_turns):
            decisions_by_rule = self.turn_judge_decisions.get(turn_idx, {})
            if not decisions_by_rule:
                # Default to pedagogically aligned if no judge decisions are present for this turn.
                ped_by_turn[turn_idx] = 1.0
                continue

            is_ok = True
            for decisions in decisions_by_rule.values():
                if any(d.decision == JudgeDecision.REJECT for d in decisions):
                    is_ok = False
                    break
            ped_by_turn[turn_idx] = 1.0 if is_ok else 0.0

        self.turn_pedagogical_alignment = ped_by_turn

        reject_cutoff_enabled = self.generation_cfg.reject_cutoff_enabled
        self.first_reject_turn = None
        if reject_cutoff_enabled:
            for turn_idx in range(self.total_teacher_turns):
                if self.turn_pedagogical_alignment.get(turn_idx, 1.0) == 0.0:
                    self.first_reject_turn = turn_idx
                    break

        if self.first_reject_turn is None:
            self.participating_teacher_turns = self.total_teacher_turns
        else:
            self.participating_teacher_turns = self.first_reject_turn + 1

        self.failed_judges = self.first_reject_turn is not None

        if self.participating_teacher_turns > 0:
            self.cutoff_message_index = teacher_indices[self.participating_teacher_turns - 1]
        else:
            self.cutoff_message_index = None

    def _build_accuracy_shares(self, accuracy_reward: float, active_turns: int):
        """
        Build the accuracy shares for each turn based on the configured mode.
        """
        if active_turns <= 0:
            return []

        mode = self.generation_cfg.accuracy_share_mode
        if mode == "discount":
            gamma = self.generation_cfg.accuracy_share_discount_gamma
            if gamma <= 0:
                weights = [1.0 for _ in range(active_turns)]
            else:
                weights = [gamma ** (active_turns - 1 - idx) for idx in range(active_turns)]
        elif mode == "uniform":
            weights = [1.0 for _ in range(active_turns)]
        else:
            logger.warning(f"Unknown accuracy share mode '{mode}', defaulting to uniform.")
            weights = [1.0 for _ in range(active_turns)]

        denominator = sum(weights)
        if denominator <= 0:
            return [accuracy_reward / active_turns for _ in range(active_turns)]
        return [accuracy_reward * w / denominator for w in weights]

    def compute_turn_rewards(self, accuracy_reward: Optional[float]):
        self.turn_rewards = {}
        self.aggregated_turn_reward = None

        if accuracy_reward is None:
            return

        if self.total_teacher_turns == 0:
            self.aggregated_turn_reward = accuracy_reward
            return

        active_turns = self.participating_teacher_turns
        if active_turns <= 0:
            self.aggregated_turn_reward = accuracy_reward
            return

        lambda_penalty = self.generation_cfg.extra_penalty_for_rejected_judges
        pre_reject_weight = self.generation_cfg.pre_reject_weight_w

        accuracy_shares = self._build_accuracy_shares(accuracy_reward, active_turns)
        valid_turn_rewards = []

        for turn_idx in range(self.total_teacher_turns):
            if turn_idx >= active_turns:
                self.turn_rewards[turn_idx] = None
                continue

            ped_t = self.turn_pedagogical_alignment.get(turn_idx, 1.0)
            base_reward = accuracy_shares[turn_idx] - lambda_penalty * (1.0 - ped_t)

            if self.first_reject_turn is not None and turn_idx < self.first_reject_turn:
                scale = pre_reject_weight
            else:
                scale = 1.0

            # Pre-REJECT W are multiplied by a factor to give them less credit, but they can still get some reward if they are pedagogically aligned.
            reward = scale * base_reward
            self.turn_rewards[turn_idx] = reward
            valid_turn_rewards.append(reward)

        if valid_turn_rewards:
            self.aggregated_turn_reward = sum(valid_turn_rewards) / len(valid_turn_rewards)
        else:
            self.aggregated_turn_reward = accuracy_reward

    def add_solutions(self, solutions: List[str]):
        if self.state != ConversationState.GENERATE_SOLUTION:
            raise ValueError("We are not in the generate solution state")
        self.solutions = solutions
        self.state = ConversationState.REWARD_TURN

    def get_solutions_for_reward(self):
        return [
            self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "Please reason step by step, and put your final answer within \\boxed{}.",
                    },
                    {"role": "user", "content": self.problem},
                    {"role": "assistant", "content": solution},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            for solution in self.solutions
        ]

    def get_initial_solutions_for_reward(self):
        return [
            self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "Please reason step by step, and put your final answer within \\boxed{}.",
                    },
                    {"role": "user", "content": self.problem},
                    {"role": "assistant", "content": solution},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            for solution in self.initial_attempts
        ]

    def add_initial_rewards(self, rewards: List[float]):
        self.initial_rewards = rewards

    def add_rewards(self, rewards: List[float]):
        if self.state != ConversationState.REWARD_TURN:
            raise ValueError("We are not in the reward turn state")
        self.rewards = rewards
        self.state = ConversationState.END

    def add_initial_attempts(self, attempts: List[str]):
        self.initial_attempts = attempts

    def get_judge_ok_rate(self):
        all_decisions = []
        for decisions in self.judge_decisions.values():
            all_decisions.extend(decisions)
        if not all_decisions:
            return None
        ok_count = sum(1 for d in all_decisions if d.decision == JudgeDecision.OK)
        return ok_count / len(all_decisions)

    def get_end_rm_reward(self):
        return sum(self.rewards) / len(self.rewards) if len(self.rewards) > 0 else None

    def get_turn_aggregated_reward(self):
        return self.aggregated_turn_reward

    def _get_active_messages_for_training(self):
        if (
            self.generation_cfg.reject_cutoff_enabled
            and self.cutoff_message_index is not None
        ):
            return self.conversation[: self.cutoff_message_index + 1]
        return self.conversation

    def get_initial_rm_reward(self):
        return (
            sum(self.initial_rewards) / len(self.initial_rewards)
            if len(self.initial_rewards) > 0
            else None
        )

    def get_thinking_reward(self):
        if len(self.rewards) == 0:
            return 0.0
        penalty_for_missing_closing_think = 0.0
        count_used_thinking, count_total = 0, 0
        for message in self._get_active_messages_for_training():
            if message["role"] == "teacher":
                if message["content"].count("<think>") != message["content"].count(
                    "</think>"
                ):
                    penalty_for_missing_closing_think -= 0.5
                elif message["content"].count("<think>") > 0:
                    count_used_thinking += 1
                count_total += 1
        if count_total == 0:
            return 0.0
        return (
            penalty_for_missing_closing_think
            + (count_used_thinking / count_total) * 0.5
        )

    def get_end_of_conversation_reward(self):
        if len(self.rewards) == 0:
            return 0.0
        return (
            0.1
            if any(
                "<end_of_conversation>" in message["content"]
                for message in self._get_active_messages_for_training()
            )
            else 0.0
        )

    def get_length_reward(self):
        texts = []
        for message in self._get_active_messages_for_training():
            if message["role"] == "teacher":
                texts.append(message["content"])

        text_tokens_count = [len(self.tokenizer.encode(t)) for t in texts]

        return (
            -0.5
            if any(
                [
                    t >= self.generation_cfg.max_tokens_per_turn - 1
                    for t in text_tokens_count
                ]
            )
            else 0.0
        )

    def get_logging_stats(self):
        active_turn_rewards = [
            reward for idx, reward in sorted(self.turn_rewards.items())
            if reward is not None
        ]
        active_ped = [
            self.turn_pedagogical_alignment.get(idx, 1.0)
            for idx in range(self.participating_teacher_turns)
        ]

        return {
            "total_teacher_turns": self.total_teacher_turns,
            "participating_teacher_turns": self.participating_teacher_turns,
            "first_reject_turn": self.first_reject_turn,
            "mean_turn_reward": (
                sum(active_turn_rewards) / len(active_turn_rewards)
                if len(active_turn_rewards) > 0
                else 0.0
            ),
            "mean_pedagogical_reward": (
                sum(active_ped) / len(active_ped)
                if len(active_ped) > 0
                else 0.0
            ),
        }

    def to_pd(self):
        return pd.DataFrame(
            [
                {
                    "State": self.state.name,
                    "Problem": self.problem,
                    "Problem Idx": self.problem_idx,
                    "Answer": self.answer,
                    "Conversation": self.conversation,
                    "Type": self.type.name,
                    "Student Persona": self.student_persona,
                    "Student Name": self.student_name,
                    "Judge Decisions": {
                        key: [
                            {
                                "reasoning": decision.reasoning,
                                "decision": decision.decision.name,
                            }
                            for decision in decisions
                        ]
                        for key, decisions in self.judge_decisions.items()
                    },
                    "Turn Pedagogical": self.turn_pedagogical_alignment,
                    "Turn Rewards": self.turn_rewards,
                    "First Reject Turn": self.first_reject_turn,
                    "Participating Teacher Turns": self.participating_teacher_turns,
                    "Solutions": self.solutions,
                    "Rewards": self.rewards,
                    "Aggregated Turn Reward": self.aggregated_turn_reward,
                    "Initial Attempts": self.initial_attempts,
                    "Initial Rewards": self.initial_rewards,
                    "Conversation from student perspective": self._get_conversation_from_student_perspective(),
                }
            ]
        )

    def __str__(self):
        return self.to_pd().to_string()

    def __repr__(self):
        return self.to_pd().to_string()

    def get_trainable_representation(self):
        messages = self._get_active_messages_for_training()

        return [{"role": "system", "content": self.system_prompt_teacher}] + self._get_conversation_from_teacher_perspective(messages)


class Classroom:
    def __init__(
        self,
        student_model_cfg: StudentModelConfig,
        teacher_model_cfg: TeacherModelConfig,
        judge_model_cfg: JudgeModelConfig,
        reward_model_cfg: RewardModelConfig,
        generation_cfg: GenerationConfig,
        model_save_path: str,
        log_file_path: Optional[str] = None,
    ):
        self.student_model_cfg = student_model_cfg
        self.teacher_model_cfg = teacher_model_cfg
        self.judge_model_cfg = judge_model_cfg
        self.reward_model_cfg = reward_model_cfg
        self.generation_cfg = generation_cfg

        if self.teacher_model_cfg.use_openrouter:
            self.teacher_model = OpenRouterInference(
                self.teacher_model_cfg.model_name_or_path
            )
        elif self.teacher_model_cfg.use_gemini:
            self.teacher_model = GeminiInference(
                self.teacher_model_cfg.model_name_or_path
            )
        else:
            self.teacher_model = ParallelvLLMInference(
                model_path=teacher_model_cfg.model_name_or_path,
                gpus_per_instance=teacher_model_cfg.vllm.number_of_gpus_per_instance,
                gpu_memory_utilization=teacher_model_cfg.vllm.gpu_memory_utilization,
                max_model_len=teacher_model_cfg.vllm.max_length,
                max_num_seqs=teacher_model_cfg.vllm.max_num_seqs,
                model_save_path=model_save_path,
                use_lora=teacher_model_cfg.lora.enable,
                load_and_unload=teacher_model_cfg.vllm.load_and_unload,
                max_number_of_instances=teacher_model_cfg.vllm.max_number_of_instances,
                enable_sleep_mode=teacher_model_cfg.vllm.enable_sleep_mode,
                bits_and_bytes=teacher_model_cfg.vllm.bits_and_bytes,
                use_awq=teacher_model_cfg.vllm.use_awq,
                from_0=teacher_model_cfg.vllm.from_0,
                use_v0=teacher_model_cfg.vllm.use_v0,
                enforce_eager=teacher_model_cfg.vllm.enforce_eager,
                logging_enabled=log_file_path is not None,
                log_file_path=log_file_path,
            )
        self.teacher_model.sleep()

        if self.student_model_cfg.use_openrouter:
            self.student_model = OpenRouterInference(
                self.student_model_cfg.model_name_or_path
            )
        elif self.student_model_cfg.use_gemini:
            self.student_model = GeminiInference(
                self.student_model_cfg.model_name_or_path
            )
        else:
            self.student_model = ParallelvLLMInference(
                model_path=student_model_cfg.model_name_or_path,
                gpus_per_instance=student_model_cfg.vllm.number_of_gpus_per_instance,
                gpu_memory_utilization=student_model_cfg.vllm.gpu_memory_utilization,
                max_model_len=student_model_cfg.vllm.max_length,
                max_num_seqs=student_model_cfg.vllm.max_num_seqs,
                model_save_path=None,
                load_and_unload=student_model_cfg.vllm.load_and_unload,
                max_number_of_instances=student_model_cfg.vllm.max_number_of_instances,
                bits_and_bytes=student_model_cfg.vllm.bits_and_bytes,
                use_awq=student_model_cfg.vllm.use_awq,
                enable_sleep_mode=student_model_cfg.vllm.enable_sleep_mode,
                from_0=student_model_cfg.vllm.from_0,
                use_v0=student_model_cfg.vllm.use_v0,
                enforce_eager=student_model_cfg.vllm.enforce_eager,
                logging_enabled=log_file_path is not None,
                log_file_path=log_file_path,
            )
        self.student_model.sleep()

        if self.judge_model_cfg.use_openrouter:
            self.judge_model = OpenRouterInference(
                self.judge_model_cfg.model_name_or_path
            )
        elif self.judge_model_cfg.use_gemini:
            self.judge_model = GeminiInference(self.judge_model_cfg.model_name_or_path)
        else:
            self.judge_model = ParallelvLLMInference(
                model_path=judge_model_cfg.model_name_or_path,
                gpus_per_instance=judge_model_cfg.vllm.number_of_gpus_per_instance,
                gpu_memory_utilization=judge_model_cfg.vllm.gpu_memory_utilization,
                max_model_len=judge_model_cfg.vllm.max_length,
                max_num_seqs=judge_model_cfg.vllm.max_num_seqs,
                model_save_path=None,
                load_and_unload=judge_model_cfg.vllm.load_and_unload,
                max_number_of_instances=judge_model_cfg.vllm.max_number_of_instances,
                bits_and_bytes=judge_model_cfg.vllm.bits_and_bytes,
                use_awq=judge_model_cfg.vllm.use_awq,
                enable_sleep_mode=judge_model_cfg.vllm.enable_sleep_mode,
                enforce_eager=judge_model_cfg.vllm.enforce_eager,
                from_0=judge_model_cfg.vllm.from_0,
                use_v0=judge_model_cfg.vllm.use_v0,
                logging_enabled=log_file_path is not None,
                log_file_path=log_file_path,
            )
        self.judge_model.sleep()

        if self.reward_model_cfg.model_name_or_path not in ["None", "Answer"]:
            self.reward_model = ParallelvLLMInference(
                model_path=reward_model_cfg.model_name_or_path,
                gpus_per_instance=reward_model_cfg.vllm.number_of_gpus_per_instance,
                gpu_memory_utilization=reward_model_cfg.vllm.gpu_memory_utilization,
                max_model_len=reward_model_cfg.vllm.max_length,
                max_num_seqs=reward_model_cfg.vllm.max_num_seqs,
                model_save_path=None,
                load_and_unload=reward_model_cfg.vllm.load_and_unload,
                max_number_of_instances=reward_model_cfg.vllm.max_number_of_instances,
                bits_and_bytes=reward_model_cfg.vllm.bits_and_bytes,
                use_awq=reward_model_cfg.vllm.use_awq,
                inference_task=InferenceTask.REWARD,
                enable_sleep_mode=reward_model_cfg.vllm.enable_sleep_mode,
                enforce_eager=reward_model_cfg.vllm.enforce_eager,
                from_0=reward_model_cfg.vllm.from_0,
                use_v0=reward_model_cfg.vllm.use_v0,
                logging_enabled=log_file_path is not None,
                log_file_path=log_file_path,
            )
            self.reward_model.sleep()

        self.sampling_params_student = SamplingParams(
            temperature=student_model_cfg.vllm.temperature,
            top_k=student_model_cfg.vllm.top_k,
            top_p=student_model_cfg.vllm.top_p,
            max_tokens=generation_cfg.max_tokens_per_turn,
        )

        self.sampling_params_judge = SamplingParams(
            temperature=judge_model_cfg.vllm.temperature,
            top_k=judge_model_cfg.vllm.top_k,
            top_p=judge_model_cfg.vllm.top_p,
            max_tokens=generation_cfg.max_tokens_per_judge_attempt,
        )

        self.sampling_params_student_solution = SamplingParams(
            n=generation_cfg.number_student_attempts,
            temperature=student_model_cfg.vllm.temperature,
            top_k=student_model_cfg.vllm.top_k,
            top_p=student_model_cfg.vllm.top_p,
            max_tokens=generation_cfg.max_tokens_per_student_attempt,
        )

        teacher_tokenizer = AutoTokenizer.from_pretrained(
            generation_cfg.tokenizer_to_use
        )
        thinking_tokens = teacher_tokenizer.encode("<think>", add_special_tokens=False)

        def force_thinking_processor(token_ids, logits):
            if len(token_ids) < len(thinking_tokens):
                logits[thinking_tokens[len(token_ids)]] = 10000
            return logits

        self.sampling_params_teacher = SamplingParams(
            temperature=teacher_model_cfg.vllm.temperature,
            top_k=teacher_model_cfg.vllm.top_k,
            top_p=teacher_model_cfg.vllm.top_p,
            max_tokens=generation_cfg.max_tokens_per_turn,
            logits_processors=(
                [force_thinking_processor] if generation_cfg.force_thinking else []
            ),
        )

        self.conversation_sets = []

    def _compute_rewards_from_prompts(
        self, prompts: List[str], answers: List[str]
    ) -> List[float]:
        if self.reward_model_cfg.model_name_or_path not in ["None", "Answer"]:
            responses = self.reward_model.run_batch(prompts, None)
            rewards = [
                output.outputs.data[-1].item() if hasattr(output, "outputs") else 1.0
                for output in responses
            ]
        elif self.reward_model_cfg.model_name_or_path == "Answer":
            extracted_answers = [extract_answer(prompt) for prompt in prompts]
            rewards = [
                1.0 if check_equal(answer, extracted_answer) else 0.0
                for answer, extracted_answer in zip(answers, extracted_answers)
            ]
        else:
            rewards = [0.0 for _ in prompts]
        return rewards

    def generate_next_teacher_utterances(
        self, conversations: List[Conversation], meta: Optional[dict] = None
    ) -> List[str]:
        if meta is None:
            meta = {}
        prompts = [conv.get_conversation() for conv in conversations]
        responses = self.teacher_model.run_batch(
            prompts, self.sampling_params_teacher, meta
        )
        teacher_utterances = [response.outputs[0].text for response in responses]
        for conv, utterance in zip(conversations, teacher_utterances):
            conv.add_message(utterance)
        return teacher_utterances

    def generate_next_student_utterances(
        self, conversations: List[Conversation]
    ) -> List[str]:
        prompts = [conv.get_conversation() for conv in conversations]
        responses = self.student_model.run_batch(prompts, self.sampling_params_student)
        student_utterances = [response.outputs[0].text for response in responses]
        for conv, utterance in zip(conversations, student_utterances):
            conv.add_message(utterance)
        return student_utterances

    def _parse_judge_response(self, text: str) -> Optional[JudgeResponse]:
        try:
            out_text = text[text.find("{") : text.rfind("}") + 1].replace("\\", "")
            return JudgeResponse(**json.loads(out_text, strict=False))
        except Exception:
            return None

    def run_turn_judges(self, conversations: List[Conversation]):
        logger.info(("=" * 10) + "Running turn-wise judges" + ("=" * 10))
        start_time = time.time()

        conversations_to_process = [
            conv for conv in conversations if conv.state == ConversationState.JUDGE_TURN
        ]
        if not conversations_to_process:
            logger.info("No conversations in JUDGE_TURN")
            return

        rule_paths = dict(self.generation_cfg.judges_rules_prompts_paths)
        num_attempts_required = max(1, self.generation_cfg.number_judge_attempts)
        context_turns = self.generation_cfg.turn_judge_context_turns

        if self.generation_cfg.number_judge_attempts == 0 or len(rule_paths) == 0:
            for conv in conversations_to_process:
                conv.finalize_turn_judges()
                conv.state = ConversationState.GENERATE_SOLUTION
            logger.info("Skipped judges due to config")
            return

        max_retry = 5

        for rule_name, rule_path in rule_paths.items():
            tasks = []
            for conv in conversations_to_process:
                teacher_turn_count = len(conv._get_teacher_message_indices())
                for turn_idx in range(teacher_turn_count):
                    prompt = conv.get_turn_judge_prompt(
                        rule_path=rule_path,
                        teacher_turn_idx=turn_idx,
                        context_turns=context_turns,
                    )
                    tasks.append((conv, turn_idx, prompt))

            if not tasks:
                continue

            valid_responses: Dict[int, List[JudgeResponse]] = {
                idx: [] for idx in range(len(tasks))
            }

            pending_task_indices = list(range(len(tasks)))
            retry_round = 0
            while pending_task_indices and retry_round <= max_retry:
                pending_messages = [tasks[idx][2] for idx in pending_task_indices]
                responses = self.judge_model.run_batch(
                    pending_messages, self.sampling_params_judge
                )

                next_pending = []
                for local_idx, task_idx in enumerate(pending_task_indices):
                    response = responses[local_idx]
                    parsed = None
                    for output in response.outputs:
                        parsed = self._parse_judge_response(output.text)
                        if parsed is not None:
                            valid_responses[task_idx].append(parsed)
                            break

                    if len(valid_responses[task_idx]) < num_attempts_required:
                        next_pending.append(task_idx)

                pending_task_indices = next_pending
                retry_round += 1

            for task_idx, (conv, turn_idx, _) in enumerate(tasks):
                decisions = valid_responses[task_idx]
                while len(decisions) < num_attempts_required:
                    decisions.append(
                        JudgeResponse(
                            reasoning="invalid judge response",
                            decision=JudgeDecision.REJECT,
                        )
                    )
                conv.add_turn_judge_decisions(turn_idx, rule_name, decisions)

        for conv in conversations_to_process:
            conv.finalize_turn_judges()
            conv.state = ConversationState.GENERATE_SOLUTION

        self.judge_model.sleep()
        logger.info(f"Took {time.time() - start_time} seconds.")

    def sample_conversations(
        self,
        problems: List[str],
        answers: List[str],
        forced_type: Optional[ConversationType] = None,
        meta: dict = {},
        compute_initial_attempt: bool = False,
    ) -> List[Conversation]:
        if forced_type is None:
            if self.generation_cfg.forced_conversation_type == "guided":
                forced_type = ConversationType.GUIDED
            elif self.generation_cfg.forced_conversation_type == "attempt":
                forced_type = ConversationType.ATTEMPTED
            else:
                forced_type = None

        conversations = []
        for idx, (problem, answer) in enumerate(
            tqdm(
                zip(problems, answers),
                total=len(problems),
                desc="Initializing conversations",
            )
        ):
            conversations.append(
                Conversation(
                    problem_idx=idx,
                    problem=problem,
                    answer=answer,
                    solve_rate=0.0,
                    generation_cfg=self.generation_cfg,
                    forced_type=forced_type,
                )
            )

        for conversation in conversations:
            conversation.start_conversation()

        if compute_initial_attempt:
            logger.info(("=" * 10) + "Computing initial attempts" + ("=" * 10))
            messages = [
                conversation.get_student_no_tutor_attempt()
                for conversation in conversations
            ]
            responses = self.student_model.run_batch(
                messages, self.sampling_params_student_solution
            )
            for conversation, response in zip(conversations, responses):
                conversation.add_initial_attempts(
                    [output.text for output in response.outputs]
                )

            prompts_for_rewards = [
                conversation.get_initial_solutions_for_reward()
                for conversation in conversations
            ]
            lengths = [len(prompts) for prompts in prompts_for_rewards]

            all_prompts = [
                prompt for prompts in prompts_for_rewards for prompt in prompts
            ]
            all_answers = []
            for conversation in conversations:
                all_answers.extend(
                    [conversation.answer] * len(conversation.initial_attempts)
                )

            rewards = self._compute_rewards_from_prompts(all_prompts, all_answers)

            for conv in conversations:
                curr_len = lengths.pop(0)
                conv_rewards = rewards[:curr_len]
                conv.add_initial_rewards(conv_rewards)
                rewards = rewards[curr_len:]

        round_counter = 1

        while any(
            [
                conversation.state
                in [ConversationState.TEACHER_TURN, ConversationState.STUDENT_TURN]
                for conversation in conversations
            ]
        ):
            for state_to_process in [
                ConversationState.TEACHER_TURN,
                ConversationState.STUDENT_TURN,
            ]:
                logger.info(
                    ("=" * 10)
                    + f"Executing turn {round_counter}: {'Teacher' if state_to_process == ConversationState.TEACHER_TURN else 'Student'}"
                    + ("=" * 10)
                )

                start_time = time.time()
                conversations_to_process = [
                    conversation
                    for conversation in conversations
                    if conversation.state == state_to_process
                ]
                if len(conversations_to_process) == 0:
                    continue

                if state_to_process == ConversationState.TEACHER_TURN:
                    self.generate_next_teacher_utterances(conversations_to_process, meta)
                else:
                    self.generate_next_student_utterances(conversations_to_process)

                round_counter += 1
                logger.info(f"Took {time.time() - start_time} seconds.")

        self.teacher_model.sleep()
        self.student_model.sleep()

        self.run_turn_judges(conversations)

        logger.info(("=" * 10) + "Sampling solutions" + ("=" * 10))
        start_time = time.time()
        conversations_to_process = [
            conversation
            for conversation in conversations
            if conversation.state == ConversationState.GENERATE_SOLUTION
        ]
        logger.info(
            f"Generating solutions for {len(conversations_to_process)} conversations"
        )

        if len(conversations_to_process) > 0:
            messages = [
                conversation.get_conversation()
                for conversation in conversations_to_process
            ]
            responses = self.student_model.run_batch(
                messages, self.sampling_params_student_solution
            )
            for conversation, response in zip(conversations_to_process, responses):
                conversation.add_solutions([output.text for output in response.outputs])

        self.student_model.sleep()
        logger.info(f"Took {time.time() - start_time} seconds.")

        logger.info(("=" * 10) + "Computing Rewards" + ("=" * 10))
        start_time = time.time()
        reward_convs = [
            conv
            for conv in conversations
            if conv.state == ConversationState.REWARD_TURN
        ]
        if reward_convs:
            all_prompts = []
            all_answers = []
            lengths = []
            for conv in reward_convs:
                prompts = conv.get_solutions_for_reward()
                lengths.append(len(prompts))
                all_prompts.extend(prompts)
                all_answers.extend([conv.answer] * len(prompts))
            rewards = self._compute_rewards_from_prompts(all_prompts, all_answers)
            for conv in reward_convs:
                curr_len = lengths.pop(0)
                conv_rewards = rewards[:curr_len]
                conv.add_rewards(conv_rewards)
                conv.compute_turn_rewards(conv.get_end_rm_reward())
                rewards = rewards[curr_len:]

        logger.info(f"Took {time.time() - start_time} seconds.")
        gc.collect()
        torch.cuda.empty_cache()

        self.conversation_sets.append(conversations)
        return conversations

    def run_judges(self, conversations: List[Conversation]):
        self.run_turn_judges(conversations)

    def to_pd_latest(self):
        return pd.concat(
            [conversation.to_pd() for conversation in self.conversation_sets[-1]]
        )

    def get_conversation_by_text(self, text: str):
        conversations = self.conversation_sets[-1]
        max_messages_overlap = 0
        conversation = None
        for conv in conversations:
            trainable_representation = conv.get_trainable_representation()
            messages_overlap = sum(
                [
                    len(message["content"])
                    for message in trainable_representation
                    if message["content"] in text
                ]
            )
            if messages_overlap > max_messages_overlap:
                max_messages_overlap = messages_overlap
                conversation = conv

        if max_messages_overlap == 0:
            raise ValueError("No conversation found")
        return conversation

    def get_end_rm_reward(self, conversation: Conversation):
        accuracy = conversation.get_end_rm_reward()
        if accuracy is None:
            lambda_penalty = self.generation_cfg.extra_penalty_for_rejected_judges
            minimum_reward = -lambda_penalty
            return 0.0, minimum_reward

        total = conversation.get_turn_aggregated_reward()
        if total is None:
            total = accuracy
        return accuracy, total

    def get_thinking_reward(self, conversation: Conversation):
        return conversation.get_thinking_reward()

    def get_end_of_conversation_reward(self, conversation: Conversation):
        return conversation.get_end_of_conversation_reward()

    def get_length_reward(self, conversation: Conversation):
        return conversation.get_length_reward()
