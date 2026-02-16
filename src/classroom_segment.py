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
from typing import Dict, List, Tuple, Optional
from random import choice
from jinja2 import Template
from pydantic import BaseModel
from transformers import AutoTokenizer
from vllm import PoolingOutput, SamplingParams, RequestOutput
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

from tree_structure import ConversationTree, TreeNode, TurnPair

logger = logging.getLogger(__name__)


# Each conversation will have a small state machine to track the conversation state.
class ConversationState(Enum):
    START = 0
    TEACHER_TURN = 1
    STUDENT_TURN = 2
    JUDGE_TURN = 3
    GENERATE_SOLUTION = 4
    REWARD_TURN = 5
    END = 6

# This is the type of conversation we are having.
class ConversationType(Enum):
    GUIDED = 0
    ATTEMPTED = 1

# This is the decision the judge can make.
class JudgeDecision(Enum):
    OK = "OK"
    REJECT = "REJECT"

# This is the response from the judge. Which also includes the reasoning behind the decision.
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
        generation_cfg: GenerationConfig,
        forced_type: ConversationType = None,
        forced_student_name: str = None,
    ):
        self.problem = problem
        self.problem_idx = problem_idx
        self.answer = answer
        self.generation_cfg = generation_cfg
        self.conversation = []  # list of dicts: {role: str, content: str}
        self.state = ConversationState.START
        
        # NOTE: Tree related attributes
        self.conversation_id = ""  # Classroom에서 할당
        self.current_node_id: Optional[int] = None  # 현재 TreeNode ID
        self.parent_node_id: Optional[int] = None  # 부모 노드 ID

        # NOTE: Conversation attributes
        self.current_conversation: List[dict] = []  # 현재 그룹의 대화
        self.grouped_conversation = {}
        self.teacher_turns = 0
        self.student_turns = 0

        problem_hash = hash(problem)
        self.type: ConversationType = (
            # TODO: Since single-Turn needs student-teacher turn, I fixed to ATTEMPTED. We can add a new type for single-turn in the future if needed.
            # [ConversationType.GUIDED, ConversationType.ATTEMPTED][problem_hash % 2]
            ConversationType.ATTEMPTED
            if forced_type is None
            else forced_type
        )
        self.student_name = (
            generation_cfg.student_names[
                problem_hash % len(generation_cfg.student_names)
            ]
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

        self.judge_evaluation_type = None
        self.judge_decisions: Dict[str, list[JudgeResponse]] = {}
        self.solutions: list[str] = []
        self.rewards: list[float] = []
        self.accuracy_rewards: list[float] = []

        self.tokenizer = get_tokenizer(generation_cfg.tokenizer_to_use)

        self.initial_attempts = []
        self.initial_rewards = []

        self.failed_judges = False

        self.constructivist_failed_judges = {}

        # SocraticLM special treatment.
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
            self.state = ConversationState.STUDENT_TURN

    def copy(self):
        """Conversation 복사 (branching용) - MODIFIED"""
        new_conv = Conversation(
            problem_idx=self.problem_idx,
            problem=self.problem,
            answer=self.answer,
            generation_cfg=self.generation_cfg,
            forced_type=self.type,
            forced_student_name=self.student_name,
        )
        
        # 기존 필드 복사
        new_conv.teacher_turns = self.teacher_turns
        new_conv.student_turns = self.student_turns
        new_conv.state = self.state
        
        # Tree 관련 필드는 새로 할당됨
        # current_conversation은 비움 (새 그룹)
        
        return new_conv
    
    def get_current_turn_pairs(self) -> List[Tuple[dict, dict]]:
        """현재 대화에서 (student, teacher) 쌍 추출 - NEW"""
        turn_pairs = []
        i = 0
        while i < len(self.current_conversation) - 1:
            msg1 = self.current_conversation[i]
            msg2 = self.current_conversation[i + 1]
            
            if msg1.get('role') == 'student' and msg2.get('role') == 'teacher':
                turn_pairs.append((msg1, msg2))
                i += 2
            else:
                i += 1
        
        return turn_pairs

    def clear_current_conversation(self):
        """현재 그룹 초기화 - NEW"""
        self.current_conversation = []

    @classmethod
    def from_dataframe(
        cls, row: any, generation_cfg: GenerationConfig
    ) -> "Conversation":

        # Extract the answer (if not present, default to an empty string)
        answer = row.get("Answer", "")

        # Convert the 'Type' column to a ConversationType enum if available.
        forced_type = None
        type_val = row.get("Type")
        if isinstance(type_val, str):
            if type_val in ConversationType.__members__:
                forced_type = ConversationType[type_val]
        elif type_val in ConversationType.__members__:
            forced_type = ConversationType[type_val]

        # Initialize the Conversation instance.
        instance = cls(
            problem=row["Problem"],
            answer=answer,
            generation_cfg=generation_cfg,
            forced_type=forced_type,
            forced_student_name=row.get("Student Name"),
        )

        # Restore conversation list. If stored as string, assume JSON and load.
        conv_data = row.get("Conversation", [])
        if isinstance(conv_data, str):
            try:
                conv_data = eval(conv_data)
            except Exception as e:
                raise ValueError(f"Failed to load 'Conversation' field: {e}")
        instance.conversation = conv_data

        # Restore the conversation state.
        state_val = row.get("State")
        if isinstance(state_val, str):
            if state_val in ConversationState.__members__:
                instance.state = ConversationState[state_val]
        elif state_val in ConversationState.__members__:
            instance.state = ConversationState[state_val]

        # Restore student persona and name.
        instance.student_persona = row.get("Student Persona", instance.student_persona)

        # Restore judge decisions.
        jd_data = row.get("Judge Decisions", {})
        if isinstance(jd_data, str):
            try:
                jd_data = eval(jd_data)
            except Exception as e:
                raise ValueError(f"Failed to load 'Judge Decisions': {e}")
        jd = {}
        for key, decisions in jd_data.items():
            # If decisions is a string, load it as JSON.
            if isinstance(decisions, str):
                try:
                    decisions = eval(decisions)
                except Exception as e:
                    raise ValueError(
                        f"Failed to load judge decisions for key {key}: {e}"
                    )
            jd[key] = [
                JudgeResponse(
                    reasoning=d["reasoning"], decision=JudgeDecision[d["decision"]]
                )
                for d in decisions
            ]
        instance.judge_decisions = jd

        # Restore solutions.
        solutions = row.get("Solutions", [])
        if isinstance(solutions, str):
            try:
                solutions = eval(solutions)
            except Exception as e:
                raise ValueError(f"Failed to load 'Solutions': {e}")
        instance.solutions = solutions

        # Restore rewards.
        rewards = row.get("Rewards", [])
        if isinstance(rewards, str):
            try:
                rewards = eval(rewards)
            except Exception as e:
                raise ValueError(f"Failed to load 'Rewards': {e}")
        instance.rewards = rewards

        # Restore initial attempts.
        initial_attempts = row.get("Initial Attempts", [])
        if isinstance(initial_attempts, str):
            try:
                initial_attempts = eval(initial_attempts)
            except Exception as e:
                raise ValueError(f"Failed to load 'Initial Attempts': {e}")
        instance.initial_attempts = initial_attempts

        # Restore initial rewards.
        initial_rewards = row.get("Initial Rewards", [])
        if isinstance(initial_rewards, str):
            try:
                initial_rewards = eval(initial_rewards)
            except Exception as e:
                raise ValueError(f"Failed to load 'Initial Rewards': {e}")
        instance.initial_rewards = initial_rewards

        return instance

    def get_student_no_tutor_attempt(self):
        messages = [{"role": "user", "content": self.student_attempt}]
        return messages

    def start_conversation(self):

        # If we already started.
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
        # We remove everything between <think> and </think>
        return re.sub(r"<think>.*?</think>", "", content, flags=re.S).replace(
            "<end_of_conversation>", ""
        )

    def _get_hidden_conversation(self):
        conversation = []
        for message in self.conversation:
            conversation.append(
                {
                    "role": message["role"],
                    "content": self._hide_thinking(message["content"]),
                }
            )
        return conversation

    def _get_conversation_from_teacher_perspective(self):
        conversation = []
        for message in self.conversation:
            if message["role"] == "teacher":
                conversation.append(
                    {"role": "assistant", "content": message["content"]}
                )
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

    def get_conversation(self):
        if self.state == ConversationState.TEACHER_TURN:
            conversation = [
                {"role": "system", "content": self.system_prompt_teacher}
            ] + self._get_conversation_from_teacher_perspective()
            return conversation

        elif self.state == ConversationState.STUDENT_TURN:
            # If this is the first message in a guided conversation we request the student to start the conversation
            if self.type == ConversationType.ATTEMPTED and len(self.conversation) == 0:
                return [
                    {"role": "system", "content": self.system_prompt_student_attempt}
                ]
            conversation = [
                {"role": "system", "content": self.system_prompt_student}
            ] + self._get_conversation_from_student_perspective()
            return conversation

        elif self.state == ConversationState.JUDGE_TURN:
            remaining_judge_rules = list(
                set(self.generation_cfg.judges_rules_prompts_paths.keys())
                - set(self.judge_decisions.keys())
            )
            if len(remaining_judge_rules) == 0:
                raise ValueError(
                    "All judge rules have been evaluated, makes no sense we are at this state"
                )
            judge_rule = remaining_judge_rules[0]
            self.judge_evaluation_type = judge_rule
            return [
                {
                    "role": "user",
                    "content": Template(
                        open(
                            self.generation_cfg.judges_rules_prompts_paths[judge_rule]
                        ).read()
                    ).render(conversation=self._get_hidden_conversation()),
                }
            ]

        elif self.state == ConversationState.GENERATE_SOLUTION:
            conversation = [
                {"role": "system", "content": self.system_prompt_student}
            ] + self._get_conversation_from_student_perspective()
            conversation.append({"role": "user", "content": self.student_final_prompt})
            return conversation

    def get_constructivist_conversation(self):
        if self.state == ConversationState.TEACHER_TURN:
            conversation = [
                {"role": "system", "content": self.system_prompt_teacher}
            ] + self._get_conversation_from_teacher_perspective()
            return conversation

        elif self.state == ConversationState.STUDENT_TURN:
            # If this is the first message in a guided conversation we request the student to start the conversation
            if self.type == ConversationType.ATTEMPTED and len(self.conversation) == 0:
                return [
                    {"role": "system", "content": self.system_prompt_student_attempt}
                ]
            conversation = [
                {"role": "system", "content": self.system_prompt_student}
            ] + self._get_conversation_from_student_perspective()
            return conversation

        elif self.state == ConversationState.JUDGE_TURN:
            remaining_judge_rules = list(
                set(self.generation_cfg.judges_rules_constructivist_prompts_paths.keys())
                - set(self.judge_decisions.keys())
            )
            if len(remaining_judge_rules) == 0:
                raise ValueError(
                    "All judge rules have been evaluated, makes no sense we are at this state"
                )
            judge_rule = remaining_judge_rules[0]
            self.judge_evaluation_type = judge_rule
            return [
                {
                    "role": "user",
                    "content": Template(
                        open(
                            self.generation_cfg.judges_rules_constructivist_prompts_paths[judge_rule]
                        ).read()
                    ).render(conversation=self._get_hidden_conversation()),
                }
            ]

        elif self.state == ConversationState.GENERATE_SOLUTION:
            conversation = [
                {"role": "system", "content": self.system_prompt_student}
            ] + self._get_conversation_from_student_perspective()
            conversation.append({"role": "user", "content": self.student_final_prompt})
            return conversation

    def add_message(self, content: str):
        if self.state == ConversationState.TEACHER_TURN:
            self.conversation.append({"role": "teacher", "content": content})
            self.current_conversation.append({"role": "teacher", "content": content})
            self.state = ConversationState.STUDENT_TURN
            if (
                len(self.conversation) >= self.generation_cfg.max_turns
                or "<end_of_conversation>" in content
            ):
                self.state = ConversationState.GENERATE_SOLUTION
        elif self.state == ConversationState.STUDENT_TURN:
            if self.type == ConversationType.ATTEMPTED and len(self.conversation) == 0:
                self.conversation.append(
                    {
                        "role": "student",
                        "content": self.initial_attempt_wrapper.render(attempt=content),
                    }
                )
                self.current_conversation.append(
                    {
                        "role": "student",
                        "content": self.initial_attempt_wrapper.render(attempt=content)
                    }
                )
                self.state = ConversationState.TEACHER_TURN
            else:
                self.conversation.append({"role": "student", "content": content})
                self.current_conversation.append({"role": "student", "content": content})
                self.state = ConversationState.TEACHER_TURN
        if self._exceeded_max_tokens():
            self.state = ConversationState.GENERATE_SOLUTION

        # If there is no judge messages in the config we skip to GENERATE_SOLUTION
        if (
            self.generation_cfg.number_judge_attempts == 0
            and self.state == ConversationState.JUDGE_TURN
        ):
            self.state = ConversationState.GENERATE_SOLUTION

    def add_judge_decisions(self, decisions: List[JudgeResponse]):
        if self.state != ConversationState.JUDGE_TURN:
            raise ValueError("We are not in the judge turn state")
        self.judge_decisions[self.judge_evaluation_type] = decisions
        for decision in decisions:
            if decision.decision == JudgeDecision.REJECT:
                self.failed_judges = True
                if not self.generation_cfg.ignore_rejected_judge:
                    self.state = ConversationState.END
                    return
        if len(self.judge_decisions) == len(
            self.generation_cfg.judges_rules_prompts_paths
        ):
            self.state = ConversationState.GENERATE_SOLUTION

    def add_constructivist_judge_decisions(self, decisions: List[JudgeResponse]):
        if self.state != ConversationState.JUDGE_TURN:
            raise ValueError("We are not in the judge turn state")
        self.judge_decisions[self.judge_evaluation_type] = decisions
        failed = False
        for decision in decisions:
            if decision.decision == JudgeDecision.REJECT:
                failed = True
                if not self.generation_cfg.ignore_rejected_judge:
                    self.constructivist_failed_judges[self.judge_evaluation_type] = failed
                    self.state = ConversationState.END
                    return
        self.constructivist_failed_judges[self.judge_evaluation_type] = failed
        if len(self.judge_decisions) == len(
            self.generation_cfg.judges_rules_constructivist_prompts_paths
        ):
            self.state = ConversationState.GENERATE_SOLUTION

    def add_judge_decision_for_single_turn(self, decisions: List[JudgeResponse], judge_rule_name, turn_hash: str,):
        if self.state != ConversationState.JUDGE_TURN:
            raise ValueError("We are not in the judge turn state")
        self.judge_decisions[turn_hash] = decisions
        for decision in decisions:
            if decision.decision == JudgeDecision.REJECT:
                self.failed_judges = True
                # NOTE: For single-turn setting, making conversation to end when the judge rejected it is risky.
                if not self.generation_cfg.ignore_rejected_judge:
                    self.state = ConversationState.END
                    return
        if len(self.judge_decisions) == self.generation_cfg.number_judge_attempts:
            self.state = ConversationState.GENERATE_SOLUTION

    def add_solutions(self, solutions: List[str]):
        if self.state != ConversationState.GENERATE_SOLUTION:
            raise ValueError("We are not in the generate solution state")
        self.solutions = solutions
        self.state = ConversationState.REWARD_TURN

    def get_solutions_for_reward(self):
        chats = [
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
        return chats

    def get_initial_solutions_for_reward(self):
        chats = [
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
        return chats

    def add_initial_rewards(self, rewards: List[float]):
        self.initial_rewards = rewards

    def add_rewards(self, rewards: List[float]):
        if self.state != ConversationState.REWARD_TURN:
            raise ValueError("We are not in the reward turn state")
        self.rewards = rewards
        self.state = ConversationState.END

    def add_accuracy_rewards(self, rewards: List[float]):
        if self.state != ConversationState.REWARD_TURN:
            raise ValueError("We are not in the reward turn state")
        self.accuracy_rewards = rewards
        self.state = ConversationState.END

    def add_initial_attempts(self, attempts: List[str]):
        self.initial_attempts = attempts

    def get_end_rm_reward(self):
        average_rm_reward = (
            sum(self.rewards) / len(self.rewards) if len(self.rewards) > 0 else None
        )
        return average_rm_reward

    def get_accuracy_reward(self):
        # NOTE: This will be always 0 or 1 in SPO.
        average_accuracy_reward = (
            sum(self.accuracy_rewards) / len(self.accuracy_rewards) if len(self.accuracy_rewards) > 0 else None
        )
        return average_accuracy_reward

    def get_initial_rm_reward(self):
        average_rm_reward = (
            sum(self.initial_rewards) / len(self.initial_rewards)
            if len(self.initial_rewards) > 0
            else None
        )
        return average_rm_reward

    def get_thinking_reward(self):
        # if len(self.rewards) == 0:
        if len(self.accuracy_rewards) == 0:
            return 0.0
        penalty_for_missing_closing_think = 0.0
        count_used_thinking, count_total = 0, 0
        for message in self.conversation:
            if message["role"] == "teacher":
                if message["content"].count("<think>") != message["content"].count(
                    "</think>"
                ):
                    penalty_for_missing_closing_think -= 0.5
                elif message["content"].count("<think>") > 0:
                    count_used_thinking += 1
                count_total += 1
        return (
            penalty_for_missing_closing_think
            + (count_used_thinking / count_total) * 0.5
        )

    def get_end_of_conversation_reward(self):
        # if len(self.rewards) == 0:
        if len(self.accuracy_rewards) == 0:
            return 0.0

        return (
            0.1
            if any(
                "<end_of_conversation>" in message["content"]
                for message in self.conversation
            )
            else 0.0
        )

    def get_length_reward(self):
        texts = []
        for message in self.conversation:
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

    def to_pd(self):
        return pd.DataFrame(
            [
                {
                    "State": self.state.name,
                    "Problem": self.problem,
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
                    "Solutions": self.solutions,
                    # "Rewards": self.rewards,
                    "Accuracy Rewards": self.accuracy_rewards,
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
        conversation = [
            {"role": "system", "content": self.system_prompt_teacher}
        ] + self._get_conversation_from_teacher_perspective()
        return conversation
    
    def split_conversation_into_single_turn(self, start_idx, end_idx):
        actual_end = min(end_idx, len(self.conversation))
        segment = self.conversation[start_idx:actual_end]
        single_turn_conversations = []
        for i in range(0, len(segment), 2):
            pair = segment[i : i + 2]
            if len(pair) > 0:
                single_turn_conversations.append(pair)
                
        return single_turn_conversations

class Classroom:
    def __init__(
        self,
        student_model_cfg: StudentModelConfig,
        teacher_model_cfg: TeacherModelConfig,
        judge_model_cfg: JudgeModelConfig,
        reward_model_cfg: RewardModelConfig,
        generation_cfg: GenerationConfig,
        model_save_path: str,
        log_file_path: str = None,
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
                max_lora_rank=teacher_model_cfg.lora.rank,
                load_and_unload=teacher_model_cfg.vllm.load_and_unload,
                max_number_of_instances=teacher_model_cfg.vllm.max_number_of_instances,
                enable_sleep_mode=teacher_model_cfg.vllm.enable_sleep_mode,
                bits_and_bytes=teacher_model_cfg.vllm.bits_and_bytes,
                use_awq=teacher_model_cfg.vllm.use_awq,
                from_0=teacher_model_cfg.vllm.from_0,
                use_v0=teacher_model_cfg.vllm.use_v0,
                enforce_eager=teacher_model_cfg.vllm.enforce_eager,
                logging_enabled=log_file_path != None,
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
                logging_enabled=log_file_path != None,
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
                logging_enabled=log_file_path != None,
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
                logging_enabled=log_file_path != None,
                log_file_path=log_file_path,
            )
            self.reward_model.sleep()

        self.sampling_params_student = SamplingParams(
            temperature=student_model_cfg.vllm.temperature,
            top_k=student_model_cfg.vllm.top_k,
            top_p=student_model_cfg.vllm.top_p,
            max_tokens=generation_cfg.max_tokens_per_turn,
        )

        # Sadly this is too slow.
        # guided_decoding_params = GuidedDecodingParams(
        #   json=JudgeResponse.model_json_schema(),
        #   backend='lm-format-enforcer'
        # )

        self.sampling_params_judge = SamplingParams(
            temperature=judge_model_cfg.vllm.temperature,
            top_k=judge_model_cfg.vllm.top_k,
            top_p=judge_model_cfg.vllm.top_p,
            max_tokens=generation_cfg.max_tokens_per_judge_attempt,
            # guided_decoding=guided_decoding_params
        )

        self.sampling_params_student_solution = SamplingParams(
            # NOTE: For SPO, we fix n to 1
            # n=generation_cfg.number_student_attempts,
            n=1,
            temperature=student_model_cfg.vllm.temperature,
            top_k=student_model_cfg.vllm.top_k,
            top_p=student_model_cfg.vllm.top_p,
            max_tokens=generation_cfg.max_tokens_per_student_attempt,
        )

        teacher_tokenizer = AutoTokenizer.from_pretrained(
            generation_cfg.tokenizer_to_use
        )
        self.sampling_params_teacher = SamplingParams(
            temperature=teacher_model_cfg.vllm.temperature,
            top_k=teacher_model_cfg.vllm.top_k,
            top_p=teacher_model_cfg.vllm.top_p,
            max_tokens=generation_cfg.max_tokens_per_turn,
            # logits_processors=(
            #     [force_thinking_processor] if generation_cfg.force_thinking else []
            # ),
        )

        self.conversation_sets = []

        # NOTE: Variables for tree-based conversations
        self.conversation_trees = {}
        self.global_tree_idx = {}
        self.conversation_id_counter = 0
        
        # NEW: Advantage storage
        self.node_advantages = {}  # node_id -> advantages
        self.turn_pair_advantages = {}  # (node_id, turn_idx) -> turn advantages

    def _initialize_conversation_tree(self, problem_idx: int):
        """문제별 Tree 초기화"""
        if problem_idx not in self.conversation_trees:
            self.conversation_trees[problem_idx] = ConversationTree(problem_idx)
    
    def _assign_conversation_id(self, conv: Conversation) -> str:
        """Conversation ID 부여"""
        self.global_conversation_id_counter += 1
        conv_id = f"conv_{conv.problem_idx}_{self.global_conversation_id_counter}"
        conv.conversation_id = conv_id
        return conv_id
    
    def _create_initial_tree_node(self, conv: Conversation) -> TreeNode:
        """초기 TreeNode 생성"""
        problem_idx = conv.problem_idx
        self._initialize_conversation_tree(problem_idx)
        
        tree = self.conversation_trees[problem_idx]
        node = tree.create_node(
            conversation_id=conv.conversation_id,
            parent_node_id=None  # Root
        )
        
        conv.current_node_id = node.node_id
        conv.parent_node_id = None
        
        tree.update_leaf_node_mapping(conv.conversation_id, node.node_id)
        
        logger.info(f"Created initial node {node.node_id} for conversation {conv.conversation_id}")
        return node
    
    def _save_current_conversation_to_tree(self, conv: Conversation):
        """현재 대화를 TreeNode에 저장"""
        if conv.current_node_id is None:
            logger.warning(f"Conversation {conv.conversation_id} has no current_node_id")
            return
        
        tree = self.conversation_trees[conv.problem_idx]
        node = tree.nodes.get(conv.current_node_id)
        
        if not node:
            logger.warning(f"Node {conv.current_node_id} not found")
            return
        
        # Turn pairs 추출 및 저장
        turn_pairs = conv.get_current_turn_pairs()
        for student_msg, teacher_msg in turn_pairs:
            node.add_turn_pair(student_msg, teacher_msg)
        
        logger.info(
            f"Saved {len(turn_pairs)} turn pairs to node {node.node_id} "
            f"(conversation {conv.conversation_id})"
        )
    
    def _branch_conversation(
        self,
        original_conv: Conversation,
        branch_size: int
    ) -> List[Conversation]:
        """Conversation branching"""
        # 1. 현재 대화 저장
        self._save_current_conversation_to_tree(original_conv)
        
        # 2. 자식 노드 및 conversations 생성
        tree = self.conversation_trees[original_conv.problem_idx]
        parent_node_id = original_conv.current_node_id
        
        new_conversations = []
        for _ in range(branch_size):
            # Conversation 복사
            new_conv = original_conv.copy()
            
            # 새 ID 할당
            self._assign_conversation_id(new_conv)
            
            # 자식 TreeNode 생성
            child_node = tree.create_node(
                conversation_id=new_conv.conversation_id,
                parent_node_id=parent_node_id
            )
            
            new_conv.current_node_id = child_node.node_id
            new_conv.parent_node_id = parent_node_id
            
            # Leaf 매핑 업데이트
            tree.update_leaf_node_mapping(new_conv.conversation_id, child_node.node_id)
            
            # 현재 대화 초기화
            new_conv.clear_current_conversation()
            
            new_conversations.append(new_conv)
        
        logger.info(
            f"Branched conversation {original_conv.conversation_id} "
            f"into {branch_size} children from node {parent_node_id}"
        )
        
        return new_conversations

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
        elif self.reward_model_cfg.model_name_or_path == "None":
            rewards = [0.0 for _ in prompts]
        return rewards

    def _compute_constructivist_rewards_from_prompts(
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
        elif self.reward_model_cfg.model_name_or_path == "None":
            rewards = [0.0 for _ in prompts]
        return rewards

    def generate_next_teacher_utterances(
        self, conversations: List[Conversation], meta: dict = None
    ) -> List[str]:
        """
        Given a list of Conversation objects in TEACHER_TURN, generate the next teacher utterance
        for each and add it to the conversation.
        """
        if meta is None:
            meta = {}
        # prompts = [conv.get_conversation() for conv in conversations]
        prompts = [conv.get_constructivist_conversation() for conv in conversations]
        responses = self.teacher_model.run_batch(
            prompts, self.sampling_params_teacher, meta
        )
        teacher_utterances = [response.outputs[0].text for response in responses]
        for conv, utterance in zip(conversations, teacher_utterances):
            conv.add_message(utterance)
            conv.teacher_turns += 1
        return teacher_utterances

    def generate_next_student_utterances(
        self, conversations: List[Conversation]
    ) -> List[str]:
        """
        Given a list of Conversation objects in STUDENT_TURN, generate the next student utterance
        for each and add it to the conversation.
        """
        # prompts = [conv.get_conversation() for conv in conversations]
        prompts = [conv.get_constructivist_conversation() for conv in conversations]
        responses = self.student_model.run_batch(prompts, self.sampling_params_student)
        student_utterances = [response.outputs[0].text for response in responses]
        for conv, utterance in zip(conversations, student_utterances):
            conv.add_message(utterance)
            conv.student_turns += 1
        return student_utterances

    def sample_conversations(
        self,
        problems: List[str],
        answers: List[str],
        forced_type: ConversationType = None,
        meta: dict = {},
        compute_initial_attempt: bool = False,
        original_prompts: bool = False
    ) -> List[Conversation]:
        # If we force a certain type of conversation we set it here.
        if forced_type is None:
            if self.generation_cfg.forced_conversation_type == "guided":
                forced_type = ConversationType.GUIDED
            elif self.generation_cfg.forced_conversation_type == "attempt":
                forced_type = ConversationType.ATTEMPTED
            else:
                forced_type = None

        # `conversations` is a list of Conversation class with different problems(repeated for branch_size)
        self.conversation_trees = {}

        for problem_idx in range(len(problems)):
            tree = ConversationTree(problem_idx)
        
            # Create virtual root node (no actual conversation)
            virtual_root = tree.create_node(
                conversation_id=f"virtual_root_{problem_idx}",
                parent_node_id=None
            )
            
            self.conversation_trees[problem_idx] = tree
            logger.info(f"Created tree for problem {problem_idx} with virtual root")

        conversations = []
        branch_size = self.generation_cfg.branch_size if self.generation_cfg.branch_size is not None else 1
        
        for problem_idx, (problem, answer) in tqdm(
            enumerate(zip(problems, answers)),
            total=len(problems),
            desc="Initializing conversations",
        ):
            tree = self.conversation_trees[problem_idx]
            virtual_root_id = 0  # Virtual root is always node_id=0
            
            # Create branch_size conversations for each problem
            for branch_idx in range(branch_size):
                # Create conversation
                conv = Conversation(
                    problem_idx=problem_idx,
                    problem=problem,
                    answer=answer,
                    generation_cfg=self.generation_cfg,
                    forced_type=forced_type
                )
                
                # Assign conversation ID
                self._assign_conversation_id(conv)
                
                # Create child node of virtual root
                node = tree.create_node(
                    conversation_id=conv.conversation_id,
                    parent_node_id=virtual_root_id
                )
                
                # Set node IDs
                conv.current_node_id = node.node_id
                conv.parent_node_id = virtual_root_id
                
                # Update leaf node mapping
                tree.update_leaf_node_mapping(conv.conversation_id, node.node_id)
                
                conversations.append(conv)
        
        logger.info(f"Initialized {len(conversations)} conversations")

        # Start the conversations. (= initialization)
        for conversation in conversations:
            conversation.start_conversation()

        # NOTE: Skipping initial attempt!
        # Only for eval we compute how good the model was initially.
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

            if original_prompts:
                rewards = self._compute_rewards_from_prompts(all_prompts, all_answers)
            else:
                rewards = self._compute_constructivist_rewards_from_prompts(all_prompts, all_answers)

            for conv in conversations:
                curr_len = lengths.pop(0)
                conv_rewards = rewards[:curr_len]
                conv.add_initial_rewards(conv_rewards)
                rewards = rewards[curr_len:]

        round_counter = 0

        # We now alternate between teacher and student turns until the conversation is not in the conversation student/teacher turn state
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
                # NOTE: Copy conversations that reached max_group_size
                if state_to_process == ConversationState.STUDENT_TURN:
                    idx_to_branch = [
                        idx
                        for idx, conv in enumerate(conversations)
                        if (conv.teacher_turns % self.generation_cfg.max_group_size == 0
                            and conv.teacher_turns > 0
                            and conv.state == state_to_process)
                    ]
                    if idx_to_branch:
                        logger.info(
                            f"Branching {len(idx_to_branch)} conversations that reached max group size of {self.generation_cfg.max_group_size}"
                        )
                        conversations = conversations.copy()
                        for idx in sorted(idx_to_branch, reverse=True):
                            original_conv = conversations[idx]
                            
                            # Branch the conversation
                            new_convs = self._branch_conversation(
                                original_conv,
                                branch_size
                            )

                            del conversations[idx]
                            conversations.extend(new_convs)

                logger.info(
                    ("=" * 10)
                    + f"Executing turn {round_counter}: {'Teacher' if state_to_process == ConversationState.TEACHER_TURN else 'Student'}"
                    + ("=" * 10)
                )

                start_time = time.time()

                # We get all conversations that are in the state_to_process state
                conversations_to_process = [
                    conversation
                    for conversation in conversations
                    if conversation.state == state_to_process
                ]
                if len(conversations_to_process) == 0:
                    continue

                # We get the responses from the model using our helper functions.
                if state_to_process == ConversationState.TEACHER_TURN:
                    self.generate_next_teacher_utterances(
                        conversations_to_process, meta
                    )
                else:
                    self.generate_next_student_utterances(conversations_to_process)

                # Next round counter.
                round_counter += 1
                logger.info(f"Took {time.time() - start_time} seconds.")

        # We can put both to sleep.
        self.teacher_model.sleep()
        self.student_model.sleep()

        # Save remaining conversations to tree
        logger.info("Saving remaining conversations to tree")
        for conv in conversations:
            if conv.current_conversation:
                self._save_current_conversation_to_tree(conv)

        # We now evaluate the judge rules and add the judge decisions to the conversations.
        self.run_pedagogical_judges_on_tree()

        # We now generate the solutions
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
                # conversation.get_conversation()
                conversation.get_conversation() if original_prompts else conversation.get_constructivist_conversation()
                for conversation in conversations_to_process
            ]
            responses = self.student_model.run_batch(
                messages, self.sampling_params_student_solution
            )
            for conversation, response in zip(conversations_to_process, responses):
                conversation.add_solutions([output.text for output in response.outputs])

        self.student_model.sleep()
        logger.info(f"Took {time.time() - start_time} seconds.")

        # Compute rewards.
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
            rewards = self._compute_rewards_from_prompts(all_prompts, all_answers) if original_prompts else self._compute_constructivist_rewards_from_prompts(all_prompts, all_answers)
            for conv in reward_convs:
                curr_len = lengths.pop(0)
                conv_rewards = rewards[:curr_len]
                conv.add_accuracy_rewards(conv_rewards)
                rewards = rewards[curr_len:]

                # NOTE: Add accuracy reward to tree as well for later use.
                accuracy_reward = conv.get_accuracy_reward()
                tree = self.conversation_trees[conv.problem_idx]
                tree.assing_reward_to_conversation(conv.current_node_id, accuracy_reward)

        logger.info(f"Took {time.time() - start_time} seconds.")
        # Free memory
        gc.collect()
        torch.cuda.empty_cache()

        self.conversation_sets.append(conversations)
        return conversations

    def run_judges(self, conversations: List[Conversation]):
        """
        Given a list of Conversation objects in JUDGE_TURN, generate the next judge utterance
        for each and add it to the conversation.
        """
        # We now evaluate the judge rules
        logger.info(("=" * 10) + "Running judges" + ("=" * 10))
        start_time = time.time()

        num_attempts_required = self.generation_cfg.number_judge_attempts
        max_rounds = 5
        while any(
            [
                conversation.state == ConversationState.JUDGE_TURN
                for conversation in conversations
            ]
        ):
            logger.info(("=" * 15) + "Judges round" + ("=" * 15))
            # Select conversations in judge turn.
            conversations_to_process = [
                conv
                for conv in conversations
                if conv.state == ConversationState.JUDGE_TURN
            ]

            # Dictionary to collect valid JudgeResponse objects per conversation.
            valid_responses = {conv: [] for conv in conversations_to_process}

            for _judge_round in range(max_rounds):
                logger.info(
                    ("=" * 10) + f"Judges inner round {_judge_round}" + ("=" * 10)
                )
                pending = []  # List of tuples: (conversation, message)
                # For each conversation, schedule as many generations as are missing.
                for conv in conversations_to_process:
                    missing = num_attempts_required - len(valid_responses[conv])
                    if missing > 0:
                        for _ in range(missing):
                            # pending.append((conv, conv.get_conversation()))
                            pending.append((conv, conv.get_constructivist_conversation()))

                if not pending:
                    break  # All conversations have enough valid responses.
                logger.info("Number of pending judge responses:" + str(len(pending)))

                # Run a batch for all pending messages.
                pending_messages = [msg for conv, msg in pending]
                responses = self.judge_model.run_batch(
                    pending_messages, self.sampling_params_judge
                )

                # Map each response back to its conversation.
                for (conv, _), response in zip(pending, responses):
                    for output in response.outputs:
                        try:
                            # We only take stuff that is between { and }
                            out_text = output.text[
                                output.text.find("{") : output.text.rfind("}") + 1
                            ].replace("\\", "")
                            decision = JudgeResponse(
                                **json.loads(out_text, strict=False)
                            )
                            valid_responses[conv].append(decision)
                        except Exception as e:
                            continue

            for conv in conversations_to_process:
                while len(valid_responses[conv]) < num_attempts_required:
                    logger.warning(
                        "Judge decision ran out of attempts, adding default decision"
                    )
                    valid_responses[conv].append(
                        JudgeResponse(reasoning="max turns exceeded", decision="OK")
                    )
                # conv.add_judge_decisions(valid_responses[conv])
                conv.add_constructivist_judge_decisions(valid_responses[conv])

        self.judge_model.sleep()
        logger.info(f"Took {time.time() - start_time} seconds.")

    def run_pedagogical_judges_on_tree(self):
        """
        Run pedagogical judges on all turn pairs in the tree
        """
        logger.info("=" * 20 + " Running pedagogical judges on tree " + "=" * 20)
        rule_names = self.generation_cfg.judges_rules_constructivist_prompts_paths.keys()

        all_judge_tasks = []
        
        # Collect all turn pairs from all trees
        for problem_idx, tree in self.conversation_trees.items():
            for node_id, node in tree.nodes.items():
                # Skip virtual root (no turn pairs)
                if not node.turn_pairs:
                    continue
                
                for turn_idx, turn_pair in enumerate(node.turn_pairs):
                    # Construct judge messages
                    for rule_name in rule_names:
                        messages = self._construct_judge_messages(turn_pair, rule_name)
                        all_judge_tasks.append((tree, node, turn_idx, rule_name, messages))
        
        logger.info(f"Total pedagogical judge tasks: {len(all_judge_tasks)}")
        
        if not all_judge_tasks:
            logger.warning("No judge tasks to process")
            return
        
        # Process in batches
        batch_size = 16
        num_attempts = self.generation_cfg.number_judge_attempts
        
        for i in range(0, len(all_judge_tasks), batch_size):
            batch = all_judge_tasks[i:i+batch_size]
            batch_messages = [task[4] for task in batch]
            
            # Run judges multiple times
            all_responses = []
            for _ in range(num_attempts):
                responses = self.judge_model.run_batch(
                    batch_messages,
                    self.sampling_params_judge
                )
                all_responses.append(responses)
            
            # Store results
            for task_idx, (tree, node, turn_idx, rule_name, _) in enumerate(batch):
                turn_pair = node.turn_pairs[turn_idx]
                
                # Parse each attempt's results
                valid_decisions = []
                for attempt_responses in all_responses:
                    response = attempt_responses[task_idx]
                    
                    for output in response.outputs:
                        try:
                            out_text = output.text[
                                output.text.find("{"):output.text.rfind("}")+1
                            ].replace("\\", "")
                            decision = JudgeResponse(**json.loads(out_text, strict=False))
                            valid_decisions.append(decision)
                        except Exception as e:
                            continue
                
                # Store judge results
                if valid_decisions:
                    if rule_name not in turn_pair.judge_results:
                        turn_pair.judge_results[rule_name] = []
                    turn_pair.judge_results[rule_name].extend(valid_decisions)
                else:
                    logger.warning(
                        f"No valid judge decisions for node {node.node_id}, "
                        f"turn {turn_idx}, rule '{rule_name}'"
                    )
        
        # Compute pedagogical rewards from judge results
        self._compute_pedagogical_rewards_from_judges()
        
        logger.info("Pedagogical judges completed")

    def _hide_thinking(self, content: str):
        # We remove everything between <think> and </think>
        return re.sub(r"<think>.*?</think>", "", content, flags=re.S).replace(
            "<end_of_conversation>", ""
        )

    def _get_hidden_conversation(self, messages):
        conversation = []
        for message in messages:
            conversation.append(
                {
                    "role": message["role"],
                    "content": self._hide_thinking(message["content"]),
                }
            )
        return conversation

    def _construct_judge_messages(
        self,
        current_turn: TurnPair,
        rule_name: str
    ) -> List[dict]:
        """Construct messages for judge evaluation"""
        single_turn_messages = []
        # Include current turn
        single_turn_messages.append(current_turn.student_message)
        single_turn_messages.append(current_turn.teacher_message)

        rule_prompt_path = self.generation_cfg.judges_rules_constructivist_prompts_paths[rule_name]

        if hasattr(self.generation_cfg, 'judge_system_prompt_path'):
            prompt = [
                {
                    "role": "user",
                    "content": Template(
                        open(
                            rule_prompt_path
                        ).read()
                    ).render(conversation=self._get_hidden_conversation(single_turn_messages)),
                }
            ]

        return prompt

    def _compute_pedagogical_rewards_from_judges(self):
        """Compute pedagogical rewards from judge results"""
        for tree in self.conversation_trees.values():
            for node in tree.nodes.values():
                for turn_pair in node.turn_pairs:
                    if turn_pair.judge_results:
                        total_ok_count = 0
                        number_of_judge_decisions = 0
                        for decisions in turn_pair.judge_results.values():
                            if decisions:
                                # Compute OK ratio as reward
                                ok_count = sum(
                                    1 for d in decisions
                                    if d.decision == JudgeDecision.OK
                                )
                                total_ok_count += ok_count
                                number_of_judge_decisions += len(decisions)
                        if total_ok_count > 0 and number_of_judge_decisions > 0:
                            turn_pair.pedagogical_reward = total_ok_count / number_of_judge_decisions
                        else:
                            turn_pair.pedagogical_reward = 0.0
                    else:
                        turn_pair.pedagogical_reward = 0.0

    def compute_all_advantages(
        self,
        lambda_pedagogical: float = 1.0
    ) -> Dict[int, Dict[str, float]]:
        """모든 Tree의 advantages 계산 - TURN PAIR 레벨로!"""
        logger.info("Computing turn-pair-level advantages for all trees")
        
        all_advantages = {}  # node_id -> advantages (backward compatibility)
        all_turn_advantages = {}  # (node_id, turn_idx) -> turn pair advantages
        
        for problem_idx, tree in self.conversation_trees.items():
            logger.info(f"Computing advantages for problem {problem_idx}")
            
            # Tree 레벨 계산
            tree.build_levels()
            tree.compute_v_values()
            tree.compute_accuracy_advantages()
            tree.compute_pedagogical_advantages()
            
            # Turn pair에 advantage 할당
            tree.assign_advantages_to_turn_pairs(lambda_pedagogical)
            
            # Node-level advantages (backward compatibility)
            node_advantages = tree.get_all_advantages()
            all_advantages.update(node_advantages)
            
            # Turn-pair-level advantages (NEW!)
            turn_advantages = tree.get_all_turn_pair_advantages()
            for turn_adv in turn_advantages:
                key = (turn_adv['node_id'], turn_adv['turn_idx'])
                all_turn_advantages[key] = turn_adv
        
        logger.info(f"Computed advantages for {len(all_advantages)} nodes and {len(all_turn_advantages)} turn pairs")
        
        # 두 가지를 모두 저장
        self.node_advantages = all_advantages
        self.turn_pair_advantages = all_turn_advantages
        
        return all_advantages  # backward compatibility

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
        import os

        reward = conversation.get_end_rm_reward()
        if reward == None:
            conversations = [
                conv
                for conv in self.conversation_sets[-1]
                if conv.problem == conversation.problem
            ]
            rewards = [conv.get_end_rm_reward() for conv in conversations]
            rewards = [reward for reward in rewards if reward is not None]
            minimum_reward = -self.generation_cfg.extra_penalty_for_rejected_judges

            return minimum_reward

        if conversation.failed_judges:
            reward -= self.generation_cfg.extra_penalty_for_rejected_judges
        return reward
    
    def get_accuracy_reward(self, conversation: Conversation):
        reward = conversation.get_accuracy_reward() # student accuracy
        if reward == None:
            conversations = [
                conv
                for conv in self.conversation_sets[-1]
                if conv.problem == conversation.problem
            ]
            rewards = [conv.get_accuracy_reward() for conv in conversations] # student accuracy
            rewards = [reward for reward in rewards if reward is not None]
            minimum_reward = -self.generation_cfg.extra_penalty_for_rejected_judges

            return minimum_reward
        return reward

    def get_pedagogical_alignment_reward(self, conversation: Conversation):
        # We need to know if judge finished at all. 
        if conversation.get_accuracy_reward() == None:
            return -self.generation_cfg.extra_penalty_for_rejected_judges

        reward = 0
        # Make pedagogical reward softly
        # -> Change constructivist_failed_judges type to Dict[String, Bool]
        if self.generation_cfg.use_soft_pedagogical_reward:
            sum_reward_from_judges = 0
            for is_failed in conversation.constructivist_failed_judges.values():
                # Add reward if judge's output is OK
                sum_reward_from_judges += 0 if is_failed else 1
            reward -= self.generation_cfg.extra_penalty_for_rejected_judges * (1 - (sum_reward_from_judges / len(conversation.constructivist_failed_judges)))
        else:
            if not all(conversation.constructivist_failed_judges.values()):
                reward -= self.generation_cfg.extra_penalty_for_rejected_judges
        return reward

    def get_thinking_reward(self, conversation: Conversation):
        return conversation.get_thinking_reward()

    def get_end_of_conversation_reward(self, conversation: Conversation):
        return conversation.get_end_of_conversation_reward()

    def get_length_reward(self, conversation: Conversation):
        return conversation.get_length_reward()