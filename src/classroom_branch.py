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

from collections import defaultdict

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

from src.turn_pair import TurnPair

class Conversation:
    def __init__(
        self,
        problem_idx: int,
        problem: str,
        answer: str,
        solve_rate: float,
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
        self.solve_rate = solve_rate
        self.conversation_id = ""  # Conversation ID

        # NOTE: Conversation attributes
        self.current_conversation: List[dict] = []  # Current conversation turns (for grouping into nodes)
        self.grouped_conversation = {}
        self.teacher_turns = 0
        self.student_turns = 0

        # NOTE: Branching related attributes
        self.auxilary_teacher_message = defaultdict(list)  # To store teacher messages that are generated after the main teacher message in the same turn.
        self.turn_pairs = defaultdict(list)

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
    
    def get_current_turn_pairs(self) -> List[Tuple[dict, dict]]:
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

    def _get_hidden_conversation(self, turn_idx: int, turn: TurnPair, n_turns: int):
        messages = []
        for i in range(max(0, turn_idx - (n_turns-1)), turn_idx):
            for prev_turn in self.turn_pairs[i]:
                if prev_turn.is_main_turn:
                    student_message = {
                        "role": prev_turn.student_message["role"],
                        "content": self._hide_thinking(prev_turn.student_message["content"]),
                    }
                    teacher_message = {
                        "role": prev_turn.teacher_message["role"],
                        "content": self._hide_thinking(prev_turn.teacher_message["content"]),
                    }
                    messages.append(student_message)
                    messages.append(teacher_message)

        student_message = {
            "role": turn.student_message["role"],
            "content": self._hide_thinking(turn.student_message["content"]),
        }
        teacher_message = {
            "role": turn.teacher_message["role"],
            "content": self._hide_thinking(turn.teacher_message["content"]),
        }
        messages.append(student_message)
        messages.append(teacher_message)
        return messages

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
                    ).render(
                        n_turns=self.generation_cfg.n_turns_to_sample,
                        conversation=self._get_hidden_conversation()
                    ),
                }
            ]

        elif self.state == ConversationState.GENERATE_SOLUTION:
            conversation = [
                {"role": "system", "content": self.system_prompt_student}
            ] + self._get_conversation_from_student_perspective()
            conversation.append({"role": "user", "content": self.student_final_prompt})
            return conversation

    def get_constructivist_conversation(self, rule_name: Optional[str] = None):
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
            if rule_name is None:
                raise ValueError("You need to give rule prompts to get the constructivist judge decisions.")
            return [
                (
                    [
                        {
                            "role": "user",
                            "content": Template(
                                open(
                                    self.generation_cfg.judges_rules_constructivist_prompts_paths[rule_name]
                                ).read()
                            ).render(
                                n_turns=min(self.generation_cfg.n_turns_to_sample, turn_idx+1),
                                conversation=self._get_hidden_conversation(turn_idx, turn, self.generation_cfg.n_turns_to_sample)
                            ),
                        }
                    ],
                    turn
                )
                for turn_idx, turns in self.turn_pairs.items()
                for turn in turns
            ]

        elif self.state == ConversationState.GENERATE_SOLUTION:
            conversation = [
                {"role": "system", "content": self.system_prompt_student}
            ] + self._get_conversation_from_student_perspective()
            conversation.append({"role": "user", "content": self.student_final_prompt})
            return conversation

    def add_message(self, content: str, is_processed: bool = False):
        if self.state == ConversationState.TEACHER_TURN:
            if not is_processed:
                # Main turn: add to conversation and advance state
                self.conversation.append({"role": "teacher", "content": content})
                self.state = ConversationState.STUDENT_TURN
                if (
                    len(self.conversation) >= self.generation_cfg.max_turns
                    or "<end_of_conversation>" in content
                ):
                    self.state = ConversationState.JUDGE_TURN
            else:
                # Auxiliary branch turn: store separately, do NOT change state
                self.auxilary_teacher_message[self.teacher_turns -1].append({"role": "teacher", "content": content})
                return  # Skip the max-token check below; state must not change
        elif self.state == ConversationState.STUDENT_TURN:
            if self.type == ConversationType.ATTEMPTED and len(self.conversation) == 0:
                self.conversation.append(
                    {
                        "role": "student",
                        "content": self.initial_attempt_wrapper.render(attempt=content),
                    }
                )
                self.state = ConversationState.TEACHER_TURN
            else:
                self.conversation.append({"role": "student", "content": content})
                self.state = ConversationState.TEACHER_TURN
        if self._exceeded_max_tokens():
            self.state = ConversationState.JUDGE_TURN

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

    def get_accuracy_reward(self, accumulated_accuracy_reward_gamma: Optional[float] = None):
        average_accuracy_reward = (
            sum(self.accuracy_rewards) / len(self.accuracy_rewards) if len(self.accuracy_rewards) > 0 else None
        )
        if average_accuracy_reward is not None:
            if accumulated_accuracy_reward_gamma is not None:
                delta_accuracy_reward = average_accuracy_reward - self.solve_rate
                decayed_accuracy_reward = delta_accuracy_reward + self.solve_rate * accumulated_accuracy_reward_gamma
                return decayed_accuracy_reward
            else:
                return average_accuracy_reward - self.solve_rate
        return average_accuracy_reward

    def get_initial_rm_reward(self):
        average_rm_reward = (
            sum(self.initial_rewards) / len(self.initial_rewards)
            if len(self.initial_rewards) > 0
            else None
        )
        return average_rm_reward

    def get_pedagogical_alignment_reward_for_turns(self):
        # if len(self.rewards) == 0:
        if len(self.accuracy_rewards) == 0:
            return 0.0
        reward = 0.0
        total_turns = 0
        for turn_idx, turns in self.turn_pairs.items():
            for turn in turns:
                if turn.pedagogical_reward is not None:
                    reward += turn.pedagogical_reward
                    total_turns += 1
        return reward / total_turns if total_turns > 0 else 0.0

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

    def get_thinking_reward_for_turns(self):
        # if len(self.rewards) == 0:
        if len(self.accuracy_rewards) == 0:
            return 0.0
        reward = 0.0
        total_turns = 0
        for turn_idx, turns in self.turn_pairs.items():
            for turn in turns:
                if turn.think_reward is not None:
                    reward += turn.think_reward
                    total_turns += 1
        return reward / total_turns if total_turns > 0 else 0.0

    def get_length_reward_for_turns(self):
        # if len(self.rewards) == 0:
        if len(self.accuracy_rewards) == 0:
            return 0.0
        reward = 0.0
        total_turns = 0
        for turn_idx, turns in self.turn_pairs.items():
            for turn in turns:
                if turn.length_reward is not None:
                    reward += turn.length_reward
                    total_turns += 1
        return reward / total_turns if total_turns > 0 else 0.0

    def get_end_of_conversation_reward(self):
        # if len(self.rewards) == 0:
        if len(self.accuracy_rewards) == 0:
            return 0.0

        return (
            1.0
            if any(
                "<end_of_conversation>" in message["content"]
                for message in self.conversation
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
                    "Solutions": self.solutions,
                    "Accuracy Rewards": self.accuracy_rewards,
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

        self.tokenizer = get_tokenizer(generation_cfg.tokenizer_to_use)

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
            n=generation_cfg.number_student_attempts,
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
        self.turn_pair_sets = []

        self.global_conversation_id_counter = 0

        self.conversation_problem_mapping = defaultdict(list)  # problem_idx -> list of conversation_ids

        self.turn_pair_advantages = {}  # (node_id, turn_idx) -> turn advantages

        self.accuracy_decay_gamma = 1.0
    
    def _assign_conversation_id(self, conv: Conversation) -> str:
        self.global_conversation_id_counter += 1
        conv_id = f"conv_{conv.problem_idx}_{self.global_conversation_id_counter}"
        conv.conversation_id = conv_id
        return conv_id

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
        self, conversations: List[Conversation], meta: dict = None, branch_size: int = 1
    ) -> List[str]:
        """
        Given a list of Conversation objects in TEACHER_TURN, generate the next teacher utterance
        for each and add it to the conversation.
        """
        if meta is None:
            meta = {}
        # prompts = [conv.get_conversation() for conv in conversations]
        conv_ids_to_conversations = {conv.conversation_id: [conv, False] for conv in conversations}
        conv_ids = [conv.conversation_id for conv in conversations for _ in range(branch_size)]
        prompts = [conv.get_constructivist_conversation() for conv in conversations for _ in range(branch_size)]
        responses = self.teacher_model.run_batch(
            prompts, self.sampling_params_teacher, meta
        )
        teacher_utterances = [response.outputs[0].text for response in responses]
        for conv_id, utterance in zip(conv_ids, teacher_utterances):
            conv, is_processed = conv_ids_to_conversations[conv_id]
            conv.add_message(utterance, is_processed)
            if not is_processed:
                conv.teacher_turns += 1
            conv_ids_to_conversations[conv_id] = [conv, True]
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
        problem_idx: List[int],
        solve_rates: List[float],
        forced_type: ConversationType = None,
        meta: dict = {},
        compute_initial_attempt: bool = False,
        original_prompts: bool = False,
        active_rewards: List[str] = None,
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
        self.conversation_problem_mapping = defaultdict(list)
        conversations = []
        
        # Using Repeat Sampler to create multiple conversations for the same problem
        for problem, idx, answer, solve_rate in tqdm(zip(problems, problem_idx, answers, solve_rates),
            total=len(problems),
            desc="Initializing conversations",
        ):
            
            conv = Conversation(
                problem_idx=idx,
                problem=problem,
                answer=answer,
                solve_rate=solve_rate,
                generation_cfg=self.generation_cfg,
                forced_type=forced_type
            )

            self._assign_conversation_id(conv)
                
            conversations.append(conv)
            self.conversation_problem_mapping[idx].append(conv.conversation_id)
        
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
                start_time = time.time()

                # We get all conversations that are in the state_to_process state
                conversations_to_process = [
                    conversation
                    for conversation in conversations
                    if conversation.state == state_to_process
                ]
                if len(conversations_to_process) == 0:
                    continue

                logger.info(
                    ("=" * 10)
                    + f"Executing turn {round_counter}: {'Teacher' if state_to_process == ConversationState.TEACHER_TURN else 'Student'}"
                    + ("=" * 10)
                )

                # We get the responses from the model using our helper functions.
                if state_to_process == ConversationState.TEACHER_TURN:
                    self.generate_next_teacher_utterances(
                        conversations_to_process, meta, branch_size=self.generation_cfg.branch_size
                    )
                else:
                    self.generate_next_student_utterances(conversations_to_process)

                # Next round counter.
                round_counter += 1
                logger.info(f"Took {time.time() - start_time} seconds.")

        # We can put both to sleep.
        self.teacher_model.sleep()
        self.student_model.sleep()

        self.align_turn_pairs(conversations)

        # We now evaluate the judge rules and add the judge decisions to the conversations.
        if active_rewards is None or "pedagogical_alignment" in active_rewards:
            self.run_pedagogical_judges_on_each_turn_pair(conversations)

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

            logger.info(f"Preparing solutions for reward computation.")
            for conv in reward_convs:
                prompts = conv.get_solutions_for_reward()
                lengths.append(len(prompts))
                all_prompts.extend(prompts)
                all_answers.extend([conv.answer] * len(prompts))
            rewards = self._compute_rewards_from_prompts(all_prompts, all_answers) if original_prompts else self._compute_constructivist_rewards_from_prompts(all_prompts, all_answers)
            
            logger.info(f"Computed rewards for {len(all_prompts)} solutions. Assigning accuracy/end_of_conversation rewards to conversations.")
            for conv in reward_convs:
                curr_len = lengths.pop(0) # Current number of solutions for each conversation
                conv_rewards = rewards[:curr_len]
                conv.add_accuracy_rewards(conv_rewards)
                rewards = rewards[curr_len:]
                

        logger.info(f"Computing rewards for all turn pairs.")
        if active_rewards is None or "length" in active_rewards:
            self.get_length_reward_on_each_pair(reward_convs)
        if active_rewards is None or "think" in active_rewards:
            if self.generation_cfg.use_thinking and self.generation_cfg.convert_think_to_turn_reward:
                self.get_think_reward_on_each_pair(reward_convs)

        logger.info(f"Took {time.time() - start_time} seconds.")
        # Free memory
        gc.collect()
        torch.cuda.empty_cache()

        self.conversation_sets.append(conversations)

        return conversations

    def align_turn_pairs(self, conversations: List[Conversation]):
        """
        Align turn pairs across conversations.
        """
        logger.info("=" * 20 + " Aligning turn pairs " + "=" * 20)
        for conv in conversations:
            teacher_turn = 0
            for idx, real_conv in enumerate(conv.conversation):
                if real_conv["role"] == "teacher":
                    if (idx > 0 or conv.conversation[idx - 1]["role"] == "student"):
                        conv.turn_pairs[teacher_turn].append(
                            TurnPair(
                                conversation_id=conv.conversation_id,
                                teacher_turn=teacher_turn,
                                is_main_turn=True,
                                teacher_message=real_conv,
                                student_message=conv.conversation[idx - 1])
                        )  # Create a new turn pair for each teacher message
                        auxilary_teacher_message = conv.auxilary_teacher_message[teacher_turn]
                        for aux_message in auxilary_teacher_message:
                            conv.turn_pairs[teacher_turn].append(
                                TurnPair(
                                    conversation_id=conv.conversation_id,
                                    teacher_turn=teacher_turn,
                                    is_main_turn=False,
                                    teacher_message=aux_message,
                                    student_message=conv.conversation[idx - 1])
                            )
                    teacher_turn += 1

    def run_pedagogical_judges_on_each_turn_pair(self, conversations: List[Conversation]):
        """
        Run pedagogical judges on all turn pairs including auxiliary teacher messages.
        """
        logger.info("=" * 20 + " Running pedagogical judges on each pair " + "=" * 20)
        rule_names = self.generation_cfg.judges_rules_constructivist_prompts_paths.keys()

        # Select conversations in judge turn.
        conversations_to_process = [
            conv for conv in conversations
            if conv.state == ConversationState.JUDGE_TURN
        ]

        if not conversations_to_process:
            logger.warning("No conversations in JUDGE_TURN state")
            return

        # Collect all (TurnPair, rule_name, judge_message) tasks
        all_judge_tasks = []  # List of (TurnPair, rule_name, judge_message)

        for conv in conversations_to_process:
            for rule_name in rule_names:
                turns_for_judging = conv.get_constructivist_conversation(rule_name)
                if turns_for_judging:
                    for turn_message_for_judge, turn in turns_for_judging:
                        all_judge_tasks.append(
                            (turn, rule_name, turn_message_for_judge)
                        )

        logger.info(f"Total pedagogical judge tasks: {len(all_judge_tasks)}")

        if not all_judge_tasks:
            logger.warning("No judge tasks to process")
            # Still need to advance state
            for conv in conversations_to_process:
                conv.state = ConversationState.GENERATE_SOLUTION
            return

        num_attempts = self.generation_cfg.number_judge_attempts
        judge_messages = [task[2] for task in all_judge_tasks]

        # Run judges num_attempts times, collecting all responses
        all_responses = []
        for _ in range(num_attempts):
            responses = self.judge_model.run_batch(
                judge_messages,
                self.sampling_params_judge
            )
            all_responses.append(responses)


        failed_tasks = []  # List of (task_idx, TurnPair, rule_name, judge_message)

        for task_idx, (turn_pair, rule_name, messages) in enumerate(all_judge_tasks):
            for attempt_responses in all_responses:
                response = attempt_responses[task_idx]
                for output in response.outputs:
                    try:
                        out_text = output.text[
                            output.text.find("{") : output.text.rfind("}") + 1
                        ].replace("\\", "")
                        decision = JudgeResponse(**json.loads(out_text, strict=False))
                        turn_pair.judge_results.setdefault(rule_name, []).append(decision)
                    except Exception:
                        continue

            if len(turn_pair.judge_results.get(rule_name, [])) < num_attempts:
                failed_tasks.append((task_idx, turn_pair, rule_name, messages))
                logger.warning(
                    f"Failed to get valid judge response for "
                    f"conv_id={turn_pair.conversation_id}, "
                    f"teacher_turn={turn_pair.teacher_turn}, "
                    f"rule='{rule_name}'"
                )

        number_of_retry = 0
        max_retry = 5

        while failed_tasks and number_of_retry < max_retry:
            logger.info(
                f"Retrying {len(failed_tasks)} failed judge tasks "
                f"(attempt {number_of_retry + 1}/{max_retry})"
            )
            retry_messages = [t[3] for t in failed_tasks]
            responses = self.judge_model.run_batch(retry_messages, self.sampling_params_judge)

            for i, (task_idx, turn_pair, rule_name, _) in enumerate(failed_tasks):
                response = responses[i]
                for output in response.outputs:
                    try:
                        out_text = output.text[
                            output.text.find("{") : output.text.rfind("}") + 1
                        ].replace("\\", "")
                        decision = JudgeResponse(**json.loads(out_text, strict=False))
                        turn_pair.judge_results.setdefault(rule_name, []).append(decision)
                        break
                    except Exception:
                        continue

            number_of_retry += 1

            failed_tasks = [
                (task_idx, turn_pair, rule_name, messages)
                for task_idx, turn_pair, rule_name, messages in failed_tasks
                if len(turn_pair.judge_results.get(rule_name, [])) < num_attempts
            ]

        if failed_tasks:
            logger.warning(
                f"After {max_retry} retries, {len(failed_tasks)} judge tasks still failed"
            )

        # Compute pedagogical rewards from collected judge results
        self._compute_pedagogical_rewards_from_judges(conversations_to_process)

        for conv in conversations_to_process:
            conv.state = ConversationState.GENERATE_SOLUTION

        logger.info("Pedagogical judges completed")

    def get_length_reward_on_each_pair(self, conversations: List[Conversation]):
        texts = []

        for conv in conversations:
            for turn_pairs in conv.turn_pairs.values():
                for turn_pair in turn_pairs:
                    texts.append((turn_pair, turn_pair.teacher_message["content"]))

        text_tokens_count = [len(self.tokenizer.encode(t[1])) for t in texts]

        def length_reward(tokens_count):
            return -1.0 if tokens_count >= self.generation_cfg.max_tokens_per_turn - 1 else 0.0

        for (turn_pair, _), tokens_count in zip(texts, text_tokens_count):
            turn_pair.length_reward = length_reward(tokens_count)

        logger.info(f"Length rewards computed for all {len(texts)} turn pairs")

    def get_think_reward_on_each_pair(self, conversations: List[Conversation]):
        texts = []

        for conv in conversations:
            for turn_pairs in conv.turn_pairs.values():
                for turn_pair in turn_pairs:
                    texts.append((turn_pair, turn_pair.teacher_message["content"]))

        def think_reward(message):
            penalty_for_missing_closing_think = 0.0
            used_thinking = 0.0
            if message.count("<think>") != message.count(
                    "</think>"
                ):
                penalty_for_missing_closing_think = -1.0
            elif message.count("<think>") > 0:
                used_thinking += 1.0
            return (penalty_for_missing_closing_think + used_thinking)

        for (turn_pair, teacher_message) in texts:
            turn_pair.think_reward = think_reward(teacher_message)
        
        logger.info(f"Think rewards computed for all {len(texts)} turn pairs")

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
        rule_name: str,
        context_turns: List[TurnPair] = []
    ) -> List[dict]:
        """Construct messages for judge evaluation"""
        messages = []

        for turn in context_turns:
            messages.append(turn.student_message)
            messages.append(turn.teacher_message)

        # Include current turn
        messages.append(current_turn.student_message)
        messages.append(current_turn.teacher_message)

        rule_prompt_path = self.generation_cfg.judges_rules_constructivist_prompts_paths[rule_name]

        prompt = [
            {
                "role": "user",
                "content": Template(
                    open(
                        rule_prompt_path
                    ).read()
                ).render(
                    conversation=self._get_hidden_conversation(messages),
                    n_turns=len(context_turns) + 1,
                ),
            }
        ]

        return prompt

    def _compute_pedagogical_rewards_from_judges(self, conversations_to_process: List[Conversation]):
        """Compute pedagogical rewards from judge results"""
        for conv in conversations_to_process:
            for turn_idx, turn_pairs in conv.turn_pairs.items():
                for turn_pair in turn_pairs:
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

    def compute_all_advantages_flat(
        self,
        conversations: List[Conversation],
        reward_list: List[str] = [],
        reward_weights: List[float] = [],
        is_think_turn_reward: bool = False,
    ) -> None:
        """
        Compute per-TurnPair advantages for all rewards and store them in the TurnPair objects.

        Rollout-level rewards  (accuracy, end_of_conversation, think when
        is_think_turn_reward=False) are normalised within the group of
        conversations that share the same problem_idx.

        Turn-level rewards  (pedagogical_alignment, length, think when
        is_think_turn_reward=True) are normalised within the group of
        TurnPair objects that share the same
        (problem_idx, teacher_turn) key — this intentionally includes both
        main and auxiliary branch turns so all branch_size responses for a
        given student state are compared against each other.

        Combined advantage is computed in the trainer as a weighted sum of
        per-reward token-level advantages; it is NOT stored back to TurnPair.
        """

        reward_weight_map = dict(zip(reward_list, reward_weights))

        # ── 1. Group conversations by problem_idx ─────────────────────────
        problem_to_convs: Dict[int, List[Conversation]] = defaultdict(list)
        for conv in conversations:
            problem_to_convs[conv.problem_idx].append(conv)

        # ── 2. Rollout-level rewards ──────────────────────────────────────
        rollout_reward_names = [
            r for r in ["accuracy", "end_of_conversation"]
            if r in reward_list
        ]
        if not is_think_turn_reward and "think" in reward_list:
            rollout_reward_names.append("think")

        for reward_name in rollout_reward_names:
            for problem_idx, convs in problem_to_convs.items():
                raw: List[float] = []
                for conv in convs:
                    if reward_name == "accuracy":
                        r = conv.get_accuracy_reward(self.accuracy_decay_gamma)
                    elif reward_name == "end_of_conversation":
                        r = conv.get_end_of_conversation_reward()
                    else:  # think (rollout)
                        r = conv.get_thinking_reward()
                    raw.append(r if r is not None else 0.0)

                rt = torch.tensor(raw, dtype=torch.float32)
                mean_r = rt.mean().item()
                std_r = rt.std().item() if len(raw) > 1 else 0.0

                for conv, r_val in zip(convs, raw):
                    adv_val = (r_val - mean_r) / (std_r + 1e-4)
                    for turn_pairs in conv.turn_pairs.values():
                        for turn_pair in turn_pairs:
                            if turn_pair.is_main_turn:
                                if reward_name == "accuracy":
                                    turn_pair.accuracy_advantage = adv_val
                                elif reward_name == "end_of_conversation":
                                    turn_pair.end_of_conversation_advantage = adv_val
                                else:
                                    turn_pair.think_advantage = adv_val

        # ── 3. Turn-level rewards ─────────────────────────────────────────
        turn_reward_names = [
            r for r in ["pedagogical_alignment", "length"]
            if r in reward_list
        ]
        if is_think_turn_reward and "think" in reward_list:
            turn_reward_names.append("think")

        # key: (problem_idx, teacher_turn, is_main_turn)
        turn_groups: Dict[tuple, List] = defaultdict(list)
        for conv in conversations:
            for turn_pairs in conv.turn_pairs.values():
                for turn_pair in turn_pairs:
                    key = (conv.problem_idx, turn_pair.teacher_turn)
                    turn_groups[key].append(turn_pair)

        for reward_name in turn_reward_names:
            for key, group in turn_groups.items():
                raw: List[float] = []
                for turn_pair in group:
                    if reward_name == "pedagogical_alignment":
                        r = turn_pair.pedagogical_reward
                    elif reward_name == "length":
                        r = turn_pair.length_reward
                    else:  # think (turn)
                        r = turn_pair.think_reward
                    raw.append(r if r is not None else 0.0)

                rt = torch.tensor(raw, dtype=torch.float32)
                mean_r = rt.mean().item()
                std_r = rt.std().item() if len(raw) > 1 else 0.0

                for turn_pair, r_val in zip(group, raw):
                    adv_val = (r_val - mean_r) / (std_r + 1e-4)
                    if reward_name == "pedagogical_alignment":
                        turn_pair.pedagogical_advantage = adv_val
                    elif reward_name == "length":
                        turn_pair.length_advantage = adv_val
                    else:
                        turn_pair.think_advantage = adv_val

        n_turn_groups = len(turn_groups)
        n_problems = len(problem_to_convs)
        logger.info(
            f"[compute_all_advantages_flat] "
            f"{len(conversations)} conversations, "
            f"{n_problems} problems, "
            f"{n_turn_groups} turn groups"
        )

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
        reward = conversation.get_accuracy_reward(self.accuracy_decay_gamma) # student accuracy
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

    # Computing Average Reward per conversation.
    def get_pedagogical_alignment_reward(self, conversation: Conversation):
        return conversation.get_pedagogical_alignment_reward_for_turns()

    def get_thinking_reward(self, conversation: Conversation):
        if self.generation_cfg.convert_think_to_turn_reward:
            return conversation.get_thinking_reward_for_turns()
        else:
            return conversation.get_thinking_reward()

    def get_end_of_conversation_reward(self, conversation: Conversation):
        return conversation.get_end_of_conversation_reward()

    def get_length_reward(self, conversation: Conversation):
        return conversation.get_length_reward_for_turns()