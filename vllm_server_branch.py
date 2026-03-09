import os
import json
import pandas as pd
import wandb
import hydra
import uvicorn
import threading
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from src.classroom_branch import Classroom, Conversation
from config.train_rl_model import RLModelTrainingConfig
from src.utils.utils import init_logger


def _gamma_state_path() -> str:
    """Return path to the gamma state file, derived from the save_dir in config."""
    global config
    return os.path.join(config.logging.save_dir, "accuracy_gamma_state.json")


def _save_gamma_state(gamma: float, batch_count: int) -> None:
    path = _gamma_state_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"accuracy_decay_gamma": gamma, "batch_count": batch_count}, f, indent=2)
    logger.info(f"[gamma] Saved gamma={gamma:.6f} (batch_count={batch_count}) → {path}")


def _load_gamma_state() -> Optional[dict]:
    path = _gamma_state_path()
    if os.path.exists(path):
        with open(path) as f:
            state = json.load(f)
        logger.info(f"[gamma] Loaded gamma state from {path}: {state}")
        return state
    return None

logger = init_logger()

import warnings

warnings.filterwarnings("ignore")
load_dotenv()

lock = threading.Lock()

cs = ConfigStore.instance()
cs.store(name="config", node=RLModelTrainingConfig)

classroom: Classroom = None
config: RLModelTrainingConfig = None
batch_count: int = 0
app = FastAPI()


class ConversationSampleRequest(BaseModel):
    problems: List[str]
    answers: List[str]
    meta: dict = {}
    problem_idx: List[int]
    solve_rates: List[float]


class ConversationBranchRequest(BaseModel):
    """Request for the branch-aware endpoint that returns per-turn-pair advantages."""
    problems: List[str]
    answers: List[str]
    meta: dict = {}
    problem_idx: List[int]
    solve_rates: List[float]

class RewardRequest(BaseModel):
    conversations: list[str]

@app.post("/sample_conversations_branch")
def sample_conversations_branch(request: ConversationBranchRequest):
    """
    Branch-aware sampling endpoint.

    Returns per-conversation data including:
    - All turn pairs (main + auxiliary) with computed GRPO advantages
    - Rollout-level reward values for logging

    Advantage computation (compute_all_advantages_flat):
    - accuracy, end_of_conversation, [think if not turn]:  GRPO within same problem_idx group
    - pedagogical_alignment, length, [think if turn]:      GRPO within same (problem_idx, teacher_turn, is_main_turn) group
    """
    global classroom, config

    reward_list = config.train.reward_list
    reward_weights = config.train.reward_weights
    active_rewards = [r for r, w in zip(reward_list, reward_weights) if w > 0]
    is_think_turn_reward = (
        config.generation.use_thinking
        and config.generation.convert_think_to_turn_reward
    )
    accuracy_decay_gamma = config.train.accuracy_reward_gamma

    global batch_count
    with lock:
        conversations = classroom.sample_conversations(
            problems=request.problems,
            answers=request.answers,
            solve_rates=request.solve_rates,
            meta=request.meta,
            problem_idx=request.problem_idx,
            active_rewards=active_rewards,
        )

        classroom.accuracy_decay_gamma *= accuracy_decay_gamma
        batch_count += 1
        _save_gamma_state(classroom.accuracy_decay_gamma, batch_count)

        classroom.compute_all_advantages_flat(
            conversations=conversations,
            reward_list=reward_list,
            reward_weights=reward_weights,
            is_think_turn_reward=is_think_turn_reward
        )

    # ── Serialise conversations ───────────────────────────────────────────
    result_conversations = []
    for conv in conversations:
        turn_pairs_data = []
        for turn_idx in sorted(conv.turn_pairs.keys()):
            for turn_pair in conv.turn_pairs[turn_idx]:
                turn_pairs_data.append(
                    {
                        "teacher_turn": turn_pair.teacher_turn,
                        "is_main_turn": turn_pair.is_main_turn,
                        "student_message": turn_pair.student_message,
                        "teacher_message": turn_pair.teacher_message,
                        "judge_results": {
                            key: [{"reasoning": d.reasoning, "decision": d.decision.name} for d in decisions]
                            for key, decisions in turn_pair.judge_results.items()
                        },
                        # rewards
                        "pedagogical_reward": turn_pair.pedagogical_reward,
                        "length_reward": turn_pair.length_reward,
                        "think_reward": turn_pair.think_reward,
                        # advantages
                        "accuracy_advantage": turn_pair.accuracy_advantage,
                        "end_of_conversation_advantage": turn_pair.end_of_conversation_advantage,
                        "pedagogical_advantage": turn_pair.pedagogical_advantage,
                        "length_advantage": turn_pair.length_advantage,
                        "think_advantage": turn_pair.think_advantage,
                    }
                )
        result_conversations.append(
            {
                "problem_idx": conv.problem_idx,
                "problem": conv.problem,
                "answer": conv.answer,
                "conversation": conv.conversation,
                "student_persona": conv.student_persona,
                "student_name": conv.student_name,
                "turn_pairs": turn_pairs_data,
                "solutions": conv.solutions,
                "accuracy_reward": conv.get_accuracy_reward(classroom.accuracy_decay_gamma),
                "end_of_conversation_reward": conv.get_end_of_conversation_reward(),
                "think_reward": (
                    conv.get_thinking_reward()
                    if config.generation.use_thinking and not config.generation.convert_think_to_turn_reward
                    else None
                ),
            }
        )
        
    def _smean(lst):
        return sum(lst) / len(lst) if lst else None

    if config.logging.wandb:
        turn_rows = []
        for conv_d in result_conversations:
            for turn_pair in conv_d["turn_pairs"]:
                turn_rows.append(
                    {
                        "problem_idx": conv_d["problem_idx"],
                        "teacher_turn": turn_pair["teacher_turn"],
                        "is_main_turn": turn_pair["is_main_turn"],
                        "student_message": turn_pair["student_message"],
                        "teacher_message": turn_pair["teacher_message"],
                        "judge_results": turn_pair["judge_results"],
                        "pedagogical_reward": turn_pair["pedagogical_reward"],
                        "think_reward": turn_pair["think_reward"],
                        "length_reward": turn_pair["length_reward"],                      
                        "end_of_conversation_advantage": turn_pair["end_of_conversation_advantage"],
                        "accuracy_advantage": turn_pair["accuracy_advantage"],
                        "pedagogical_advantage": turn_pair["pedagogical_advantage"],
                        "think_advantage": turn_pair["think_advantage"],
                        "length_advantage": turn_pair["length_advantage"],
                        "end_of_conversation_advantage": turn_pair["end_of_conversation_advantage"],
                    }
                )
        df_turns = pd.DataFrame(turn_rows)

        # Per-conversation table
        rows = []
        for conv_d in result_conversations:
            main_tps = [turn_pair for turn_pair in conv_d["turn_pairs"] if turn_pair.get("is_main_turn", True)]
            if config.generation.use_thinking and config.generation.convert_think_to_turn_reward:
                rows.append(
                    {
                        "problem_idx": conv_d["problem_idx"],
                        "problem": conv_d["problem"],
                        "answer": conv_d["answer"],
                        "conversation": conv_d["conversation"],
                        "student_persona": conv_d["student_persona"],
                        "student_name": conv_d["student_name"],
                        "accuracy_reward": conv_d.get("accuracy_reward"),
                        "eoc_reward": conv_d.get("end_of_conversation_reward"),
                        "n_main_turns": len(main_tps),
                        "avg_ped_reward": _smean(
                            [turn_pair["pedagogical_reward"] for turn_pair in main_tps if turn_pair.get("pedagogical_reward") is not None]
                        ),
                        "avg_length_reward": _smean(
                            [turn_pair["length_reward"] for turn_pair in main_tps if turn_pair.get("length_reward") is not None]
                        ),
                        "avg_think_reward": _smean(
                            [turn_pair["think_reward"] for turn_pair in main_tps if turn_pair.get("think_reward") is not None]
                        ),
                    }
                )
            elif config.generation.use_thinking and not config.generation.convert_think_to_turn_reward:
                rows.append(
                    {
                        "problem_idx": conv_d["problem_idx"],
                        "problem": conv_d["problem"],
                        "answer": conv_d["answer"],
                        "conversation": conv_d["conversation"],
                        "student_persona": conv_d["student_persona"],
                        "student_name": conv_d["student_name"],
                        "accuracy_reward": conv_d.get("accuracy_reward"),
                        "eoc_reward": conv_d.get("end_of_conversation_reward"),
                        "think_reward": conv_d.get("think_reward"),
                        "n_main_turns": len(main_tps),
                        "avg_ped_reward": _smean(
                            [turn_pair["pedagogical_reward"] for turn_pair in main_tps if turn_pair.get("pedagogical_reward") is not None]
                        ),
                        "avg_length_reward": _smean(
                            [turn_pair["length_reward"] for turn_pair in main_tps if turn_pair.get("length_reward") is not None]
                        ),
                    }
                )
            else:
                rows.append(
                    {
                        "problem_idx": conv_d["problem_idx"],
                        "problem": conv_d["problem"],
                        "answer": conv_d["answer"],
                        "conversation": conv_d["conversation"],
                        "student_persona": conv_d["student_persona"],
                        "student_name": conv_d["student_name"],
                        "accuracy_reward": conv_d.get("accuracy_reward"),
                        "eoc_reward": conv_d.get("end_of_conversation_reward"),
                        "n_main_turns": len(main_tps),
                        "avg_ped_reward": _smean(
                            [turn_pair["pedagogical_reward"] for turn_pair in main_tps if turn_pair.get("pedagogical_reward") is not None]
                        ),
                        "avg_length_reward": _smean(
                            [turn_pair["length_reward"] for turn_pair in main_tps if turn_pair.get("length_reward") is not None]
                        ),
                    }
                )
            
        df = pd.DataFrame(rows)

        wandb.log(
            {f"batch_{len(classroom.conversation_sets)}": wandb.Table(dataframe=df.astype(str))}
        )
        wandb.log(
            {f"turns_batch_{len(classroom.conversation_sets)}": wandb.Table(dataframe=df_turns.astype(str))}
        )

    return {"conversations": result_conversations}


@app.post("/get_end_rm_reward")
def get_end_rm_reward(request: RewardRequest):
    global classroom
    conversations: list[Conversation] = [
        classroom.get_conversation_by_text(c) for c in request.conversations
    ]
    rewards = [classroom.get_end_rm_reward(c) for c in conversations]
    return rewards

@app.post("/get_constructivist_reward")
def get_constructivist_reward(request: RewardRequest):
    global classroom
    conversations: list[Conversation] = [
        classroom.get_conversation_by_text(c) for c in request.conversations
    ]
    rewards = [classroom.get_constructivist_reward(c) for c in conversations]
    return rewards


@app.post("/get_thinking_reward")
def get_thinking_reward(request: RewardRequest):
    global classroom
    conversations: list[Conversation] = [
        classroom.get_conversation_by_text(c) for c in request.conversations
    ]
    rewards = [classroom.get_thinking_reward(c) for c in conversations]
    return rewards


@app.post("/get_end_of_conversation_reward")
def get_end_of_conversation_reward(request: RewardRequest):
    global classroom
    conversations: list[Conversation] = [
        classroom.get_conversation_by_text(c) for c in request.conversations
    ]
    rewards = [classroom.get_end_of_conversation_reward(c) for c in conversations]
    return rewards


@app.post("/get_length_reward")
def get_length_reward(request: RewardRequest):
    global classroom
    conversations: list[Conversation] = [
        classroom.get_conversation_by_text(c) for c in request.conversations
    ]
    rewards = [classroom.get_length_reward(c) for c in conversations]
    return rewards


@app.get("/wait_batch")
def wait_batch():
    # This endpoint waits (blocks) until the current batch (if any) is finished.
    with lock:
        return {"message": "Batch has been run."}


@app.get("/gamma_state")
def get_gamma_state():
    """Return the current accumulated accuracy_decay_gamma and batch count."""
    return {
        "accuracy_decay_gamma": classroom.accuracy_decay_gamma,
        "batch_count": batch_count,
        "state_file": _gamma_state_path(),
    }


@hydra.main(config_path="config/train_rl", version_base=None)
def main(cfg: RLModelTrainingConfig):
    global classroom, config, batch_count

    # We merge the config with the defaults
    default_config = OmegaConf.structured(RLModelTrainingConfig)

    # Merge loaded config with defaults
    cfg = OmegaConf.merge(
        default_config, cfg
    )  # Unspecified keys will use defaults from RLModelTrainingConfig

    config = cfg

    if cfg.logging.wandb:
        wandb.init(
            project=cfg.logging.wandb_project + "-server",
            name=cfg.logging.wandb_run_name,
            entity=cfg.logging.wandb_entity,
            group=cfg.logging.run_group,
            tags=cfg.logging.wandb_tags,
            config=OmegaConf.to_object(cfg),
        )

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    classroom = Classroom(
        cfg.student_model,
        cfg.teacher_model,
        cfg.judge_model,
        cfg.reward_model,
        cfg.generation,
        os.path.join(cfg.logging.save_dir, "policy"),
        log_file_path=None,  # hydra_cfg['runtime']['output_dir']
    )

    # Restore gamma state if resuming from a previous run
    gamma_state = _load_gamma_state()
    if gamma_state is not None:
        classroom.accuracy_decay_gamma = gamma_state["accuracy_decay_gamma"]
        batch_count = gamma_state["batch_count"]
        logger.info(
            f"[gamma] Resumed: accuracy_decay_gamma={classroom.accuracy_decay_gamma:.6f}, "
            f"batch_count={batch_count}"
        )
    else:
        logger.info("[gamma] No previous gamma state found, starting from 1.0")

    uvicorn.run(app, host="0.0.0.0", port=cfg.generation.server_port)


if __name__ == "__main__":
    main()