import os
import wandb
import hydra
import uvicorn
import threading
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from src.classroom import Classroom, Conversation
from config.train_rl_model import RLModelTrainingConfig
from src.utils.utils import init_logger

logger = init_logger()

import warnings

warnings.filterwarnings("ignore")
load_dotenv()

lock = threading.Lock()

cs = ConfigStore.instance()
cs.store(name="config", node=RLModelTrainingConfig)

classroom: Classroom = None
config: RLModelTrainingConfig = None
app = FastAPI()


class ConversationSampleRequest(BaseModel):
    problems: List[str]
    answers: List[str]
    meta: dict = {}


class RewardRequest(BaseModel):
    conversations: list[str]


@app.post("/sample_conversations")
def sample_conversations(request: ConversationSampleRequest):
    global classroom, config

    problems = request.problems
    answers = request.answers
    meta = request.meta
    conversations = None
    with lock:
        conversations = classroom.sample_conversations(
            problems=problems, answers=answers, meta=meta
        )

    accuracy_and_rewards = [classroom.get_end_rm_reward(c) for c in conversations]
    accuracy_rewards = [a for a, _ in accuracy_and_rewards]
    end_rm_rewards   = [r for _, r in accuracy_and_rewards]

    thinking_rewards = [classroom.get_thinking_reward(c) for c in conversations]
    eoc_rewards      = [classroom.get_end_of_conversation_reward(c) for c in conversations]
    length_rewards   = [classroom.get_length_reward(c) for c in conversations]

    turn_stats = [c.get_logging_stats() for c in conversations]
    total_teacher_turns = [s["total_teacher_turns"] for s in turn_stats]
    participating_teacher_turns = [s["participating_teacher_turns"] for s in turn_stats]
    first_reject_turns = [s["first_reject_turn"] for s in turn_stats]
    mean_turn_rewards = [s["mean_turn_reward"] for s in turn_stats]
    mean_pedagogical_rewards = [s["mean_pedagogical_reward"] for s in turn_stats]

    ok_rates = [c.get_judge_ok_rate() for c in conversations]
    ok_rates_valid = [r for r in ok_rates if r is not None]

    df_table = classroom.to_pd_latest()
    df_table["accuracy_reward"]       = accuracy_rewards
    df_table["end_rm_reward"]         = end_rm_rewards
    df_table["thinking_reward"]       = thinking_rewards
    df_table["end_of_conversation_reward"] = eoc_rewards
    df_table["length_reward"]         = length_rewards
    df_table["total_teacher_turns"] = total_teacher_turns
    df_table["participating_teacher_turns"] = participating_teacher_turns
    df_table["first_reject_turn"] = first_reject_turns
    df_table["mean_turn_reward"] = mean_turn_rewards
    df_table["mean_pedagogical_reward"] = mean_pedagogical_rewards
    df_table["total_reward"] = [
        e + t + o + l
        for e, t, o, l in zip(end_rm_rewards, thinking_rewards, eoc_rewards, length_rewards)
    ]
    df_table = df_table.astype(str)
    if config.logging.wandb:
        step = len(classroom.conversation_sets)
        log_dict = {
            f"batch_{step}": wandb.Table(dataframe=df_table),
        }
        wandb.log(log_dict, step=step)

    return [c.get_trainable_representation() for c in conversations]


@app.post("/get_end_rm_reward")
def get_end_rm_reward(request: RewardRequest):
    global classroom
    conversations: list[Conversation] = [
        classroom.get_conversation_by_text(c) for c in request.conversations
    ]
    rewards = [classroom.get_end_rm_reward(c) for c in conversations]
    return [total for _, total in rewards]


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

@app.get("/get_batch_metrics")
def get_batch_metrics():
    global classroom
    conversations = classroom.conversation_sets[-1]
    accuracy_rewards = [c.get_end_rm_reward() or 0.0 for c in conversations]
    ok_rates = [c.get_judge_ok_rate() for c in conversations]
    ok_rates_valid = [r for r in ok_rates if r is not None]
    turn_stats = [c.get_logging_stats() for c in conversations]
    total_turns = [s["total_teacher_turns"] for s in turn_stats]
    participating_turns = [s["participating_teacher_turns"] for s in turn_stats]
    return {
        "accuracy": sum(accuracy_rewards) / len(accuracy_rewards),
        "judge_ok_rate": sum(ok_rates_valid) / len(ok_rates_valid) if ok_rates_valid else None,
        "avg_total_teacher_turns": sum(total_turns) / len(total_turns) if total_turns else 0.0,
        "avg_participating_teacher_turns": sum(participating_turns) / len(participating_turns) if participating_turns else 0.0,
    }


@hydra.main(config_path="config/train_rl", version_base=None)
def main(cfg: RLModelTrainingConfig):
    global classroom, config

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

    uvicorn.run(app, host="0.0.0.0", port=cfg.generation.server_port)


if __name__ == "__main__":
    main()
