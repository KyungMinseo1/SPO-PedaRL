import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
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
from src.classroom_segment import Classroom, Conversation
from config.train_rl_model import RLModelTrainingConfig
from src.utils.utils import init_logger
logger = init_logger()

import warnings

warnings.filterwarnings("ignore")
load_dotenv(override=False)

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
    
    with lock:
        # Conversations 샘플링
        conversations = classroom.sample_conversations(
            problems=problems,
            answers=answers,
            meta=meta
        )
        
        # Advantages 계산 (turn pair 레벨!)
        lambda_pedagogical = getattr(config, 'pedagogical_advantage_lambda', 1.0)
        node_advantages = classroom.compute_all_advantages(lambda_pedagogical)
    
    # Response 구성 - turn pair별 advantages 포함
    conversations_with_advantages = []
    for conv in conversations:
        node_id = conv.current_node_id
        
        # 이 conversation의 tree와 node 가져오기
        tree = classroom.conversation_trees[conv.problem_idx]
        node = tree.nodes.get(node_id)
        
        if node is None:
            logger.warning(f"Node {node_id} not found for conversation {conv.conversation_id}")
            # Fallback: node-level advantage 사용
            adv = node_advantages.get(node_id, {
                'accuracy_advantage': 0.0,
                'pedagogical_advantage': 0.0,
                'combined_advantage': 0.0,
                'v_value': 0.0
            })
            conversations_with_advantages.append({
                "conversation_id": conv.conversation_id,
                "node_id": node_id,
                "messages": conv.get_trainable_representation(),
                "advantages": adv,
                "turn_pair_advantages": []  # 빈 리스트
            })
            continue
        
        # FIXED: Root부터 현재 노드까지 경로상의 모든 turn pairs 추출!
        turn_pair_advs = tree.get_turn_pairs_along_path(node_id)
        
        # Node-level advantage (backward compatibility)
        node_adv = node_advantages.get(node_id, {
            'accuracy_advantage': 0.0,
            'pedagogical_advantage': 0.0,
            'combined_advantage': 0.0,
            'v_value': 0.0
        })
        
        conversations_with_advantages.append({
            "conversation_id": conv.conversation_id,
            "node_id": node_id,
            "messages": conv.get_trainable_representation(),
            "advantages": node_adv,  # backward compatibility
            "turn_pair_advantages": turn_pair_advs  # FIXED: 전체 경로!
        })
    
    # WandB 로깅용 데이터 수집
    if config.logging.wandb:
        # Turn pair별 메트릭 수집
        all_turn_metrics = []
        for conv_data in conversations_with_advantages:
            conv_id = conv_data["conversation_id"]
            for turn_adv in conv_data["turn_pair_advantages"]:
                all_turn_metrics.append({
                    "conversation_id": conv_id,
                    "node_id": conv_data["node_id"],
                    "turn_idx": turn_adv["turn_idx"],
                    "pedagogical_reward": turn_adv["pedagogical_reward"],
                    "accuracy_advantage": turn_adv["accuracy_advantage"],
                    "pedagogical_advantage": turn_adv["pedagogical_advantage"],
                    "combined_advantage": turn_adv["combined_advantage"],
                })
        
        # DataFrame으로 변환하여 로깅
        if all_turn_metrics:
            import pandas as pd
            df_turns = pd.DataFrame(all_turn_metrics)
            df_turns = df_turns.astype(str)
            wandb.log({
                f"turn_pairs_batch_{len(classroom.conversation_sets)}": wandb.Table(dataframe=df_turns)
            })
        
        # 기존 conversation-level 메트릭도 로깅
        df_table = classroom.to_pd_latest()
        
        rewards = [classroom.get_accuracy_reward(c) for c in conversations]
        df_table["accuracy_reward"] = rewards
        
        rewards = [classroom.get_pedagogical_alignment_reward(c) for c in conversations]
        df_table["pedagogical_alignment_reward"] = rewards
        
        rewards = [classroom.get_thinking_reward(c) for c in conversations]
        df_table["thinking_reward"] = rewards
        
        rewards = [classroom.get_end_of_conversation_reward(c) for c in conversations]
        df_table["end_of_conversation_reward"] = rewards
        
        rewards = [classroom.get_length_reward(c) for c in conversations]
        df_table["length_reward"] = rewards
        
        df_table["total_reward"] = (
            df_table["accuracy_reward"]
            + df_table["pedagogical_alignment_reward"]
            + df_table["thinking_reward"]
            + df_table["end_of_conversation_reward"]
            + df_table["length_reward"]
        )
        
        df_table = df_table.astype(str)
        wandb.log({
            f"conversations_batch_{len(classroom.conversation_sets)}": wandb.Table(dataframe=df_table)
        })
    
    return {
        "conversations": conversations_with_advantages,
        "node_advantages": node_advantages  # backward compatibility
    }


@app.post("/get_end_rm_reward")
def get_end_rm_reward(request: RewardRequest):
    global classroom
    conversations: list[Conversation] = [
        classroom.get_conversation_by_text(c) for c in request.conversations
    ]
    rewards = [classroom.get_end_rm_reward(c) for c in conversations]
    return rewards

@app.post("/get_accuracy_reward")
def get_accuracy_reward(request: RewardRequest):
    global classroom
    conversations: list[Conversation] = [
        classroom.get_conversation_by_text(c) for c in request.conversations
    ]
    rewards = [classroom.get_accuracy_reward(c) for c in conversations]
    return rewards

@app.post("/get_pedagogical_alignment_reward")
def get_pedagogical_alignment_reward(request: RewardRequest):
    global classroom
    conversations: list[Conversation] = [
        classroom.get_conversation_by_text(c) for c in request.conversations
    ]
    rewards = [classroom.get_pedagogical_alignment_reward(c) for c in conversations]
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