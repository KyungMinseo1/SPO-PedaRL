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
    """
    DEPRECATED: Backward compatibility only.
    Use /sample_nodes for TreeNode-based learning.
    """
    # Redirect to sample_nodes
    return sample_nodes(request)


@app.post("/sample_nodes")
def sample_nodes(request: ConversationSampleRequest):
    global classroom, config
    
    problems = request.problems
    answers = request.answers
    meta = request.meta
    
    with lock:
        # Conversations 샘플링 (Tree 생성)
        conversations = classroom.sample_conversations(
            problems=problems,
            answers=answers,
            meta=meta
        )
        
        # Advantages 계산 (turn pair 레벨!)
        lambda_pedagogical = getattr(config, 'pedagogical_advantage_lambda', 1.0)
        reward_list = getattr(config, 'reward_list', [])
        reward_weights = getattr(config, 'reward_weights', [])
        
        node_advantages = classroom.compute_all_advantages(
            lambda_pedagogical=lambda_pedagogical,
            reward_list=reward_list,
            reward_weights=reward_weights
        ) # Per Node(grouped turn)
    
    # ===== 핵심 변경: Node 단위로 반환 (Context 없음!) =====
    # Trees는 classroom에 이미 저장되어 있음
    all_nodes = []
    
    for problem_idx, tree in classroom.conversation_trees.items():
        # 모든 노드를 context 없이 추출!
        nodes_for_training = tree.get_all_nodes_for_training()
        all_nodes.extend(nodes_for_training)
    
    logger.info(f"Returning {len(all_nodes)} nodes from {len(problems)} problems (no context)")
    
    # WandB 로깅용 데이터 수집
    if config.logging.wandb:
        # Node별 메트릭 수집
        node_metrics = []
        for node_data in all_nodes:
            for turn_adv in node_data["node_turn_pairs"]:
                node_metrics.append({
                    "problem_idx": node_data["problem_idx"],
                    "node_id": node_data["node_id"],
                    "turn_idx": turn_adv["turn_idx"],
                    "pedagogical_reward": turn_adv["pedagogical_reward"],
                    "accuracy_advantage": turn_adv["accuracy_advantage"],
                    "pedagogical_advantage": turn_adv["pedagogical_advantage"],
                    "end_of_conversation_advantage": turn_adv["end_of_conversation_advantage"],
                    "length_advantage": turn_adv["length_advantage"],
                    "combined_advantage": turn_adv["combined_advantage"],
                })
        
        # DataFrame으로 변환하여 로깅
        if node_metrics:
            import pandas as pd
            df_nodes = pd.DataFrame(node_metrics)
            df_nodes = df_nodes.astype(str)
            wandb.log({
                f"nodes_batch_{len(classroom.conversation_sets)}": wandb.Table(dataframe=df_nodes)
            })
        
        # NOTE: Conversation-level 로깅 제거
        # Node-level 메트릭이 더 정확하고 유용함
        # Conversations는 이제 학습에 사용되지 않음
    
    return {
        "nodes": all_nodes,
        "total_nodes": len(all_nodes),
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