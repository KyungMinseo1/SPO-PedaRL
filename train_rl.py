from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers.trainer_utils import get_last_checkpoint
import os
import hydra
import wandb
import torch
import math
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from config.train_rl_model import RLModelTrainingConfig
from transformers import set_seed
from dotenv import load_dotenv
from transformers import AutoTokenizer
from peft import LoraConfig as PeftLoraConfig
from datasets import Dataset

# from src.grpo.config import ClassroomGRPOConfig
# from src.gdpo.config import ClassroomGDPOConfig
from src.grpo.config_segment import ClassroomSPOConfig

# from src.grpo.trainer import ClassroomGRPOTrainer
# from src.gdpo.trainer import ClassroomGDPOTrainer
from src.grpo.trainer_segment import ClassroomSPOTrainer

from src.utils.utils import (
    construct_end_of_conversation_reward_func,
    construct_accuracy_reward_func,
    construct_pedagogical_alignment_reward_func,
    # construct_end_rm_reward_func,
    construct_length_reward_func,
    construct_thinking_reward_func,
    init_logger,
)
import warnings

from utils.data import load_datasets

warnings.filterwarnings("ignore")
load_dotenv()

logger = init_logger()

cs = ConfigStore.instance()
cs.store(name="config", node=RLModelTrainingConfig)


@hydra.main(config_path="config/train_rl", version_base=None)
def main(cfg: RLModelTrainingConfig):

    #############################################################################
    # Setup
    #############################################################################

    # We merge the config with the default config
    default_config = OmegaConf.structured(RLModelTrainingConfig)
    cfg = OmegaConf.merge(default_config, cfg)

    model_config = cfg.teacher_model
    train_config = cfg.train
    logging_config = cfg.logging
    lora_config = model_config.lora
    data_config = cfg.dataset

    set_seed(cfg.seed)

    assert len(train_config.reward_list) == len(train_config.reward_weights), "Your presented reward list & reward weights have different size. Please check your reward list again."

    # TODO: Should change Reward Dictionary
    """
    reward_dict = {
        "accuracy": construct_accuracy_reward_func(cfg.generation.server_port),
        "pedagogical alignment": construct_pedagogical_alignment_reward_func(cfg.generation.server_port),
        "thinking": construct_thinking_reward_func(cfg.generation.server_port),
        "end of conversation": construct_end_of_conversation_reward_func(cfg.generation.server_port),
        "length": construct_length_reward_func(cfg.generation.server_port),
    }
    reward_func_list = [
        reward_dict[reward_name]
        for reward_name in train_config.reward_list
        if reward_dict.get(reward_name, None) is not None
    ]
    """

    kwargs = [InitProcessGroupKwargs(timeout=timedelta(hours=10))]
    accelerator = Accelerator(kwargs_handlers=kwargs)

    if logging_config.wandb and accelerator.is_main_process:
        wandb.init(
            project=logging_config.wandb_project,
            name=logging_config.wandb_run_name,
            entity=logging_config.wandb_entity,
            group=logging_config.run_group,
            tags=logging_config.wandb_tags,
            config=OmegaConf.to_object(cfg),
        )
    accelerator.wait_for_everyone()

    torch_dtype = torch.bfloat16
    model_kwargs = dict(
        trust_remote_code=True,
        attn_implementation="sdpa",
        dtype=torch_dtype,
        use_cache=False if train_config.gradient_checkpointing else True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=True
    )

    #############################################################################
    # Load the datasets
    #############################################################################

    logger.info(f"Loading datasets from {data_config.train_datasets}")
    train_dataset, _ = load_datasets(data_config, cfg.seed)
    logger.info(f"Loaded {len(train_dataset)} training examples")

    if data_config.lower_bound_solve_rate is not None:
        logger.info(f"Filtering training dataset with solve rate lower bound: {data_config.lower_bound_solve_rate}")
        train_dataset = train_dataset.filter(lambda x: x["llama8b_solve_rate"] >= data_config.lower_bound_solve_rate)
        logger.info(f"{len(train_dataset)} training examples remaining after filtering with solve rate lower bound of {data_config.lower_bound_solve_rate}")

    train_dataset = train_dataset.select(range(min(len(train_dataset), data_config.max_train_examples)))

    def apply_template(example):
        problem = example["problem"]
        answer = example["answer"]

        return {"prompt": problem, "answer": answer}

    train_dataset: Dataset = train_dataset.map(
        apply_template, num_proc=4, desc="Applying template"
    )

    #############################################################################
    # PEFT Config
    #############################################################################
    peft_config = None
    if lora_config.enable:
        peft_config = PeftLoraConfig(
            r=lora_config.rank,
            lora_alpha=lora_config.alpha,
            target_modules=lora_config.target_modules,
            lora_dropout=lora_config.dropout,
            bias=lora_config.bias,
            task_type="CAUSAL_LM",
        )

    #############################################################################
    # Initializing num_generations
    #############################################################################

        # For TreeNode, each node is one "generation"
        #
        # Level 0: branch_size nodes
        # Level 1: branch_size^2 nodes
        # Level 2: branch_size^3 nodes
        # Total: branch_size + branch_size^2 + branch_size^3
        #      = branch_size * (1 + branch_size + branch_size^2)
        #      = branch_size * (branch_size^levels - 1) / (branch_size - 1)
        #
        # levels = ceil(((max_turns + 1) // 2) / max_group_size)

    max_turns = cfg.generation.max_turns
    max_group_size = cfg.generation.max_group_size
    branch_size = cfg.generation.branch_size
    num_levels = math.ceil((max_turns + 1) / 2 / max_group_size)
    if branch_size == 1:
        spo_num_generations = num_levels  # Degenerate case: a single chain of generations
    else:
        spo_num_generations = branch_size * ((branch_size**num_levels - 1) // (branch_size - 1))

    logger.info(
        f"TreeNode-based learning: {num_levels} levels, "
        f"branch_size={branch_size}, "
        f"total maximum nodes={spo_num_generations}"
    )

    #############################################################################
    # Training
    #############################################################################

    trainer = ClassroomSPOTrainer(
    # trainer = ClassroomGDPOTrainer(
        model=model_config.model_name_or_path,
        # reward_funcs=reward_func_list,
        # reward_weights=train_config.reward_weights,
        # NOTE: # Using SPO makes auto num_generations calculation possible, which is more convenient for training with tree nodes.
        args=ClassroomSPOConfig(
        # args=ClassroomGDPOConfig(
            gradient_accumulation_steps=(
                math.ceil(spo_num_generations * cfg.train.number_of_problems_per_batch
                        / cfg.train.per_device_train_batch_size
                        / accelerator.num_processes)
                if cfg.train.top_k_adv is None
                else cfg.train.top_k_adv // cfg.train.per_device_train_batch_size
            ),
            number_of_problems_per_batch=cfg.train.number_of_problems_per_batch,
            num_generations=spo_num_generations,
            gradient_checkpointing=train_config.gradient_checkpointing,
            per_device_train_batch_size=cfg.train.per_device_train_batch_size,
            num_iterations=cfg.train.mu,
            epsilon=cfg.train.epsilon,
            beta=cfg.train.beta,
            learning_rate=cfg.train.learning_rate,
            optim=cfg.train.optimizer,
            bf16=True,
            run_name=cfg.logging.wandb_run_name,
            model_init_kwargs=model_kwargs,
            hub_model_id=cfg.huggingface.name,
            hub_private_repo=False,
            report_to=["wandb"] if logging_config.wandb else [],
            save_strategy="steps",
            lr_scheduler_type=train_config.lr_scheduler_type,
            num_train_epochs=train_config.epochs,
            max_steps=train_config.max_steps,
            max_completion_length=model_config.vllm.max_length,
            logging_steps=1,
            save_steps=cfg.logging.save_steps,
            save_on_each_node=False,
            save_only_model=False,
            save_total_limit=3,
            output_dir=cfg.logging.save_dir,
            max_grad_norm=1.0,
            temperature=cfg.teacher_model.vllm.temperature,
            vllm_server_port=cfg.generation.server_port,
            use_experimental_shared_memory=cfg.generation.use_experimental_shared_memory,
            batch_size_reference_model=cfg.train.batch_size_ref_model,
            save_policy_to_disk_every_n_steps=cfg.train.save_policy_to_disk_every_n,
            peft_config=peft_config,
            top_k_adv=cfg.train.top_k_adv,
            normalize_tree_advantages=cfg.train.normalize_tree_advantages,
            reward_list=train_config.reward_list,
            reward_weights=train_config.reward_weights,
        ),
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    last_ckpt = None
    if os.path.isdir(cfg.logging.save_dir):
        last_ckpt = get_last_checkpoint(cfg.logging.save_dir)
        logger.info(f"Last checkpoint: {last_ckpt}")

    logger.info("Training...")
    train_results = trainer.train(resume_from_checkpoint=last_ckpt)
    logger.info("Training complete!")
    logger.info(train_results)

    trainer.model.config.use_cache = True
    trainer.save_model(logging_config.save_dir + "/model")

    if cfg.huggingface.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
