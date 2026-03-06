from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers.trainer_utils import get_last_checkpoint
import os
import hydra
import wandb
import torch
import math
import json
import hashlib
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
from src.grpo.config_branch import ClassroomBranchConfig

# from src.grpo.trainer import ClassroomGRPOTrainer
# from src.gdpo.trainer import ClassroomGDPOTrainer
from src.grpo.trainer_branch import ClassroomBranchTrainer

from src.utils.utils import (
    init_logger,
)
import warnings

from utils.data import load_datasets, load_whole_datasets

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

    def load_sample_ids(path: str | None) -> set[str]:
        if path is None or path == "":
            return set()

        if not os.path.exists(path):
            logger.info(f"Sample ID file not found, skipping exclude: {path}")
            return set()

        try:
            if path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                return {str(sample_id) for sample_id in loaded}

            with open(path, "r", encoding="utf-8") as f:
                return {line.strip() for line in f if line.strip() != ""}
        except Exception as e:
            raise ValueError(f"Failed to load sample ids from {path}: {e}")

    def save_sample_ids(path: str | None, sample_ids: list[str]) -> None:
        if path is None or path == "":
            return

        save_dir = os.path.dirname(path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        existing_ids = load_sample_ids(path)
        merged_ids = existing_ids.union({str(sample_id) for sample_id in sample_ids})

        if path.endswith(".json"):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(sorted(merged_ids), f, ensure_ascii=False, indent=2)
        else:
            with open(path, "w", encoding="utf-8") as f:
                for sample_id in sorted(merged_ids):
                    f.write(sample_id + "\n")

    def attach_sample_id(example):
        for key in ["id", "problem_id", "question_id", "uid", "uuid"]:
            if key in example and example[key] is not None:
                return {"__sample_id": str(example[key])}

        problem = str(example.get("problem", ""))
        answer = str(example.get("answer", ""))
        sample_id = hashlib.sha1(f"{problem}\n<SEP>\n{answer}".encode("utf-8")).hexdigest()
        return {"__sample_id": sample_id}

    logger.info(f"Loading datasets from {data_config.train_datasets}")
    # train_dataset, _ = load_datasets(data_config, cfg.seed)
    train_dataset, _ = load_whole_datasets(data_config, cfg.seed)
    logger.info(f"Loaded {len(train_dataset)} training examples")

    if data_config.lower_bound_solve_rate is not None:
        logger.info(f"Filtering training dataset with solve rate lower bound: {data_config.lower_bound_solve_rate}")
        train_dataset = train_dataset.filter(lambda x: x["llama8b_solve_rate"] >= data_config.lower_bound_solve_rate)
        logger.info(f"{len(train_dataset)} training examples remaining after filtering with solve rate lower bound of {data_config.lower_bound_solve_rate}")

    train_dataset: Dataset = train_dataset.map(
        attach_sample_id, num_proc=4, desc="Attaching sample IDs"
    )

    excluded_sample_ids = load_sample_ids(data_config.exclude_sample_ids_path)
    if len(excluded_sample_ids) > 0:
        before_exclude = len(train_dataset)
        logger.info(
            f"Excluding {len(excluded_sample_ids)} used sample IDs from {data_config.exclude_sample_ids_path}"
        )
        train_dataset = train_dataset.filter(
            lambda x: x["__sample_id"] not in excluded_sample_ids,
            desc="Excluding used sample IDs",
        )
        logger.info(
            f"{len(train_dataset)} training examples remaining after exclusion (removed {before_exclude - len(train_dataset)})"
        )

    max_train_examples = data_config.max_train_examples
    skip_first_samples = max(0, cfg.skip_first_samples)

    if max_train_examples is None or max_train_examples == -1:
        start_index = min(skip_first_samples, len(train_dataset))
        end_index = len(train_dataset)
    else:
        start_index = min(skip_first_samples, len(train_dataset))
        end_index = min(len(train_dataset), skip_first_samples + max_train_examples)

    if start_index >= end_index:
        raise ValueError(
            "No training examples left after applying skip_first_samples/max_train_examples. "
            f"dataset_size={len(train_dataset)}, skip_first_samples={skip_first_samples}, "
            f"max_train_examples={max_train_examples}"
        )

    train_dataset = train_dataset.select(range(start_index, end_index))
    logger.info(
        f"Selected training examples in range [{start_index}, {end_index}) -> {len(train_dataset)} examples"
    )

    selected_sample_ids = train_dataset["__sample_id"]
    save_sample_ids(data_config.save_selected_sample_ids_path, selected_sample_ids)
    if data_config.save_selected_sample_ids_path is not None and data_config.save_selected_sample_ids_path != "":
        logger.info(
            f"Saved {len(selected_sample_ids)} selected sample IDs to {data_config.save_selected_sample_ids_path}"
        )

    train_dataset = train_dataset.remove_columns("__sample_id")

    def apply_template(example):
        problem = example["problem"]
        answer = example["answer"]
        solve_rates = example["llama8b_solve_rate"]

        return {"prompt": problem, "answer": answer, "solve_rates": solve_rates}

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
    # Training
    #############################################################################

    trainer = ClassroomBranchTrainer(
    # trainer = ClassroomGDPOTrainer(
        model=model_config.model_name_or_path,
        # reward_funcs=reward_func_list,
        # reward_weights=train_config.reward_weights,
        args=ClassroomBranchConfig(
        # args=ClassroomGDPOConfig(
            gradient_accumulation_steps=cfg.train.num_samples_per_problem
            * cfg.train.number_of_problems_per_batch
            // cfg.train.per_device_train_batch_size
            // accelerator.num_processes,
            gradient_checkpointing=train_config.gradient_checkpointing,
            num_generations=cfg.train.num_samples_per_problem,
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
            is_think_turn_reward=cfg.generation.convert_think_to_turn_reward,
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
