from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class ModelvLLMConfig:

    # Sampling parameters
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0

    # vLLM settings
    max_length: int = 8192
    max_num_seqs: int = 256

    gpu_memory_utilization: float = 0.5
    number_of_gpus_per_instance: int = 4
    max_number_of_instances: int = -1
    from_0: bool = True

    load_and_unload: bool = True

    bits_and_bytes: bool = False
    use_awq: bool = False
    enable_sleep_mode: bool = True
    use_v0: bool = True
    enforce_eager: bool = False


@dataclass
class ClassifierConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    use_openrouter: bool = False
    use_gemini: bool = False
    vllm: ModelvLLMConfig = field(default_factory=ModelvLLMConfig)

@dataclass
class GenerationConfig:
    classification_prompt_path: str = (
        "prompts/classification_prompts.txt"
    )

    max_tokens_in_conversation: int = 8192
    max_tokens_per_turn: int = 1024
    max_tokens_per_judge_attempt: int = 2048
    tokenizer_to_use: str = "Qwen/Qwen2.5-7B-Instruct"

    # Server settings
    server_port: int = 8005
    use_experimental_shared_memory: bool = False

    n_turns_to_sample: int = 1  # Number of turns we will sample for pedagogical judge reward. Should be <= max_turns.

@dataclass
class Dataset:
    name_or_path: str = "../Wandb/all_conversations_turn_segmented.json"
    ratio: float = 1.0

@dataclass
class DatasetConfig:
    train_datasets: list[Dataset] = field(default_factory=lambda: [Dataset()])

@dataclass
class LoggingConfig:
    wandb: bool = False
    wandb_project: str = "classification"
    wandb_run_name: str = "Qwen2.5-7B-Instruct"
    wandb_entity: Optional[str] = None
    run_group: str = "7b"
    wandb_tags: list[str] = field(default_factory=list)
    save_dir: str = "checkpoints"
    save_steps: int = 10

@dataclass
class ClassificationConfig:
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    skip_first_samples: int = 0

    seed: int = 42