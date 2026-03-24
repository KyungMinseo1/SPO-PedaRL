from dataclasses import dataclass, field
from typing import Optional

from config.train_rl_model import (
    TeacherModelConfig,
    StudentModelConfig,
    JudgeModelConfig,
    RewardModelConfig,
    GenerationConfig,
    DatasetConfig,
    LoggingConfig,
)


@dataclass
class ExtractionSpecificConfig:
    # How many distinct problems to process per sampling batch.
    number_of_problems_per_batch: int = 8

    # How many independent conversation rollouts to sample per problem.
    # Each rollout is a full independent conversation (teacher + student turns).
    num_rollouts_per_problem: int = 4

    # Output filename written inside logging.save_dir.
    output_filename: str = "conversations.jsonl"
    turn_output_filename: str = "turns.jsonl"

    # Hard cap on total conversations written to disk. -1 = no limit.
    max_conversations: int = -1


@dataclass
class ConversationExtractionConfig:
    extraction: ExtractionSpecificConfig = field(
        default_factory=ExtractionSpecificConfig
    )

    teacher_model: TeacherModelConfig = field(default_factory=TeacherModelConfig)
    student_model: StudentModelConfig = field(default_factory=StudentModelConfig)

    # judge_model is required by Classroom.__init__ but is NEVER called during
    # extraction (active_rewards=[] skips all judge steps).
    # Set use_openrouter: true in the yaml so no GPU is allocated.
    judge_model: JudgeModelConfig = field(default_factory=JudgeModelConfig)

    # reward_model is not used during extraction.
    # Set model_name_or_path: "None" in the yaml so Classroom skips its init.
    reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    seed: int = 42
