"""
state_classification.py
-----------------------
Standalone script for collecting pedagogical-stage classification statistics.

Workflow:
1. Load turn-segmented conversations from the dataset JSON.
2. Run batch inference through Classification (directly, no HTTP server needed).
3. Log per-batch running stats and final aggregate statistics to W&B.
4. Save the full result table as a CSV.

Usage:
    python state_classification.py --config-name 7b
    python state_classification.py --config-name 7b dataset.train_datasets.0.ratio=0.1
"""

import logging
import json
import os

import hydra
import pandas as pd
import wandb
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig

load_dotenv()

from config.classification_config import ClassificationConfig
from src.classification import Classification

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="config", node=ClassificationConfig)


@hydra.main(config_path="config/classification", version_base=None)
def main(cfg: DictConfig) -> None:
    # Merge hydra overrides with dataclass defaults
    default_config = OmegaConf.structured(ClassificationConfig)
    cfg = OmegaConf.merge(default_config, cfg)

    if cfg.logging.wandb:
        wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.logging.wandb_run_name,
            entity=cfg.logging.wandb_entity,
            group=cfg.logging.run_group,
            tags=list(cfg.logging.wandb_tags),
            config=OmegaConf.to_object(cfg),
        )


    all_turn_segments = []
    for dataset_cfg in cfg.dataset.train_datasets:
        path = dataset_cfg.name_or_path
        logger.info(f"Loading dataset from: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        seen: dict = {}
        for item in data:
            problem = item.get("Problem")
            if problem not in seen:
                seen[problem] = []
            seen[problem].append(item)

        max_per = dataset_cfg.max_turn_splits_per_problem
        n = max(1, int(len(seen) * dataset_cfg.ratio))
        problems = list(seen.items())[:n]
        logger.info(f"Unique problems: {len(problems)} (max_turn_splits_per_problem={max_per})")
        for _, items in problems:
            for item in (items if max_per < 0 else items[:max_per]):
                all_turn_segments.extend(item["Turn_Split"])

    all_turn_segments = all_turn_segments[cfg.skip_first_samples :]
    logger.info(f"Total turn segments to classify: {len(all_turn_segments)}")

    classifier = Classification(
        classifier_config=cfg.classifier,
        generation_config=cfg.generation,
    )

    all_results = classifier.sample_classification_results(all_turn_segments)

    # ------------------------------------------------------------------ #
    # Final statistics                                                      #
    # ------------------------------------------------------------------ #
    df = pd.DataFrame(all_results)
    logger.info("\n=== Classification Statistics ===\n%s", df["class"].value_counts().to_string())

    if cfg.logging.wandb:
        # Full result table
        wandb.log({"Final/Classification_Table": wandb.Table(dataframe=df.astype(str))})

        # Stage distribution bar chart
        stage_counts_df = df["class"].value_counts().reset_index()
        stage_counts_df.columns = ["stage", "count"]
        wandb.log(
            {
                "Final/Stage_Distribution": wandb.plot.bar(
                    wandb.Table(dataframe=stage_counts_df),
                    "stage",
                    "count",
                    title="Stage Distribution",
                )
            }
        )

        # Confidence distribution
        if "confidence" in df.columns:
            conf_counts_df = df["confidence"].value_counts().reset_index()
            conf_counts_df.columns = ["confidence", "count"]
            wandb.log(
                {
                    "Final/Confidence_Distribution": wandb.plot.bar(
                        wandb.Table(dataframe=conf_counts_df),
                        "confidence",
                        "count",
                        title="Confidence Distribution",
                    )
                }
            )

    # ------------------------------------------------------------------ #
    # Save results                                                          #
    # ------------------------------------------------------------------ #
    os.makedirs(cfg.logging.save_dir, exist_ok=True)
    out_path = os.path.join(cfg.logging.save_dir, "classification_results.csv")
    df.to_csv(out_path, index=False)
    logger.info("Saved classification results to: %s", out_path)

    if cfg.logging.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
