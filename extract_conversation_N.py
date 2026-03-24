"""
extract_conversation.py

Standalone script that samples teacher-student conversations using the same
Classroom / vLLM infrastructure as the RL training loop, but writes raw
conversations to disk without computing any rewards or running any judge.

Usage:
    python extract_conversation_N.py --config-name=7b_2GPU_extract

Config files live in  config/extract_conversation/.
"""

import os
import json
import hashlib
import warnings

import hydra
import wandb
import pandas as pd
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from dotenv import load_dotenv
from datasets import Dataset
from transformers import set_seed

from config.extract_conversation_model import ConversationExtractionConfig
from src.classroom_branch_N import Classroom
from src.utils.utils import init_logger
from utils.data import load_whole_datasets

warnings.filterwarnings("ignore")
load_dotenv()

logger = init_logger()

cs = ConfigStore.instance()
cs.store(name="config", node=ConversationExtractionConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attach_sample_id(example: dict) -> dict:
    for key in ["id", "problem_id", "question_id", "uid", "uuid"]:
        if key in example and example[key] is not None:
            return {"__sample_id": str(example[key])}
    problem = str(example.get("problem", ""))
    answer = str(example.get("answer", ""))
    sample_id = hashlib.sha1(
        f"{problem}\n<SEP>\n{answer}".encode("utf-8")
    ).hexdigest()
    return {"__sample_id": sample_id}


def _smean(lst: list):
    return sum(lst) / len(lst) if lst else None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="config/extract_conversation", version_base=None)
def main(cfg: ConversationExtractionConfig):

    # ── Merge defaults ──────────────────────────────────────────────────────
    default_config = OmegaConf.structured(ConversationExtractionConfig)
    cfg = OmegaConf.merge(default_config, cfg)

    extraction_cfg = cfg.extraction
    logging_cfg = cfg.logging
    data_cfg = cfg.dataset

    set_seed(cfg.seed)

    # ── wandb ───────────────────────────────────────────────────────────────
    if logging_cfg.wandb:
        wandb.init(
            project=logging_cfg.wandb_project,
            name=logging_cfg.wandb_run_name,
            entity=getattr(logging_cfg, "wandb_entity", None),
            group=logging_cfg.run_group,
            tags=logging_cfg.wandb_tags,
            config=OmegaConf.to_object(cfg),
        )

    # ── Dataset ─────────────────────────────────────────────────────────────
    logger.info(f"Loading dataset...")
    train_dataset, _ = load_whole_datasets(data_cfg, cfg.seed)
    logger.info(f"Loaded {len(train_dataset)} examples")

    if data_cfg.lower_bound_solve_rate is not None:
        train_dataset = train_dataset.filter(
            lambda x: x.get("llama8b_solve_rate", 1.0) >= data_cfg.lower_bound_solve_rate
        )
        logger.info(
            f"{len(train_dataset)} examples after solve-rate filter "
            f"(>= {data_cfg.lower_bound_solve_rate})"
        )

    train_dataset: Dataset = train_dataset.map(
        _attach_sample_id, num_proc=4, desc="Attaching sample IDs"
    )

    # Apply max_train_examples cap (deterministic: take first N after shuffle)
    if data_cfg.max_train_examples is not None and data_cfg.max_train_examples > 0:
        train_dataset = train_dataset.select(
            range(min(data_cfg.max_train_examples, len(train_dataset)))
        )
        logger.info(f"Capped dataset at {len(train_dataset)} examples")

    # ── Output file ─────────────────────────────────────────────────────────
    os.makedirs(logging_cfg.save_dir, exist_ok=True)
    output_path = os.path.join(logging_cfg.save_dir, extraction_cfg.output_filename)
    turn_output_path = os.path.join(logging_cfg.save_dir, extraction_cfg.turn_output_filename)
    logger.info(f"Conversations will be written to: {output_path}")
    logger.info(f"Turns will be written to: {turn_output_path}")

    # ── Classroom initialisation ────────────────────────────────────────────
    # • judge_model  — configured as use_openrouter=True so no GPU is reserved;
    #                  active_rewards=[] ensures its run_batch() is never called.
    # • reward_model — model_name_or_path="None" so Classroom skips its init.
    classroom = Classroom(
        student_model_cfg=cfg.student_model,
        teacher_model_cfg=cfg.teacher_model,
        judge_model_cfg=cfg.judge_model,
        reward_model_cfg=cfg.reward_model,
        generation_cfg=cfg.generation,
        model_save_path=os.path.join(logging_cfg.save_dir, "policy"),
        log_file_path=None,
    )

    # ── Sampling loop ────────────────────────────────────────────────────────
    total_written = 0
    batch_num = 0
    n_problems = extraction_cfg.number_of_problems_per_batch
    n_rollouts = extraction_cfg.num_rollouts_per_problem
    max_conversations = extraction_cfg.max_conversations  # -1 = unlimited

    for batch_start in range(0, len(train_dataset), n_problems):
        if max_conversations > 0 and total_written >= max_conversations:
            logger.info(
                f"Reached max_conversations={max_conversations}. Stopping."
            )
            break

        batch = train_dataset[batch_start : batch_start + n_problems]

        # Accommodate both list-type and Arrow-backed batch formats
        problems_raw: list = list(batch["problem"])
        answers_raw: list = [str(a) for a in batch["answer"]]
        solve_rates_raw: list = [
            float(sr) for sr in batch.get("llama8b_solve_rate", [0.5] * len(problems_raw))
        ]
        # Assign a stable global index (base offset within the full dataset)
        base_idx = batch_start

        # Expand each problem into num_rollouts_per_problem independent rollouts
        problems: list = []
        answers: list = []
        solve_rates: list = []
        problem_idxs: list = []

        for local_i, (prob, ans, sr) in enumerate(
            zip(problems_raw, answers_raw, solve_rates_raw)
        ):
            global_idx = base_idx + local_i
            for _ in range(n_rollouts):
                problems.append(prob)
                answers.append(ans)
                solve_rates.append(sr)
                problem_idxs.append(global_idx)

        logger.info(
            f"[Batch {batch_num}] {len(problems_raw)} problems × "
            f"{n_rollouts} rollouts = {len(problems)} conversations"
        )

        # ── Sample conversations (no judges, no rewards, no solutions) ──────
        # active_rewards=[] skips all judge evaluation steps inside
        # sample_conversations.  number_judge_attempts > 0 in generation config
        # keeps conversation states at JUDGE_TURN (not GENERATE_SOLUTION),
        # so no student solution generation is triggered either.
        conversations = classroom.sample_conversations(
            problems=problems,
            answers=answers,
            solve_rates=solve_rates,
            problem_idx=problem_idxs,
            active_rewards=[],   # skip judges + rewards
        )

        # ── Serialise & persist ─────────────────────────────────────────────
        conversation_rows_for_wandb = []

        with open(output_path, "a", encoding="utf-8") as f_out:
            for conv in conversations:
                if max_conversations > 0 and total_written >= max_conversations:
                    break

                ended_with_eoc = any(
                    "<end_of_conversation>" in msg["content"]
                    for msg in conv.conversation
                    if msg["role"] == "teacher"
                )

                record = {
                    "problem_idx": conv.problem_idx,
                    "problem": conv.problem,
                    "answer": conv.answer,
                    "conversation_id": conv.conversation_id,
                    "student_name": conv.student_name,
                    "student_persona": conv.student_persona,
                    "teacher_prompt": conv.system_prompt_teacher,
                    "num_teacher_turns": conv.teacher_turns,
                    "ended_with_eoc": ended_with_eoc,
                    "conversation": conv.conversation,
                }

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1

                conversation_rows_for_wandb.append(
                    {
                        "problem_idx": conv.problem_idx,
                        "problem": conv.problem,
                        "answer": conv.answer,
                        "conversation_id": conv.conversation_id,
                        "student_name": conv.student_name,
                        "student_persona": conv.student_persona,
                        "teacher_prompt": conv.system_prompt_teacher,
                        "conversation": conv.conversation,
                        "num_teacher_turns": conv.teacher_turns,
                        "ended_with_eoc": ended_with_eoc,
                        "num_messages": len(conv.conversation),
                    }
                )

        turn_rows_for_wandb = []

        with open(turn_output_path, "a", encoding="utf-8") as f_turn_out:
            # Build turn pairs list
            for conv in conversations:
                total_turn_pairs = []
                for turn_idx in sorted(conv.turn_pairs.keys()):
                    for tp in conv.turn_pairs[turn_idx]:
                        total_turn_pairs.append(
                            {
                                "problem_idx": conv.problem_idx,
                                "problem": conv.problem,
                                "answer": conv.answer,
                                "conversation_id": conv.conversation_id,
                                "teacher_turn": tp.teacher_turn,
                                "is_main_turn": tp.is_main_turn,
                                "student_message": tp.student_message,
                                "teacher_message": tp.teacher_message,
                                "next_student_message": tp.next_student_message,
                            }
                        )
            
                f_turn_out.write(json.dumps(total_turn_pairs, ensure_ascii=False) + "\n")

                for total_turn_pair in total_turn_pairs:
                    turn_rows_for_wandb.append(
                        {
                            "problem_idx": conv.problem_idx,
                            "problem": conv.problem,
                            "answer": conv.answer,
                            "conversation_id": conv.conversation_id,
                            "teacher_turn": total_turn_pair["teacher_turn"],
                            "is_main_turn": total_turn_pair["is_main_turn"],
                            "student_message": total_turn_pair["student_message"],
                            "teacher_message": total_turn_pair["teacher_message"],
                            "next_student_message": total_turn_pair["next_student_message"],
                        }
                    )


        # ── wandb logging ───────────────────────────────────────────────────
        if logging_cfg.wandb and conversation_rows_for_wandb:
            conversation_df = pd.DataFrame(conversation_rows_for_wandb)
            turn_df = pd.DataFrame(turn_rows_for_wandb)
            wandb.log(
                {
                    f"batch_{batch_num}": wandb.Table(dataframe=conversation_df.astype(str)),
                    "total_conversations": total_written,
                    "batch": batch_num,
                    "avg_teacher_turns": _smean(
                        [r["num_teacher_turns"] for r in conversation_rows_for_wandb]
                    ),
                    "eoc_rate": _smean(
                        [float(r["ended_with_eoc"]) for r in conversation_rows_for_wandb]
                    ),
                }
            )
            wandb.log(
                {
                    f"turns_batch_{batch_num}": wandb.Table(dataframe=turn_df.astype(str)),
                }
            )

        batch_num += 1
        logger.info(
            f"[Batch {batch_num - 1}] Done. Total conversations written: {total_written}"
        )

    logger.info(
        f"Extraction complete. {total_written} conversations saved to {output_path}"
    )

    if logging_cfg.wandb:
        wandb.log({"final_total_conversations": total_written})
        wandb.finish()


if __name__ == "__main__":
    main()
