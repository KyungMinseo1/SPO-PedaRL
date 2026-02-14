import os
import hydra
import wandb
import warnings
from typing import List
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from dotenv import load_dotenv

from src.classroom import Classroom, JudgeDecision, Conversation
from utils.data import load_datasets
from config.eval import EvalConfig
from src.utils.utils import init_logger

load_dotenv()
logger = init_logger()
cs = ConfigStore.instance()
cs.store(name="config", node=EvalConfig)
warnings.filterwarnings("ignore")

def check_judge_decision(checklist_name: str, problem_num: int, times_num: int, conversations: List[Conversation]):
    """
    Function 

    Args:
        checklist_name (str): Name of checklist (One of keys in judge's decision)
        problem_num (int): Number of problems for student model to solve
        times_num (int): Number of trys for each problem
        conversations (List[Conversation]): Actual conversations between student & teacher model
    """
    overall_decisions = []
    for i in range(problem_num):
        current_decisions = []
        for j in range(times_num):
            decisions = [
                d.decision
                for d in conversations[
                    i * times_num + j
                ].judge_decisions[checklist_name]
            ]
            current_decisions.append(
                decisions.count(JudgeDecision.REJECT) / len(decisions)
            )
        overall_decisions.append(sum(current_decisions) / len(current_decisions))

    overall_mean = sum(overall_decisions) / len(overall_decisions)
    print(f"{checklist_name.upper()} mean: {overall_mean}")

    return overall_mean

@hydra.main(config_path="config/eval", version_base=None)
def main(cfg: EvalConfig):
    # Merge loaded config with defaults
    default_config = OmegaConf.structured(EvalConfig)
    cfg = OmegaConf.merge(default_config, cfg)

    # Initialize wandb logging if enabled in the config
    if hasattr(cfg, "logging") and cfg.logging.get("wandb", False):
        wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.logging.wandb_run_name,
            entity=cfg.logging.wandb_entity,
            group=cfg.logging.run_group,
            tags=cfg.logging.wandb_tags,
            config=OmegaConf.to_object(cfg),
        )
        logger.info("Initialized wandb logging.")

    logger.info("Loading evaluation data and constructing the Classroom instance...")

    # Instantiate the Classroom for evaluation
    classroom = Classroom(
        cfg.student_model,
        cfg.teacher_model,
        cfg.judge_model,
        cfg.reward_model,
        cfg.generation,
        None,
    )

    # Load evaluation datasets
    _, eval_data = load_datasets(cfg.dataset, cfg.seed)
    print(eval_data)

    _problems_we_sample = eval_data["problem"]
    _answers_we_sample = eval_data["answer"]

    number_of_times_to_average = cfg.num_samples_per_problem

    problem_we_sample = []
    answer_we_sample = []
    for i in range(len(_problems_we_sample)):
        problem_we_sample.extend([_problems_we_sample[i]] * number_of_times_to_average)
        answer_we_sample.extend([_answers_we_sample[i]] * number_of_times_to_average)

    logger.info("Sampling conversations...")
    conversations = classroom.sample_conversations(
        problem_we_sample,
        answer_we_sample,
        compute_initial_attempt=cfg.recompute_initial_attempts,
    )

    logger.info("Computing metrics...")

    if cfg.recompute_initial_attempts:
        # Compute reward deltas across conversations
        deltas = []
        for i in range(len(_problems_we_sample)):
            current_deltas = []
            for j in range(number_of_times_to_average):
                current_deltas.append(
                    conversations[
                        i * number_of_times_to_average + j
                    # ].get_end_rm_reward()
                    ].get_accuracy_reward() # Post-tutoring student solution accuracy
                    - conversations[ # type: ignore
                        i * number_of_times_to_average + j
                    ].get_initial_rm_reward() # Pre-tutoring student solution accuracy
                )
            deltas.append(sum(current_deltas) / len(current_deltas))
        delta_mean = sum(deltas) / len(deltas)
        print(f"Delta mean: {delta_mean}")

        # Mean before
        initial_rm_rewards = []
        for i in range(len(_problems_we_sample)):
            current_rewards = []
            for j in range(number_of_times_to_average):
                current_rewards.append(
                    conversations[
                        i * number_of_times_to_average + j
                    ].get_initial_rm_reward() # Pre-tutoring student solution accuracy
                )
            initial_rm_rewards.append(sum(current_rewards) / len(current_rewards))
        initial_rm_mean = sum(initial_rm_rewards) / len(initial_rm_rewards)
        print(f"Initial RM mean: {initial_rm_mean}")

    # Mean after
    # end_rm_rewards = []
    # for i in range(len(_problems_we_sample)):
    #     current_rewards = []
    #     for j in range(number_of_times_to_average):
    #         current_rewards.append(
    #             conversations[i * number_of_times_to_average + j].get_end_rm_reward()
    #         )
    #     end_rm_rewards.append(sum(current_rewards) / len(current_rewards))
    # end_rm_mean = sum(end_rm_rewards) / len(end_rm_rewards)
    accuracy_rewards = []
    for i in range(len(_problems_we_sample)):
        current_rewards = []
        for j in range(number_of_times_to_average):
            current_rewards.append(
                conversations[i * number_of_times_to_average + j].get_accuracy_reward() # Post-tutoring student solution accuracy
            )
        accuracy_rewards.append(sum(current_rewards) / len(current_rewards))
    accuracy_reward_mean = sum(accuracy_rewards) / len(accuracy_rewards)
    print(f"Accuracy Reward mean: {accuracy_reward_mean}")

    problem_num = len(_problems_we_sample)
    leaked_answer_mean = check_judge_decision('answer', problem_num, number_of_times_to_average, conversations)
    leaked_solution_process_mean = check_judge_decision('solution_process', problem_num, number_of_times_to_average, conversations)
    does_not_follow_scaffolding_mean = check_judge_decision('scaffolding', problem_num, number_of_times_to_average, conversations)
    does_not_follow_relevance_mean = check_judge_decision('relevance', problem_num, number_of_times_to_average, conversations)
    # does_not_follow_application_mean = check_judge_decision('application', problem_num, number_of_times_to_average, conversations)

    df_table = classroom.to_pd_latest()

    if cfg.score_using_pedagogical_reward:
        from utils.pedagogical_reward import score_each_conversation

        import gc, torch

        del classroom.student_model
        del classroom.teacher_model
        del classroom.judge_model

        gc.collect()
        torch.cuda.empty_cache()

        scores = score_each_conversation(df_table, cfg.pedagogical_reward_model)

        # Scores is a list of lists
        df_table["pedagogical_reward"] = scores
        # We compute the mean pedagogical reward
        pedagogical_rewards = [
            sum([float(s) for s in score]) / len(score) if len(score) != 0 else 0
            for score in scores
        ]
        pedagogical_reward_macro_avg = sum(pedagogical_rewards) / len(
            pedagogical_rewards
        )
        print(f"Pedagogical reward mean macro avg: {pedagogical_reward_macro_avg}")

        # Micro average
        pedagogical_reward_micro_avg = sum(
            [float(s) for score in scores for s in score]
        ) / sum([len(score) for score in scores])
        print(f"Pedagogical reward mean micro avg: {pedagogical_reward_micro_avg}")

    # Log metrics to wandb if enabled
    if hasattr(cfg, "logging") and cfg.logging.get("wandb", False):
        wandb.log(
            {
                "delta_mean": delta_mean if cfg.recompute_initial_attempts else 0,
                "initial_rm_rewards_mean": (
                    initial_rm_mean if cfg.recompute_initial_attempts else 0
                ),
                "accuracy_rewards_mean": accuracy_reward_mean,
                "leaked_answers_mean": leaked_answer_mean,
                "leaked_solution_process_mean": leaked_solution_process_mean,
                "rejects_scaffolding_mean": does_not_follow_scaffolding_mean,
                "rejects_relevance_mean": does_not_follow_relevance_mean,
                # "rejects_application_mean": does_not_follow_application_mean,
                "pedagogical_reward_macro_avg": pedagogical_reward_macro_avg,
                "pedagogical_reward_micro_avg": pedagogical_reward_micro_avg,
            }
        )

        # rewards = [classroom.get_end_rm_reward(c) for c in conversations]
        # df_table["end_rm_reward"] = rewards
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

        # sum of all rewards
        df_table["total_reward"] = (
            # df_table["end_rm_reward"]
            df_table["accuracy_reward"]
            + df_table["pedagogical_alignment_reward"]
            + df_table["thinking_reward"]
            + df_table["end_of_conversation_reward"]
            + df_table["length_reward"]
        )
        df_table = df_table.astype(str)
        print(df_table)
        if cfg.logging.wandb:
            wandb.log({"conversations": wandb.Table(dataframe=df_table)})
        wandb.finish()
    os._exit(0)


if __name__ == "__main__":
    main()
