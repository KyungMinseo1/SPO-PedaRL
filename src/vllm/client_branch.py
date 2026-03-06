import requests
from typing import List, Tuple, Dict, Optional
from transformers import PreTrainedTokenizer
from src.utils.utils import init_logger

logger = init_logger()

def sample_conversations_branch(
    problems: List[str],
    problem_idxs: List[int],
    answers: List[str],
    solve_rates: List[float],
    meta: dict = {},
    server_port: int = 8000,
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> Tuple[List[List[int]], List[List[Dict]], List[Dict], List[int]]:
    """
    POST /sample_conversations_branch  →  list of full conversations with per-turn advantages.

    The server already groups same-position turn pairs across rollouts and computes GRPO advantages
    (compute_all_advantages_flat).  The client just parses the response and tokenises.

    Only *is_main_turn=True* turn pairs are used to build the training sequence, so the token
    sequence is the main conversation path.  Auxiliary branch turns are recorded in each turn-pair
    dict and can be used for future analysis but are NOT included in the training sequence here.
    """
    server_url = f"http://localhost:{server_port}/sample_conversations_branch"

    answers_str = [str(a) for a in answers]

    response = requests.post(
        server_url,
        json={
            "problems": problems,
            "answers": answers_str,
            "meta": meta,
            "problem_idx": problem_idxs,
            "solve_rates": solve_rates,
        },
        timeout=3600,
    )
    response.raise_for_status()

    conversations_data = response.json()["conversations"]

    token_ids_list: List[List[int]] = []
    turn_pair_adv_list: List[List[Dict]] = []
    conv_rewards_list: List[Dict] = []
    problem_idx_list: List[int] = []

    for conv_data in conversations_data:
        all_turn_pairs: List[Dict] = conv_data.get("turn_pairs", [])
        if not all_turn_pairs:
            continue

        # ── Build training sequence from main-path turns only ────────────────
        main_turns = [turn_pair for turn_pair in all_turn_pairs if turn_pair.get("is_main_turn", True)]
        if not main_turns:
            main_turns = all_turn_pairs  # fallback: treat every turn as main

        messages = []

        if system_prompt := conv_data.get("system_prompt"):
            messages.append({"role": "system", "content": system_prompt})

        for turn_pair in main_turns:
            student_msg = dict(turn_pair["student_message"])
            teacher_msg = dict(turn_pair["teacher_message"])
            student_msg["role"] = "user"
            teacher_msg["role"] = "assistant"
            messages.append(student_msg)
            messages.append(teacher_msg)

        token_ids: List[int] = (
            tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
            if tokenizer is not None
            else []
        )

        token_ids_list.append(token_ids)
        turn_pair_adv_list.append(main_turns)
        conv_rewards_list.append(
            {
                "accuracy_reward": conv_data.get("accuracy_reward"),
                "end_of_conversation_reward": conv_data.get("end_of_conversation_reward"),
                "think_reward": conv_data.get("think_reward"),
            }
        )
        problem_idx_list.append(conv_data["problem_idx"])

    logger.info(
        f"[client_branch] Received {len(token_ids_list)} conversations "
        f"from {len(set(problem_idx_list))} unique problems"
    )
    return token_ids_list, turn_pair_adv_list, conv_rewards_list, problem_idx_list


def wait_batch(server_port: int = 8000) -> dict:
    """Block until the server has finished the current batch."""
    server_url = f"http://localhost:{server_port}/wait_batch"
    response = requests.get(server_url)
    response.raise_for_status()
    return response.json()