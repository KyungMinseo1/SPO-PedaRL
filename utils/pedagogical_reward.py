import gc, torch
from vllm import LLM
from vllm.config import PoolerConfig
import pandas as pd

SYSTEM_PROMPT = (
    "Judge the pedagogical quality of the responses provided by two teachers. Focus on the quality of the "
    "scaffolding guidance, correctness, and actionability of the feedback through nudges, questions "
    "and hints. Do not give high scores for revealing the full answer."
)


def format_conversation(problem, reference_solution, conv, response: str) -> list:
    """
    Formats the conversation for the Pedagogical reward model.
    Args:
        problem (str): The problem statement.
        reference_solution (str): The reference solution to the problem.
        conv (list): The conversation history as a list of dictionaries with 'role' and 'content'.
        response (str): The final response from the teacher.
    Returns:
        list: A formatted conversation list suitable for the reward model.
    """
    conversation = []

    # Add system prompt
    conversation.append({"role": "system", "content": SYSTEM_PROMPT})
    conversation.append(
        {
            "role": "user",
            "content": "Problem: "
            + problem
            + "\nReference Solution: "
            + reference_solution,
        }
    )

    # Add the dialog history
    for entry in conv:
        role = "assistant" if entry["role"] in ["Teacher", "Tutor"] else "user"
        conversation.append({"role": role, "content": entry["content"]})

    # Add the final response
    conversation.append({"role": "assistant", "content": response})
    return conversation


def get_reward_inputs(entry, only_last_message: bool = False):
    """
    Extracts from the entry and makes a list of prompts for the reward model for each assistant message
    in the conversation.
    Args:
        entry (dict): A dictionary containing the problem and conversation.
        only_last_message (bool): If True, only returns the last message for scoring.
    Returns:
        list: A list of formatted prompts for the reward model.
    """
    problem = entry["Problem"]
    conversation = entry["Conversation from student perspective"]
    conversation = [
        {
            "role": "Student" if m["role"] == "assistant" else "Teacher",
            "content": m["content"],
        }
        for m in conversation
    ]

    number_of_student_messages = len(
        [m for m in conversation if m["role"] == "Student"]
    )

    prompts = []

    for student_message_count in range(0, number_of_student_messages + 1):
        first_part = []

        student_messages_so_far = 0
        for i, m in enumerate(conversation):
            if student_messages_so_far == student_message_count:
                break
            if m["role"] == "Student":
                student_messages_so_far += 1
            first_part.append(m)

        if len(first_part) == len(conversation):
            continue
        part_to_score = conversation[len(first_part)]
        if part_to_score["role"] == "Student":
            continue

        prompt = format_conversation(
            problem=problem,
            reference_solution="",
            conv=first_part,
            response=part_to_score["content"],
        )
        prompts.append(prompt)

    if only_last_message:
        prompts = [prompts[-1]]

    return prompts


def score_each_conversation(
    df,
    reward_model_path: str,
    only_last_message: bool = False,
    gpu_memory_utilization: float = 0.5,
    max_model_len: int = 12000,
):
    """
    Scores each conversation in the DataFrame using the specified reward model.
    Args:
        df (pd.DataFrame): DataFrame containing the conversations to score.
        reward_model_path (str): Path to the reward model.
        only_last_message (bool): If True, only scores the last message in each conversation.
        gpu_memory_utilization (float): GPU memory utilization for the model.
        max_model_len (int): Maximum model length for the reward model.
    Returns:
        list: A list of scores for each conversation.
    """
    reward_model = LLM(
        reward_model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        convert="classify",
        dtype="float32",
        pooler_config=PoolerConfig(
            pooling_type="LAST",
            normalize=False,
            use_activation=False
        ),
    )

    try:
        tokenizer = reward_model.get_tokenizer()
        prompt_sets = [
            get_reward_inputs(df.iloc[i].to_dict(), only_last_message)
            for i in range(len(df))
        ]

        # We flatten and apply chat template
        prompts = []
        for p_set in prompt_sets:
            prompts.extend(
                [tokenizer.apply_chat_template(p, tokenize=False) for p in p_set]
            )

        # We get all the scores
        outputs = reward_model.encode(prompts, pooling_task="classify")
        rewards = []
        for x in outputs:
            data = x.outputs.data
            # data가 텐서면 .item(), 아니면 그대로 사용
            score = data.item() if hasattr(data, "item") else data
            
            # 혹시 리스트로 감싸져 있는 경우(예: [0.99]) 처리
            if isinstance(score, list):
                score = score[0]
                
            rewards.append(score)

        # We group the scores by conversation
        scores = []
        for i, p_set in enumerate(prompt_sets):
            scores.append(rewards[: len(p_set)])
            rewards = rewards[len(p_set) :]

        return scores
    finally:
        # Clean up vLLM model to free GPU memory
        del reward_model
        gc.collect()
        torch.cuda.empty_cache()
