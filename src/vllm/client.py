import requests
from typing import List
from vllm import RequestOutput, CompletionOutput
from transformers import PreTrainedTokenizer

####################################################################################################
# The following functions are used to interact with the vLLM server
# - For sampling conversations
# - For getting rewards
####################################################################################################


def sample_conversations(
    problems: List[str],
    answers: List[str],
    meta: dict = {},
    server_port: int = 8000,
    num_samples_per_problem: int = 1,
    tokenizer: PreTrainedTokenizer = None,
):
    """
    Returns:
        request_outputs: List[RequestOutput]
        turn_pair_advantages_list: List[List[Dict]]  # NEW! Each conversation has a list of turn pair advantages
    """

    server_url = f"http://localhost:{server_port}/sample_conversations"
    
    # NOTE: We should expand trees based on one problems sample -> discard repeating problems and answers for num_samples_per_problem times.
    actual_problems = problems
    answers = [str(answer) for answer in answers]
    response = requests.post(
        server_url, json={"problems": actual_problems, "meta": meta, "answers": answers}
    )
    response.raise_for_status()

    response_dict = response.json()
    conversations = response_dict["conversations"]

    request_outputs = []
    turn_pair_advantages_list = []  # NEW!

    for conv_data in conversations:
        messages = conv_data["messages"]
        turn_pair_advs = conv_data.get("turn_pair_advantages", [])  # NEW!
        
        request_output = RequestOutput(
            request_id=conv_data["conversation_id"],
            prompt="",
            outputs=[
                CompletionOutput(
                    index=0,
                    text=tokenizer.apply_chat_template(messages, tokenize=False),
                    token_ids=tokenizer.apply_chat_template(messages, tokenize=True),
                    cumulative_logprob=0.0,
                    logprobs=[],
                )
            ],
            prompt_token_ids=[],
            prompt_logprobs=[],
            finished=True,
        )
        request_outputs.append(request_output)
        turn_pair_advantages_list.append(turn_pair_advs)  # NEW!
    
    return request_outputs, turn_pair_advantages_list  # CHANGED!


def get_end_rm_reward(
    conversations: List[str],
    server_port: int = 8000,
) -> List[float]:
    server_url = f"http://localhost:{server_port}/get_end_rm_reward"

    response = requests.post(server_url, json={"conversations": conversations})
    response.raise_for_status()

    rewards = response.json()
    return rewards

def get_accuracy_reward(
    conversations: List[str],
    server_port: int = 8000,
) -> List[float]:
    server_url = f"http://localhost:{server_port}/get_accuracy_reward"

    response = requests.post(server_url, json={"conversations": conversations})
    response.raise_for_status()

    rewards = response.json()
    return rewards

def get_pedagogical_alignment_reward(
    conversations: List[str],
    server_port: int = 8000,
) -> List[float]:
    server_url = f"http://localhost:{server_port}/get_pedagogical_alignment_reward"

    response = requests.post(server_url, json={"conversations": conversations})
    response.raise_for_status()

    rewards = response.json()
    return rewards


def get_thinking_reward(
    conversations: List[str],
    server_port: int = 8000,
) -> List[float]:
    server_url = f"http://localhost:{server_port}/get_thinking_reward"

    response = requests.post(server_url, json={"conversations": conversations})
    response.raise_for_status()

    rewards = response.json()
    return rewards


def get_end_of_conversation_reward(
    conversations: List[str],
    server_port: int = 8000,
) -> List[float]:
    server_url = f"http://localhost:{server_port}/get_end_of_conversation_reward"

    response = requests.post(server_url, json={"conversations": conversations})
    response.raise_for_status()

    rewards = response.json()
    return rewards


def get_length_reward(
    conversations: List[str],
    server_port: int = 8000,
) -> List[float]:
    server_url = f"http://localhost:{server_port}/get_length_reward"

    response = requests.post(server_url, json={"conversations": conversations})
    response.raise_for_status()

    rewards = response.json()
    return rewards


####################################################################################################


def wait_batch(server_port: int = 8000):
    """
    Sends a request to the FastAPI server's /wait_batch endpoint.
    """
    server_url = f"http://localhost:{server_port}/wait_batch"

    response = requests.get(server_url)
    response.raise_for_status()

    return response.json()