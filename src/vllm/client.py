import requests
from typing import List
from vllm import RequestOutput, CompletionOutput
from transformers import PreTrainedTokenizer

####################################################################################################
# The following functions are used to interact with the vLLM server
# - For sampling conversations
# - For getting rewards
####################################################################################################


def sample_nodes(
    problems: List[str],
    answers: List[str],
    meta: dict = {},
    server_port: int = 8000,
    tokenizer: PreTrainedTokenizer = None,
):
    """
    TreeNode 단위로 샘플링 (NEW!)
    
    Returns:
        request_outputs: List[RequestOutput] - 각 TreeNode = 1 sequence
        node_advantages_list: List[List[Dict]] - 각 node의 turn pair advantages
    """

    server_url = f"http://localhost:{server_port}/sample_nodes"
    
    response = requests.post(
        server_url, json={"problems": problems, "meta": meta, "answers": answers}
    )
    response.raise_for_status()

    response_dict = response.json()
    nodes = response_dict["nodes"]

    request_outputs = []
    node_advantages_list = []

    for node_data in nodes:
        # Context 없음! 오직 현재 노드의 turn pairs만!
        node_turn_pairs = node_data["node_turn_pairs"]
        
        # 이 노드의 메시지만 구성 (context 제외)
        messages = []
        for turn in node_turn_pairs:
            messages.append(turn["student_message"])
            messages.append(turn["teacher_message"])
        
        # Tokenize
        sequence_text = tokenizer.apply_chat_template(messages, tokenize=False)
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True)
        
        request_output = RequestOutput(
            request_id=f"node_{node_data['node_id']}",
            prompt="",
            outputs=[
                CompletionOutput(
                    index=0,
                    text=sequence_text,
                    token_ids=token_ids,
                    cumulative_logprob=0.0,
                    logprobs=[],
                )
            ],
            prompt_token_ids=[],
            prompt_logprobs=[],
            finished=True,
        )
        request_outputs.append(request_output)
        node_advantages_list.append(node_turn_pairs)
    
    return request_outputs, node_advantages_list


def sample_conversations(
    problems: List[str],
    answers: List[str],
    meta: dict = {},
    server_port: int = 8000,
    num_samples_per_problem: int = 1,
    tokenizer: PreTrainedTokenizer = None,
):
    """
    DEPRECATED: Use sample_nodes instead.
    Kept for backward compatibility.
    """
    # Redirect to sample_nodes
    return sample_nodes(
        problems=problems,
        answers=answers,
        meta=meta,
        server_port=server_port,
        tokenizer=tokenizer,
    )


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