"""
Shared memory utilities for loading model weights across processes during training.
Not used during inference-only workloads (e.g. classification stats collection).
"""
from typing import Any, Dict


def load_shared_state_dict(meta: Any) -> Dict:
    """
    Load a model state dict from shared-memory metadata produced by the trainer.

    meta: dict mapping param_name -> (multiprocessing.shared_memory.SharedMemory,
                                       torch.dtype, shape tuple, stride tuple)

    Returns an empty dict if meta is None (inference / stats-only mode).
    """
    if meta is None:
        return {}

    import torch

    state_dict: Dict[str, Any] = {}
    for key, (shm, dtype, shape, stride) in meta.items():
        buf = torch.frombuffer(bytearray(shm.buf), dtype=dtype)
        tensor = buf.reshape(shape)
        state_dict[key] = tensor.clone()
    return state_dict
