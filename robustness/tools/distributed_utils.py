"""
Distributed training utilities for PyTorch DDP migration.
"""

import os
import torch as ch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(rank: int, world_size: int, backend: str = "nccl", init_method: str = "env://") -> None:
    """
    Initialize the distributed environment for this process.
    Call this BEFORE any CUDA ops (streams, tensors, DataPrefetcher, etc).
    """
    # 1 Pick device deterministically (one GPU per rank) *before* any CUDA call
    if backend == "nccl":
        assert ch.cuda.is_available(), "NCCL backend requires CUDA."
        # use LOCAL_RANK if present (torchrun), otherwise fall back to our spawn rank
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device_id = local_rank % ch.cuda.device_count()
        ch.cuda.set_device(device_id)

    # 2 Initialize the process group.
    #    Do NOT set MASTER_ADDR/MASTER_PORT here; let the launcher/env control that.
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )

    # 3 Fast sanity check to catch bad rank↔GPU mapping
    if backend == "nccl":
        t = ch.tensor([rank], device=ch.cuda.current_device())
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        expected = world_size * (world_size - 1) // 2
        if rank == 0:
            print(f"[DDP] all_reduce ok: sum={t.item()} expected={expected} "
                  f"on device {ch.cuda.current_device()}")



def cleanup_distributed():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_distributed():
    """Check if we're running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    """Get the rank of the current process."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size():
    """Get the total number of processes."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def barrier():
    """Synchronize all processes."""
    if is_distributed():
        dist.barrier()


def reduce_tensor(tensor, average=True):
    """
    Reduce tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        average: If True, compute average; if False, compute sum
    """
    if not is_distributed():
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    
    if average:
        rt /= get_world_size()
    
    return rt
