"""
Distributed training utilities for PyTorch DDP migration.
"""

import os
import torch as ch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed(rank, world_size, backend='nccl'):
    """
    Initialize the distributed environment.
    
    Args:
        rank (int): Rank of the current process
        world_size (int): Total number of processes
        backend (str): Communication backend ('nccl' for GPU, 'gloo' for CPU)
    """
    # Set environment variables for master node
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # Set the GPU device for this process (1 GPU per rank)
    if ch.cuda.is_available() and backend == 'nccl':
        ch.cuda.set_device(rank % ch.cuda.device_count())
    
    # Initialize the process group (env:// uses MASTER_ADDR/MASTER_PORT)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    # Tiny all_reduce sanity check (catches mis-mapping immediately)
    if ch.cuda.is_available() and backend == 'nccl':
        t = ch.tensor([rank], device=ch.cuda.current_device())
        dist.all_reduce(t)
        if rank == 0:
            print(f"[DDP] all_reduce ok: sum={t.item()} expected={sum(range(world_size))} on device {ch.cuda.current_device()}")


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
