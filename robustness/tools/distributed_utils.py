"""
Distributed training utilities for PyTorch DDP migration.
"""

import os
import torch
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
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # Set the GPU device for this process
    if torch.cuda.is_available() and backend == 'nccl':
        torch.cuda.set_device(rank)


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
