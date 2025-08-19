"""
Distributed training launcher for robustness library using DDP.
"""

import os
import sys
import torch
import torch.multiprocessing as mp
from argparse import ArgumentParser

try:
    from ..main import main, setup_args, setup_store_with_metadata, parser as base_parser
    from ..tools.distributed_utils import setup_distributed, cleanup_distributed
    from .. import defaults
except ImportError:
    # Handle running as script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from robustness.main import main, setup_args, setup_store_with_metadata, parser as base_parser
    from robustness.tools.distributed_utils import setup_distributed, cleanup_distributed
    from robustness import defaults


def run_distributed_training(rank, world_size, args, store_path=None):
    """
    Run training on a single process for distributed training.
    
    Args:
        rank (int): Process rank
        world_size (int): Total number of processes
        args: Training arguments
        store_path (str): Path to store for logging
    """
    try:
        # Setup distributed environment
        setup_distributed(rank, world_size)
        
        # Set up store for this process (only rank 0 will actually write)
        if store_path and rank == 0:
            store = setup_store_with_metadata(args)
        else:
            store = None
        
        # Add distributed training flags to args
        args.distributed = True
        args.rank = rank
        args.world_size = world_size
        
        # Ensure all processes start together
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        # Run training
        model = main(args, store=store)
        
        # Ensure all processes finish together
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        return model
        
    except KeyboardInterrupt:
        print(f"Process {rank}: Training interrupted by user")
        cleanup_distributed()
        return None
    except Exception as e:
        print(f"Error in process {rank}: {e}")
        import traceback
        traceback.print_exc()
        cleanup_distributed()
        raise
    finally:
        cleanup_distributed()


def launch_distributed(args):
    """
    Launch distributed training using torch.multiprocessing.
    
    Args:
        args: Parsed arguments object
    """
    # Determine number of GPUs
    if hasattr(args, 'world_size') and args.world_size:
        world_size = args.world_size
    else:
        world_size = torch.cuda.device_count()
    
    if world_size <= 1:
        print("Warning: Only 1 GPU available or specified. Running single-process training.")
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        store = setup_store_with_metadata(args)
        return main(args, store=store)
    
    print(f"Launching distributed training on {world_size} GPUs")
    
    # Use spawn method for better CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
    # Launch processes
    try:
        mp.spawn(
            run_distributed_training,
            args=(world_size, args, args.out_dir if hasattr(args, 'out_dir') else None),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        print(f"Distributed training failed: {e}")
        cleanup_distributed()
        raise


# # Create distributed-aware argument parser
# no need, already seteles in defaults.py
# def create_distributed_parser():
#     """Create argument parser with distributed training options."""
#     parser = ArgumentParser(parents=[base_parser], add_help=False)
    
#     # Add distributed-specific arguments
#     parser.add_argument('--distributed', action='store_true',
#                        help='Use distributed training (DDP)')
#     parser.add_argument('--world-size', type=int, default=None,
#                        help='Number of processes for distributed training (default: number of GPUs)')
#     parser.add_argument('--dist-backend', type=str, default='nccl',
#                        help='Distributed backend (nccl or gloo)')
#     parser.add_argument('--dist-url', type=str, default='env://',
#                        help='URL for distributed training coordination')
    
#     return parser


if __name__ == "__main__":
    # Create distributed parser
    dist_parser = create_distributed_parser()
    args = dist_parser.parse_args()
    
    # Convert to Parameters object
    import cox.utils
    args = cox.utils.Parameters(args.__dict__)
    
    # Setup args with defaults
    args = setup_args(args)
    
    # Launch training
    if args.distributed or torch.cuda.device_count() > 1:
        launch_distributed(args)
    else:
        # Single GPU fallback
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        store = setup_store_with_metadata(args)
        main(args, store=store)
