"""
The main file, which exposes the robustness command-line tool, detailed in
:doc:`this walkthrough <../example_usage/cli_usage>`.
"""

from argparse import ArgumentParser
import os
import git
import torch 
import torch.multiprocessing as mp
from robustness.tools.distributed_utils import setup_distributed, cleanup_distributed
import cox
import cox.utils
import cox.store

try:
    from .model_utils import make_and_restore_model
    from .datasets import DATASETS
    from .train import train_model, eval_model
    from .tools import constants, helpers
    from . import defaults, __version__
    from .defaults import check_and_fill_args
except:
    raise ValueError("Make sure to run with python -m (see README.md)")


parser = ArgumentParser()
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.DISTRIBUTED_ARGS, parser)

def main(args, store=None):
    '''Given arguments from `setup_args` and a store from `setup_store`,
    trains as a model. Check out the argparse object in this file for
    argument options.
    '''
    # MAKE DATASET AND LOADERS
    data_path = os.path.expandvars(args.data)
    dataset = DATASETS[args.dataset](data_path)

    train_loader, val_loader = dataset.make_loaders(args.workers,
                    args.batch_size, data_aug=bool(args.data_aug))

    train_loader = helpers.DataPrefetcher(train_loader)
    val_loader = helpers.DataPrefetcher(val_loader)
    loaders = (train_loader, val_loader)

    # MAKE MODEL
    model, checkpoint = make_and_restore_model(arch=args.arch,
            dataset=dataset, resume_path=args.resume)
    if 'module' in dir(model): model = model.module

    print(args)
    if args.eval_only:
        return eval_model(args, model, val_loader, store=store)

    if not args.resume_optimizer: checkpoint = None
    # Check if running in distributed mode
    distributed = getattr(args, 'distributed', False)
    
    # Pass distributed parameters to loaders
    if distributed:
        rank = getattr(args, 'rank', 0)
        world_size = getattr(args, 'world_size', 1)
        train_loader, val_loader = dataset.make_loaders(
            args.workers, args.batch_size, data_aug=bool(args.data_aug),
            distributed=distributed, rank=rank, world_size=world_size
        )
        train_loader = helpers.DataPrefetcher(train_loader)
        val_loader = helpers.DataPrefetcher(val_loader)
        loaders = (train_loader, val_loader)
    
    model = train_model(args, model, loaders, store=store,
                                    checkpoint=checkpoint, distributed=distributed)
    return model

def setup_args(args):
    '''
    Fill the args object with reasonable defaults from
    :mod:`robustness.defaults`, and also perform a sanity check to make sure no
    args are missing.
    '''
    # override non-None values with optional config_path
    if args.config_path:
        args = cox.utils.override_json(args, args.config_path)

    ds_class = DATASETS[args.dataset]
    args = check_and_fill_args(args, defaults.CONFIG_ARGS, ds_class)

    if not args.eval_only:
        args = check_and_fill_args(args, defaults.TRAINING_ARGS, ds_class)

    if args.adv_train or args.adv_eval:
        args = check_and_fill_args(args, defaults.PGD_ARGS, ds_class)

    args = check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, ds_class)
    if args.eval_only: assert args.resume is not None, \
            "Must provide a resume path if only evaluating"
    return args

def setup_store_with_metadata(args):
    '''
    Sets up a store for training according to the arguments object. See the
    argparse object above for options.
    '''
    # Add git commit to args
    try:
        repo = git.Repo(path=os.path.dirname(os.path.realpath(__file__)),
                            search_parent_directories=True)
        version = repo.head.object.hexsha
    except git.exc.InvalidGitRepositoryError:
        version = __version__
    args.version = version

    # Create the store
    store = cox.store.Store(args.out_dir, args.exp_name)
    args_dict = args.__dict__
    schema = cox.store.schema_from_dict(args_dict)
    store.add_table('metadata', schema)
    store['metadata'].append_row(args_dict)

    return store

# === DDP launcher helpers (moved from distributed_main.py) ===

def run_distributed_training(rank, world_size, args, store_path=None):
    try:
        setup_distributed(rank, world_size)
        store = setup_store_with_metadata(args) if store_path and rank == 0 else None
        args.distributed, args.rank, args.world_size = True, rank, world_size
        if torch.distributed.is_initialized(): torch.distributed.barrier()
        model = main(args, store=store)
        if torch.distributed.is_initialized(): torch.distributed.barrier()
        return model
    except KeyboardInterrupt:
        print(f"Process {rank}: Training interrupted by user"); cleanup_distributed(); return None
    except Exception:
        import traceback; traceback.print_exc(); cleanup_distributed(); raise
    finally:
        cleanup_distributed()

def launch_distributed(args):
    world_size = args.world_size if getattr(args, 'world_size', None) else torch.cuda.device_count()
    if world_size <= 1:
        print("Warning: Only 1 GPU available or specified. Running single-process training.")
        args.distributed, args.rank, args.world_size = False, 0, 1
        store = setup_store_with_metadata(args)
        return main(args, store=store)
    print(f"Launching distributed training on {world_size} GPUs")
    mp.set_start_method('spawn', force=True)
    mp.spawn(run_distributed_training,
             args=(world_size, args, getattr(args, 'out_dir', None)),
             nprocs=world_size, join=True)


if __name__ == "__main__":
    args = parser.parse_args()
    args = cox.utils.Parameters(args.__dict__)

    args = setup_args(args)
    use_ddp = bool(getattr(args, "distributed", False)) or (torch.cuda.device_count() > 1)
    if use_ddp:
        launch_distributed(args)
    else:
        store = setup_store_with_metadata(args)
        final_model = main(args, store=store)
