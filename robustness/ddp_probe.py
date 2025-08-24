import os, torch as t, torch.distributed as dist
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
rank = int(os.environ.get("RANK", 0))

t.cuda.set_device(local_rank)
dist.init_process_group("nccl")
x = t.ones(1, device=local_rank) * (rank + 1)
dist.all_reduce(x)  # sum across ranks
if rank == 0:
    print("WORLD_SIZE:", world_size)
print(f"[rank {rank}] device {local_rank} -> all_reduce result:", x.item())
dist.destroy_process_group()
