import os
import torch
import torch.distributed as dist


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        # single GPU / no distributed
        args.rank = 0
        args.gpu = 0
        args.world_size = 1
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(
        backend='nccl',
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()


def dist_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
