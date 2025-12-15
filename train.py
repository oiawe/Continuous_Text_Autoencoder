import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from config import TrainConfig
from utils.scheduler import get_cosine_schedule_with_warmup
from utils.load import continue_training, get_model_statedict
from utils.muon import get_muon_optimizer

from models.model import TextVAE, MODEL_CONFIG
from models.dataset import ParquetContentDataset

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fp32_precision = "tf32"

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '6268'
    dist.init_process_group("gloo" if os.name == "nt" else "nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def _init_config(train_config: TrainConfig):
    if not os.path.exists(train_config.model_save_path):
        print(f'Creating {train_config.model_save_path}')
        os.makedirs(train_config.model_save_path, exist_ok=True)

def train(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    train_config = TrainConfig()

    _init_config(train_config)

    model = TextVAE(**MODEL_CONFIG).to(rank)

    if rank == 0:
        print('total param: ', sum(param.numel() for param in model.parameters()) / 1e6)
        print('trainable param: ', sum(param.numel() for param in model.parameters() if param.requires_grad) / 1e6)

    model = DDP(model, device_ids=[rank])
    model.module.compute_loss = torch.compile(model.module.compute_loss, dynamic=False)

    train_dataset = ParquetContentDataset(train_config.dataset_path, train_config.tokenizer_path, chunk_size=train_config.chunk_size)

    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True, drop_last=True)
    train_dataloader = DataLoader(train_dataset, num_workers=train_config.num_workers, pin_memory=True,
                                  persistent_workers=True, sampler=train_sampler,
                                  batch_size=train_config.batch_size, prefetch_factor=8, drop_last=True)

    if rank == 0:
        writer = SummaryWriter(train_config.log_dir)

    optimizer = get_muon_optimizer([model], lr=train_config.learning_rate, wd=0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(train_config.warmup_steps),
                                                num_training_steps=train_config.max_steps)

    # load latest checkpoints if possible
    current_steps = continue_training(train_config.model_save_path, model, optimizer)

    model.train()
    stop = False
    epoch = 0
    grad_norm = 0

    while not stop:  # loop over the train_dataset multiple times
        train_sampler.set_epoch(epoch)
        epoch += 1
        dataloader = train_dataloader
        if rank == 0:
            dataloader = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, tokens in enumerate(dataloader):
            tokens = tokens.to(rank, non_blocking=True)

            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, loss_dict = model.module.compute_loss(tokens)

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            current_steps += 1

            if rank == 0 and batch_idx % train_config.log_interval == 0:
                steps = current_steps
                for key, value in loss_dict.items():
                    writer.add_scalar(f"training/{key}", value.item(), steps)
                writer.add_scalar("training/loss", loss.item(), steps)
                writer.add_scalar("training/grad_norm", grad_norm, steps)
                writer.add_scalar("learning_rate/learning_rate", scheduler.get_last_lr()[0], steps)

            if rank == 0 and current_steps % train_config.save_interval == 0:
                torch.save(get_model_statedict(model),
                           os.path.join(train_config.model_save_path, f'checkpoint_{current_steps}.pt'))
                torch.save(optimizer.state_dict(),
                           os.path.join(train_config.model_save_path, f'optimizer_{current_steps}.pt'))
                print(f"Step {current_steps}, Loss {loss.item()}")

            if current_steps > train_config.max_steps:
                stop = True
                break

    cleanup()


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
