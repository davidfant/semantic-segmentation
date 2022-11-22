import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import torch
import wandb
import yaml
from tabulate import tabulate
from torch import distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from val import evaluate

from semseg.augmentations import get_train_augmentation, get_val_augmentation
from semseg.datasets import *
from semseg.losses import get_loss
from semseg.models import *
from semseg.optimizers import get_optimizer
from semseg.schedulers import get_scheduler
from semseg.utils.utils import cleanup_ddp, fix_seeds, setup_cudnn, setup_ddp


class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main(cfg, gpu, save_dir):
    start = time.time()
    best_mIoU = 0.0
    num_workers = mp.cpu_count()
    device = torch.device(cfg['DEVICE'])
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']

    traintransform = get_train_augmentation(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])
    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])

    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train', traintransform)
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', valtransform)

    model = eval(model_cfg['NAME'])(model_cfg['BACKBONE'], trainset.n_classes)
    weights_file = save_dir / f"{cfg['INIT_EPOCH']}_{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}.pth"
    if os.path.exists(weights_file):
        model.load_state_dict(torch.load(weights_file))
        print(f"[*] Resuming from {weights_file}")
    else:
        model.init_pretrained(model_cfg['PRETRAINED'])
    model = model.to(device)

    if train_cfg['DDP']:
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        model = DDP(model, device_ids=[gpu])
    else:
        sampler = RandomSampler(trainset)

    trainloader = DataLoader(
        trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True, pin_memory=True, sampler=sampler)
    valloader = DataLoader(valset, batch_size=1, num_workers=num_workers, pin_memory=True)

    iters_per_epoch = len(trainset) // train_cfg['BATCH_SIZE']
    # class_weights = trainset.class_weights.to(device)
    loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, None)
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, epochs * iters_per_epoch, sched_cfg['POWER'], iters_per_epoch *
                              sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])
    scheduler.last_epoch = max((cfg['INIT_EPOCH']-1)*iters_per_epoch, -1)
    scaler = GradScaler(enabled=train_cfg['AMP'])
    writer = SummaryWriter(str(Path(wandb.run.dir) / 'logs'))

    for epoch in range(cfg['INIT_EPOCH'], epochs):
        model.train()
        if train_cfg['DDP']:
            sampler.set_epoch(epoch)

        train_loss = AverageMeter()
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}]")

        for iter, (img, lbl) in pbar:
            optimizer.zero_grad(set_to_none=True)

            img = img.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)

            with autocast(enabled=train_cfg['AMP']):
                logits = model(img)
                loss = loss_fn(logits, lbl)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss.update(loss.detach_(), img.size(0))
            if not iter % 20:
                lr = scheduler.get_lr()
                lr = sum(lr) / len(lr)
                info = {'lr': lr, 'Train loss': train_loss.avg.item()}
                pbar.set_postfix(info)
                writer.add_scalar('train/loss', info['Train loss'], epoch*iters_per_epoch+iter)
                wandb.log(info, step=epoch*iters_per_epoch+iter)

        wandb.log({'epoch': epoch}, step=epoch*iters_per_epoch+iter)
        torch.cuda.empty_cache()

        if (epoch+1) % train_cfg['EVAL_INTERVAL'] == 0 or (epoch+1) == epochs:
            miou = evaluate(model, valloader, loss_fn, device, epoch*iters_per_epoch+iter)[-1]
            writer.add_scalar('val/mIoU', miou, epoch)

            if miou > best_mIoU:
                best_mIoU = miou
            print(f"Current mIoU: {miou} Best mIoU: {best_mIoU}")

            torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(
            ), save_dir / f"{epoch+1}_{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}.pth")

    writer.close()
    pbar.close()
    end = time.gmtime(time.time() - start)

    table = [
        ['Best mIoU', f"{best_mIoU:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    print(tabulate(table, numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/custom.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    wandb.init(project="human-parsing", id=cfg.get('WANDB_ID'), config=cfg)
    fix_seeds(3407)
    setup_cudnn()
    gpu = setup_ddp()
    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)
    main(cfg, gpu, save_dir)
    cleanup_ddp()
