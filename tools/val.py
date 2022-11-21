import torch
import torchvision.transforms as T
import argparse
import yaml
import math
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F
import wandb

from semseg.models import *
from semseg.datasets import *
from semseg.augmentations import get_val_augmentation
from semseg.metrics import Metrics
from semseg.utils.utils import setup_cudnn


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    print('Evaluating...')
    model.eval()
    metrics = Metrics(dataloader.dataset.n_classes, dataloader.dataset.ignore_label, device)
    palette = torch.tensor([[120, 120, 120], [127, 0, 0], [254, 0, 0], [0, 84, 0], [169, 0, 50], [254, 84, 0], [255, 0, 84], [0, 118, 220], [84, 84, 0], [0, 84, 84], [84, 50, 0], [51, 85, 127], [0, 127, 0], [0, 0, 254], [50, 169, 220], [0, 254, 254], [84, 254, 169], [169, 254, 84], [254, 254, 0], [254, 169, 0], [102, 254, 0], [182, 255, 0]])
    loop = tqdm(dataloader, leave = True, position = 0)
    val_loss, l_miou, l_macc, l_mf1 = [], [], [], []
    l_images, l_targets, l_predictions = [], [], []
    for idx, (images, labels) in enumerate(loop):
        
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images)
        loss = loss_fn(preds, labels)
        val_loss.append(loss.item())
        metrics.update(preds, labels)
        ious, miou = metrics.compute_iou()
        l_miou.append(miou)
        acc, macc = metrics.compute_pixel_acc()
        l_macc.append(macc)
        f1, mf1 = metrics.compute_f1()
        l_mf1.append(mf1)

        if idx % 200 == 0:
            for i, j, k in zip(images, labels, preds):
                seg_map = k.softmax(dim=0).argmax(dim=0).cpu().to(int)
                seg_image = palette[seg_map].squeeze()
                label = palette[j].squeeze()
                img = i.squeeze()

                inv_normalize = T.Normalize(
                    mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225),
                    std=(1/0.229, 1/0.224, 1/0.225)
                )

                image = inv_normalize(img)
                image *= 255
                #images = torch.vstack([image, labels])
                l_images.append(wandb.Image(image.to(torch.uint8).cpu().numpy().transpose((1,2,0))))
                l_targets.append(wandb.Image(label.to(torch.uint8).cpu().numpy()))
                l_predictions.append(wandb.Image(seg_image.to(torch.uint8).cpu().numpy()))

    idx_loss = sum(val_loss)/len(val_loss)
    t_miou = sum(l_miou)/len(l_miou)
    t_macc = sum(l_macc)/len(l_macc)
    t_mf1 = sum(l_mf1)/len(l_mf1)

    wandb.log({'Valid loss': idx_loss, 'Valid MIoU': t_miou, 'Valid MAcc': t_macc, 'Valid MF1': t_mf1})
    wandb.log({'Image': l_images, 'Target': l_targets, 'Prediction': l_predictions})

    return acc, macc, f1, mf1, ious, miou

@torch.no_grad()
def evaluate_train(model, dataloader, device):
    print('Evaluating training epoch...')
    model.eval()
    metrics = Metrics(dataloader.dataset.n_classes, dataloader.dataset.ignore_label, device)
    palette = torch.tensor([[120, 120, 120], [127, 0, 0], [254, 0, 0], [0, 84, 0], [169, 0, 50], [254, 84, 0], [255, 0, 84], [0, 118, 220], [84, 84, 0], [0, 84, 84], [84, 50, 0], [51, 85, 127], [0, 127, 0], [0, 0, 254], [50, 169, 220], [0, 254, 254], [84, 254, 169], [169, 254, 84], [254, 254, 0], [254, 169, 0], [102, 254, 0], [182, 255, 0]])

    loop = tqdm(dataloader, leave = True, position = 0)

    for idx, (images, labels) in enumerate(loop):
        
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images)
        metrics.update(preds, labels)
        
    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
                           
    return acc, macc, f1, mf1, ious, miou



@torch.no_grad()
def evaluate_msf(model, dataloader, device, scales, flip):
    model.eval()

    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)

    for images, labels in tqdm(dataloader):
        labels = labels.to(device)
        B, H, W = labels.shape
        scaled_logits = torch.zeros(B, n_classes, H, W).to(device)

        for scale in scales:
            new_H, new_W = int(scale * H), int(scale * W)
            new_H, new_W = int(math.ceil(new_H / 32)) * 32, int(math.ceil(new_W / 32)) * 32
            scaled_images = F.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=True)
            scaled_images = scaled_images.to(device)
            logits = model(scaled_images)
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
            scaled_logits += logits.softmax(dim=1)

            if flip:
                scaled_images = torch.flip(scaled_images, dims=(3,))
                logits = model(scaled_images)
                logits = torch.flip(logits, dims=(3,))
                logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
                scaled_logits += logits.softmax(dim=1)

        metrics.update(scaled_logits, labels)
    
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    ious, miou = metrics.compute_iou()
    return acc, macc, f1, mf1, ious, miou


def main(cfg):
    device = torch.device(cfg['DEVICE'])

    eval_cfg = cfg['EVAL']
    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'val', transform)
    dataloader = DataLoader(dataset, 1, num_workers=1, pin_memory=True)

    model_path = Path(eval_cfg['MODEL_PATH'])
    if not model_path.exists(): model_path = Path(cfg['SAVE_DIR']) / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['BACKBONE']}_{cfg['DATASET']['NAME']}.pth"
    print(f"Evaluating {model_path}...")

    model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], dataset.n_classes)
    model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
    model = model.to(device)

    if eval_cfg['MSF']['ENABLE']:
        acc, macc, f1, mf1, ious, miou = evaluate_msf(model, dataloader, device, eval_cfg['MSF']['SCALES'], eval_cfg['MSF']['FLIP'])
    else:
        acc, macc, f1, mf1, ious, miou = evaluate(model, dataloader, device)

    table = {
        'Class': list(dataset.CLASSES) + ['Mean'],
        'IoU': ious + [miou],
        'F1': f1 + [mf1],
        'Acc': acc + [macc]
    }

    print(tabulate(table, headers='keys'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/custom.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    main(cfg)