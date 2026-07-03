# experiments/train_group_kan.py
"""Train GroupKAN energy heads under the EXACT recipe used for the
ConvSmoothMLP variants (rebuttal_smooth_mlp.py): AdamW lr=3e-4 wd=1e-4,
CosineAnnealingLR T_max=epochs, 40 epochs, 20,000-image CIFAR-10 train
subset, batch 128, sigma_train=0.15, grad-clip 1.0, AMP on CUDA.
No per-architecture tuning (spec section 3).

Run: py -3.12 experiments/train_group_kan.py --device cuda
Smoke: py -3.12 experiments/train_group_kan.py --device cpu --quick
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from group_kan import GroupKANEnergyModel

OUT = ROOT / 'outputs' / 'group_kan'
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = {'group_kan_8k': 136, 'group_kan_32k': 640}
SIGMA_TRAIN = 0.15


def train_one(model, loader, device, n_epochs, tag):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    history = []
    for ep in range(1, n_epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for batch in loader:
            x = batch[0].to(device)
            opt.zero_grad()
            if use_amp:
                with torch.amp.autocast('cuda'):
                    loss = model.loss(x, SIGMA_TRAIN)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss = model.loss(x, SIGMA_TRAIN)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        history.append(tot / nb)
        if ep <= 3 or ep % 5 == 0 or ep == n_epochs:
            print(f"  [{tag}] ep {ep:>3d}/{n_epochs}  loss={tot / nb:.5f}")
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--n_train', type=int, default=20000)
    p.add_argument('--quick', action='store_true',
                   help='smoke: 2 epochs, 1500 images')
    args = p.parse_args()
    if args.quick:
        args.epochs, args.n_train = 2, 1500
    device = torch.device(args.device)

    tf = T.Compose([T.ToTensor(),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ds = torchvision.datasets.CIFAR10(ROOT / 'data', train=True,
                                      download=True, transform=tf)
    subset = torch.utils.data.Subset(ds, list(range(args.n_train)))
    loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True,
                                         num_workers=0, pin_memory=True,
                                         drop_last=True)

    log = {'recipe': {'lr': 3e-4, 'wd': 1e-4, 'epochs': args.epochs,
                      'n_train': args.n_train, 'batch': 128,
                      'sigma_train': SIGMA_TRAIN, 'clip': 1.0},
           'variants': {}}
    for tag, hidden in VARIANTS.items():
        t0 = time.time()
        model = GroupKANEnergyModel(hidden=hidden).to(device)
        print(f"[train] {tag}: hidden={hidden}, {model.n_params:,} params")
        history = train_one(model, loader, device, args.epochs, tag)
        path = OUT / f"{tag}.pt"
        torch.save(model.state_dict(), path)
        log['variants'][tag] = {'hidden': hidden, 'n_params': model.n_params,
                                'final_loss': history[-1],
                                'seconds': time.time() - t0,
                                'checkpoint': str(path)}
        print(f"[saved] {path}")
    (OUT / 'training_log.json').write_text(json.dumps(log, indent=2))


if __name__ == '__main__':
    main()
