# experiments/train_stl64.py
"""STL-10 64x64 frontier extension: train the primary pair (panel EIC-W1,
resolution/dataset axis). Pre-registered in .superpowers/sdd/progress.md
BEFORE this run; criterion identical to the CIFAR frontier.

Matched recipe (train_group_kan.py): AdamW lr=3e-4 wd=1e-4, cosine,
40 epochs, 20,000-image STL-10 unlabeled subset at 64x64, batch 128,
sigma_train=0.15, grad-clip 1.0, AMP on CUDA. Training seed 0.

-> outputs/stl64/{kan_32k_stl64,kan_110k_stl64}.pt   (skip-if-done)
Run: py experiments/train_stl64.py --device cuda
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
sys.path.insert(0, str(ROOT / 'experiments'))

from exp_cifar10 import KANEnergyModel

SIGMA_TRAIN = 0.15
VARIANTS = {
    'kan_32k_stl64': dict(n_filters=16, kan_hidden=[48, 16]),
    'kan_110k_stl64': dict(n_filters=32, kan_hidden=[96, 16]),
}


def stl_loader(n_train, seed):
    tf = T.Compose([T.Resize(64), T.ToTensor(),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ds = torchvision.datasets.STL10(ROOT / 'data', split='unlabeled',
                                    download=True, transform=tf)
    subset = torch.utils.data.Subset(ds, list(range(n_train)))
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True,
                                       generator=g, num_workers=2,
                                       pin_memory=True, drop_last=True)


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
        if ep <= 3 or ep % 10 == 0 or ep == n_epochs:
            print(f"  [{tag}] ep {ep:>3d}/{n_epochs}  loss={tot / nb:.5f}")
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--n_train', type=int, default=20000)
    args = p.parse_args()
    device = torch.device(args.device)
    out = ROOT / 'outputs' / 'stl64'
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / 'training_log.json'
    log = json.loads(log_path.read_text()) if log_path.exists() else {}

    loader = None
    for tag, kw in VARIANTS.items():
        path = out / f'{tag}.pt'
        if path.exists():
            print(f'[skip] {tag} already trained')
            continue
        if loader is None:
            loader = stl_loader(args.n_train, seed=0)
        torch.manual_seed(0)
        model = KANEnergyModel(**kw).to(device)
        n_par = sum(q.numel() for q in model.parameters())
        print(f'[train] {tag}: {n_par:,} params')
        t0 = time.time()
        hist = train_one(model, loader, device, args.epochs, tag)
        torch.save(model.state_dict(), path)
        log[tag] = {'n_params': n_par, 'final_loss': hist[-1],
                    'seconds': time.time() - t0, 'checkpoint': str(path)}
        log_path.write_text(json.dumps(log, indent=2))
        print(f'[saved] {path}  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
