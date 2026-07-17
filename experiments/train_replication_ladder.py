# experiments/train_replication_ladder.py
"""Colab batch v2 training (panel findings R1-W4 and scale critique).

Two jobs, one matched recipe (identical to train_group_kan.py:
AdamW lr=3e-4 wd=1e-4, CosineAnnealingLR T_max=epochs, 40 epochs,
20,000-image CIFAR-10 subset, batch 128, sigma_train=0.15, grad-clip 1.0,
AMP on CUDA):

1. REPLICATION: retrain both primary-pair heads (KAN 16/[48,16] ~32K and
   KAN 32/[96,16] ~110K) with training seeds 1 and 2
   -> outputs/replication/{kan_32k,kan_110k}_ts{seed}.pt
   Purpose: is the pre-registered frontier verdict stable under training
   stochasticity, with the 0.3 dB margin vs train-to-train variance?

2. SCALE LADDER: ConvSmoothMLP heads at ~1M / ~8M / ~30M params
   (n_filters=16 fixed backbone; mlp_hidden 1000 / 2800 / 5450), seed 0
   -> outputs/scale_ladder/ladder_{1m,8m,30m}.pt
   Purpose: extend the frontier's parameter axis ~600x beyond 33K.

Skip-if-done per checkpoint. Run: py experiments/train_replication_ladder.py --device cuda
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
from rebuttal_smooth_mlp import ConvSmoothMLPEBM

SIGMA_TRAIN = 0.15

REPLICATION = {  # tag -> (ctor kwargs, training seeds)
    'kan_32k': dict(n_filters=16, kan_hidden=[48, 16]),
    'kan_110k': dict(n_filters=32, kan_hidden=[96, 16]),
}
REP_SEEDS = [1, 2]

LADDER = {  # tag -> mlp_hidden  (n_filters=16 backbone held fixed)
    'ladder_1m': 1000,
    'ladder_8m': 2800,
    'ladder_30m': 5450,
}


def make_loader(n_train, seed):
    tf = T.Compose([T.ToTensor(),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ds = torchvision.datasets.CIFAR10(ROOT / 'data', train=True,
                                      download=True, transform=tf)
    subset = torch.utils.data.Subset(ds, list(range(n_train)))
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True,
                                       generator=g, num_workers=0,
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
    p.add_argument('--quick', action='store_true')
    args = p.parse_args()
    if args.quick:
        args.epochs, args.n_train = 2, 1500
    device = torch.device(args.device)

    rep_dir = ROOT / 'outputs' / 'replication'
    lad_dir = ROOT / 'outputs' / 'scale_ladder'
    rep_dir.mkdir(parents=True, exist_ok=True)
    lad_dir.mkdir(parents=True, exist_ok=True)
    log_path = rep_dir / 'training_log.json'
    log = json.loads(log_path.read_text()) if log_path.exists() else {}

    jobs = []
    for tag, kw in REPLICATION.items():
        for seed in REP_SEEDS:
            jobs.append(('rep', f'{tag}_ts{seed}',
                         rep_dir / f'{tag}_ts{seed}.pt', kw, seed))
    for tag, hidden in LADDER.items():
        jobs.append(('ladder', tag, lad_dir / f'{tag}.pt', hidden, 0))

    for kind, name, path, spec, seed in jobs:
        if path.exists():
            print(f"[skip] {name} already trained")
            continue
        torch.manual_seed(seed)
        if kind == 'rep':
            model = KANEnergyModel(**spec).to(device)
        else:
            model = ConvSmoothMLPEBM(n_filters=16, mlp_hidden=spec,
                                     activation='gelu').to(device)
        n_par = sum(p_.numel() for p_ in model.parameters())
        print(f"[train] {name}: {n_par:,} params (seed {seed})")
        t0 = time.time()
        hist = train_one(model, make_loader(args.n_train, seed),
                         device, args.epochs, name)
        torch.save(model.state_dict(), path)
        log[name] = {'n_params': n_par, 'seed': seed,
                     'final_loss': hist[-1], 'seconds': time.time() - t0,
                     'checkpoint': str(path)}
        log_path.write_text(json.dumps(log, indent=2))
        print(f"[saved] {path}  ({time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()
