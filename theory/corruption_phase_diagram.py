"""
Corruption phase-diagram: WHERE does the K*(severity) ~ C*sev^alpha law hold?
=============================================================================

Theory prediction (THEORY_NOTE.md sec 3): the law holds when the corruption
displaces the signal by an amount scaling as a power of the severity -- true for
stochastic additive-type noise (Gaussian / speckle / Poisson), false for a
deterministic forward operator (blur) or quantization (JPEG).

We run, per corruption type, a fine K-sweep of energy gradient descent from the
corrupted image, find K* = argmax_K mean PSNR, fit log K* vs log(severity), and
report the power-law R^2. High R^2 + positive alpha = "law holds"; flat/low R^2 =
"law breaks". Uses the existing trained checkpoints (no training).
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'theory'))
from measure_kappa import MODELS, grad_E, load_cifar_test, DEVICE  # noqa: E402

OUT = ROOT / 'outputs' / 'theory'
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)

N_IMAGES = 40
K_MAX = 30
USE_MODELS = ['kan_f32', 'conv_mlp_gelu', 'unet']


# ── corruptions (operate on batched x in [-1,1]) ──────────────────────────────
def _to01(x): return (x + 1) / 2
def _to11(x): return x * 2 - 1


def c_gaussian(x, s):
    return x + s * torch.randn_like(x)


def c_speckle(x, s):
    x01 = _to01(x)
    return _to11((x01 * (1 + s * torch.randn_like(x01))).clamp(0, 1))


def c_poisson(x, lam):
    x01 = _to01(x).clamp(0, 1)
    return _to11((torch.poisson(x01 * lam) / lam).clamp(0, 1))


def c_blur(x, sig):
    k = 5
    return gaussian_blur(x, kernel_size=[k, k], sigma=[float(sig), float(sig)])


def c_jpeg(x, q):
    x01 = _to01(x).clamp(0, 1)
    out = torch.empty_like(x01)
    for i in range(x01.shape[0]):
        arr = (x01[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format='JPEG', quality=int(q))
        buf.seek(0)
        dec = np.asarray(Image.open(buf)).astype(np.float32) / 255.0
        out[i] = torch.from_numpy(dec).permute(2, 0, 1)
    return _to11(out.to(x.device))


# each: levels = native parameter, sev = monotone corruption strength for the fit
CORRUPTIONS = {
    'gaussian': dict(fn=c_gaussian, levels=[0.05, 0.08, 0.12, 0.18, 0.25, 0.35],
                     sev=[0.05, 0.08, 0.12, 0.18, 0.25, 0.35]),
    'speckle':  dict(fn=c_speckle, levels=[0.08, 0.13, 0.20, 0.30, 0.45, 0.65],
                     sev=[0.08, 0.13, 0.20, 0.30, 0.45, 0.65]),
    'poisson':  dict(fn=c_poisson, levels=[400, 200, 100, 50, 25, 12],
                     sev=[1/np.sqrt(l) for l in [400, 200, 100, 50, 25, 12]]),
    'blur':     dict(fn=c_blur, levels=[0.4, 0.6, 0.8, 1.1, 1.5, 2.0],
                     sev=[0.4, 0.6, 0.8, 1.1, 1.5, 2.0]),
    'jpeg':     dict(fn=c_jpeg, levels=[80, 60, 45, 30, 20, 12],
                     sev=[100 - q for q in [80, 60, 45, 30, 20, 12]]),
}


def psnr(u, clean):
    mse = F.mse_loss(u.clamp(-1, 1), clean, reduction='none').mean(dim=[1, 2, 3])
    return (10 * torch.log10(4.0 / mse)).mean().item()


def ksweep_kstar(model, corrupted, clean, K_max=K_MAX, dt=0.05, decay=0.97):
    """Return K* = argmax mean-PSNR over K in 0..K_max (0 = no iteration)."""
    u = corrupted.clone()
    step = dt
    series = [psnr(u, clean)]
    for _ in range(K_max):
        # clamp gradient to [-1,1] to match the validated denoise routines
        u = (u - step * grad_E(model, u).clamp(-1., 1.)).detach()
        step *= decay
        series.append(psnr(u, clean))
    return int(np.argmax(series)), series[0], max(series)


def fit_law(sev, kstar):
    sev = np.asarray(sev, float)
    kstar = np.asarray(kstar, float)
    ok = kstar > 0
    if ok.sum() < 4 or np.std(kstar[ok]) < 1e-6:
        return 0.0, 0.0   # flat / no productive iteration => no law
    x, y = np.log(sev[ok]), np.log(kstar[ok])
    slope, intercept = np.polyfit(x, y, 1)
    yh = intercept + slope * x
    r2 = 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(slope), float(r2)


def main():
    print(f"Device: {DEVICE}  | images={N_IMAGES}  K_max={K_MAX}")
    test_ds = load_cifar_test()
    clean = torch.stack([test_ds[i][0] for i in range(N_IMAGES)]).to(DEVICE)

    results = {}
    for mname in USE_MODELS:
        cfg = MODELS[mname]
        if not cfg['ckpt'].exists():
            print(f"[skip] {mname}: checkpoint missing")
            continue
        model = cfg['cls'](**cfg['kw']).to(DEVICE)
        model.load_state_dict(torch.load(cfg['ckpt'], map_location=DEVICE, weights_only=True))
        model.eval()
        print(f"\n=== {mname} ===")
        results[mname] = {}
        for cname, cc in CORRUPTIONS.items():
            torch.manual_seed(0)
            kstars, in_psnrs = [], []
            for lvl in cc['levels']:
                corrupted = cc['fn'](clean, lvl).to(DEVICE)
                ks, p_in, p_best = ksweep_kstar(model, corrupted, clean)
                kstars.append(ks)
                in_psnrs.append(p_in)
            alpha, r2 = fit_law(cc['sev'], kstars)
            verdict = 'LAW' if (r2 >= 0.9 and alpha > 0.3) else ('weak' if r2 >= 0.7 else 'BREAKS')
            results[mname][cname] = dict(kstars=kstars, sev=cc['sev'], in_psnr=in_psnrs,
                                         alpha=alpha, r2=r2, verdict=verdict)
            print(f"  {cname:>9}: K*={kstars}  alpha={alpha:+.2f}  R^2={r2:.3f}  [{verdict}]")
        del model

    (OUT / 'corruption_phase_diagram.json').write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT/'corruption_phase_diagram.json'}")

    # summary table: R^2 per corruption x model
    print("\n=== PHASE DIAGRAM (R^2 of power-law fit) ===")
    cnames = list(CORRUPTIONS.keys())
    print(f"  {'model':>14} | " + " ".join(f"{c:>9}" for c in cnames))
    for mname in results:
        cells = " ".join(f"{results[mname][c]['r2']:>9.3f}" for c in cnames)
        print(f"  {mname:>14} | {cells}")


if __name__ == '__main__':
    main()
