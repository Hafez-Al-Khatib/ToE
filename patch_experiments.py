"""
Patch script to update B-spline gradient flows and SR constraints in standalone experiments.
"""
import glob
from pathlib import Path

# 1. B-Spline gradient smoothing patch
target = "bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)"
replacement = """
        temp = 100.0
        bases = torch.sigmoid(temp * (x - grid[:, :-1])) * torch.sigmoid(temp * (grid[:, 1:] - x))
""".strip("\n")

count = 0
for file in glob.glob("experiments/exp_*.py"):
    path = Path(file)
    text = path.read_text("utf-8")
    if target in text:
        text = text.replace(target, replacement)
        path.write_text(text, "utf-8")
        count += 1
        print(f"Patched splines in {file}")

# 2. Add SR constraint to Multitask Script
multitask = Path("experiments/exp_multitask.py")
text = multitask.read_text("utf-8")

if "def corrupt_superres(" in text and "anchor_mask = None" in text:
    print("Found Super-Resolution logic, patching...")
    sr_patch_old = """
def corrupt_superres(x_clean, scale=4):
    \"\"\"
    Downsample by `scale` then bilinear upsample back.
    Simulates 4× super-resolution task.
    anchor_mask = None (no pixel anchoring, conditioning via input)
    \"\"\"
    B, C, H, W = x_clean.shape
    lr = F.avg_pool2d(x_clean, scale)                                # (B,C,H/s,W/s)
    x_bicubic = F.interpolate(lr, size=(H, W), mode='bilinear',
                              align_corners=False)                    # (B,C,H,W)
    return x_bicubic.clamp(-2., 2.), None, None
"""
    sr_patch_new = """
def corrupt_superres(x_clean, scale=4):
    \"\"\"
    Downsample by `scale` then bilinear upsample back.
    Simulates 4× super-resolution task.
    anchor_mask = None for exact pixel replacement, but we provide scale and lr as a tuple in anchor_vals
    to signal the denoiser to apply the global data consistency constraint.
    \"\"\"
    B, C, H, W = x_clean.shape
    lr = F.avg_pool2d(x_clean, scale)                                # (B,C,H/s,W/s)
    x_bicubic = F.interpolate(lr, size=(H, W), mode='bilinear',
                              align_corners=False)                    # (B,C,H,W)
    
    # We pass the LR array via anchor_vals to enforce it dynamically
    # Use anchor_mask = "SR" to trigger the logic.
    return x_bicubic.clamp(-2., 2.), "SR", lr
"""
    text = text.replace(sr_patch_old.strip('\n'), sr_patch_new.strip('\n'))

    denoise_old = """
    def denoise(self, x, n_steps=10, dt=0.05, dt_decay=0.97,
                anchor_mask=None, anchor_vals=None):
"""
    denoise_new = """
    def denoise(self, x, n_steps=10, dt=0.05, dt_decay=0.97,
                anchor_mask=None, anchor_vals=None):
"""
    
    denoise_logic_old = """
            if anchor_mask is not None:
                u = torch.where(anchor_mask, anchor_vals, u)
"""
    denoise_logic_new = """
            if anchor_mask is not None:
                if type(anchor_mask) is str and anchor_mask == "SR":
                    scale = u.shape[-1] // anchor_vals.shape[-1]
                    # Data consistency constraint: force downsampled reconstruction to match LR
                    u_lr = F.avg_pool2d(u, scale)
                    residual = F.interpolate(anchor_vals - u_lr, size=u.shape[-2:], mode='nearest')
                    u = u + residual  # inject missed LR signal back
                else:
                    u = torch.where(anchor_mask, anchor_vals, u)
"""
    text = text.replace(denoise_logic_old.strip('\n'), denoise_logic_new.strip('\n'))
    multitask.write_text(text, "utf-8")
    print("Patched SR in exp_multitask.py")

print("Done patching.")
