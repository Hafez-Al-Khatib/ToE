import glob
from pathlib import Path

# Reverse the exact replacement string
replacement = "bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)"
target = """
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
        print(f"Reverted splines in {file}")

print("Done patching.")
