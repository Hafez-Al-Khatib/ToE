"""
Batch runner for recipe ablation experiments.
Runs all recipes sequentially and saves incrementally.
"""
import subprocess
import sys

RECIPES = [
    'single-sigma', 'multi-sigma', 'wd-0', 'wd-1e-2',
    'epochs-20', 'epochs-80', 'batch-128', 'convmlp-multi'
]

for recipe in RECIPES:
    print(f"\n{'='*60}")
    print(f"Starting {recipe}")
    print(f"{'='*60}")
    cmd = [sys.executable, 'theory/recipe_ablation.py', '--recipes', recipe]
    result = subprocess.run(cmd, cwd=r'C:\Users\hafez\Desktop\AUB Research\ToE')
    if result.returncode != 0:
        print(f"WARNING: {recipe} exited with code {result.returncode}")
    else:
        print(f"{recipe} completed successfully")

print("\nAll recipes completed!")
