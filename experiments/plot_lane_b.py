import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    res_path = Path('outputs/lane_b/results.json')
    if not res_path.exists():
        print("No Lane B results found.")
        return
    with open(res_path, 'r') as f:
        res = json.load(f)
    
    puzzles = res['config']['puzzles']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, puz in enumerate(puzzles):
        eval_data = res['puzzles'][puz]['eval']['quadrant']
        K_list = sorted([int(k) for k in eval_data['kan_puzzle_acc'].keys()])
        
        kan_acc = [eval_data['kan_puzzle_acc'][str(k)] * 100 for k in K_list]
        mlp_acc = [eval_data['mlp_puzzle_acc'][str(k)] * 100 for k in K_list]
        
        axes[i].plot(K_list, kan_acc, 'o-', label='KAN-EBM', color='blue')
        axes[i].plot(K_list, mlp_acc, 's--', label='MLP-EBM', color='red')
        axes[i].set_title(puz.capitalize())
        axes[i].set_xlabel('Inference Steps K')
        axes[i].set_ylabel('Puzzle Accuracy (%)')
        axes[i].grid(True, alpha=0.3)
        if i == 0: axes[i].legend()
    
    plt.tight_layout()
    out_path = Path('outputs/lane_b/reasoning_scaling.pdf')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.savefig(out_path.with_suffix('.png'), bbox_inches='tight', dpi=150)
    print(f"Saved plot: {out_path}")

if __name__ == '__main__':
    main()
