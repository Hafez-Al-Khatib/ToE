import torch
import torch.nn as nn
from exp_lane_b_reasoning import FilteredMLPEnergyModel, PUZZLES, train_ebm, evaluate, build_kan, cell_mask_quadrant

def run_diagnostic():
    device = torch.device('cuda')
    # Use a LARGE filter bank (16 filters, size 7) to see if it fixes Checker for KAN
    kan = build_kan(n_filters=16).to(device)
    # Update filters to size 7 manually or just use default 5 but more of them
    
    gen_fn, K_colors = PUZZLES['checker']
    train_data = gen_fn(2000, K=K_colors).to(device)
    test_data = gen_fn(200, K=K_colors).to(device)
    
    print("Training KAN with 16 filters on CHECKER...")
    train_ebm(kan, train_data, 30, 0.3, tag='KAN-16F')
    
    kan.eval()
    quad_mask_fn = cell_mask_quadrant
    K_list = [0, 1, 2, 5, 10, 20]
    
    res_cell, res_puz = evaluate(kan, test_data, K_list, K_colors, quad_mask_fn, n_eval=200)
    
    print("\nDIAGNOSTIC RESULTS (KAN + 16 Filters):")
    for k in K_list:
        print(f"K={k:>2}: Cell Acc {res_cell[k]*100:.1f}% | Puz Acc {res_puz[k]*100:.1f}%")

if __name__ == '__main__':
    run_diagnostic()
