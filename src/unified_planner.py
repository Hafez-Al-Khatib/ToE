import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
import numpy as np

from thermodynamic_field import ThermodynamicField
from wave_solver import WaveBrain

class UnifiedPlanner(nn.Module):
    """
    The Grand Unified Planner.
    
    Merges:
    1. Global Planning (WaveBrain): Uses wave physics (Eikonal) to find geodesic paths ($\nabla T$).
    2. Local Action (TNF): Uses energy minimization to execute dynamics, guided by the global drift.
    
    Equation:
    u_{t+1} = u_t - \eta \nabla E(u) - \gamma (v \cdot \nabla u)
              (Local Physics)      (Global Advection)
    
    where v = - \nabla T_{wave} is the desired velocity field.
    """
    def __init__(
        self,
        field_model: ThermodynamicField,
        wave_model: WaveBrain,
        drift_weight: float = 1.0
    ):
        super().__init__()
        self.local_brain = field_model
        self.global_brain = wave_model
        self.drift_weight = drift_weight
    
    def compute_guidance_field(self, u_wave: torch.Tensor) -> torch.Tensor:
        """
        Compute the velocity field v = -∇u_wave.
        
        Args:
            u_wave: (B, H, W) travel time field from WaveBrain.
            
        Returns:
            v: (B, 2, H, W) velocity field (y, x).
               channel 0 is vy (vertical), channel 1 is vx (horizontal).
        """
        # Central difference gradients
        # Pad to handle boundaries
        u_pad = F.pad(u_wave, (1, 1, 1, 1), mode='replicate')
        
        # dy: (u[y+1] - u[y-1]) / 2
        du_dy = (u_pad[:, 2:, 1:-1] - u_pad[:, :-2, 1:-1]) / 2.0
        
        # dx: (u[x+1] - u[x-1]) / 2
        du_dx = (u_pad[:, 1:-1, 2:] - u_pad[:, 1:-1, :-2]) / 2.0
        
        # Velocity is -Gradient (downhill)
        vy = -du_dy
        vx = -du_dx
        
        # Normalize?
        # Physical velocity should be normalized if it represents direction.
        # But Eikonal gradient magnitude |∇u| = n (refractive index).
        # So |v| = n. High cost areas have large gradients.
        # This is actually GOOD: we want stronger drift in high-cost areas to push through?
        # Or do we want normalized direction?
        # Let's normalize to unit vectors to have consistent "push" strength.
        mag = torch.sqrt(vx**2 + vy**2 + 1e-8)
        vy = vy / mag
        vx = vx / mag
        
        v = torch.stack([vy, vx], dim=1) # (B, 2, H, W)
        return v

    def apply_advection(self, state: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Apply Semi-Lagrangian Advection.
        Supports shared flow (B, 2, H, W) or per-channel flow (B, C, 2, H, W).
        """
        B, C, H, W = state.shape
        
        # Handle Per-Channel Flow
        if flow.dim() == 5: # (B, C, 2, H, W)
            # Reshape to (B*C, 1, H, W) for state
            # Reshape flow to (B*C, 2, H, W)
            state_flat = state.view(B*C, 1, H, W)
            flow_flat = flow.view(B*C, 2, H, W)
            
            # Recurse (now flow is 4D)
            out_flat = self.apply_advection(state_flat, flow_flat)
            return out_flat.view(B, C, H, W)
            
        # Standard 4D Flow (B, 2, H, W)
        # Note: If called recursively, B here is actual B*C
        
        # Create base grid (-1 to 1)
        yy = torch.linspace(-1, 1, H, device=state.device)
        xx = torch.linspace(-1, 1, W, device=state.device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
        
        vx = flow[:, 1]
        vy = flow[:, 0]
        
        base_grid_x = grid_x.unsqueeze(0).expand(state.shape[0], -1, -1)
        base_grid_y = grid_y.unsqueeze(0).expand(state.shape[0], -1, -1)
        
        scale_y = 2.0 / H
        scale_x = 2.0 / W
        
        target_x = base_grid_x - vx * scale_x
        target_y = base_grid_y - vy * scale_y
        
        grid = torch.stack([target_x, target_y], dim=3)
        
        # Sample
        advected = F.grid_sample(state, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return advected

    def step(
        self,
        env_state: torch.Tensor,
        current_map: torch.Tensor,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        n_local_steps: int = 5
    ) -> dict:
        """
        Perform one coupled planning-action step using FORCE-BASED dynamics.
        """
        # 1. Global Planning: Get Time-to-Goal
        _, u_wave, _ = self.global_brain(current_map, source_pos=goal)
        
        # 2. Guidance Field v = -∇T (The 'Gravity' toward goal)
        v = self.compute_guidance_field(u_wave)
        
        # 3. Create Advection Fields
        B, C, H, W = env_state.shape
        flow = torch.zeros(B, C, 2, H, W, device=v.device)
        
        # Agent (ch 0) is pulled to goal
        flow[:, 0] = v * self.drift_weight
        
        # Block (ch 1) is pushed AWAY from the agent's density
        # Calculate -∇(agent)
        agent_pad = F.pad(env_state[:, 0:1], (1, 1, 1, 1), mode='replicate')
        da_dy = (agent_pad[:, :, 2:, 1:-1] - agent_pad[:, :, :-2, 1:-1]) / 2.0
        da_dx = (agent_pad[:, :, 1:-1, 2:] - agent_pad[:, :, 1:-1, :-2]) / 2.0
        
        # The block only moves if the agent is close (density > 0.1)
        # Push strength multiplier
        push_strength = 2.0
        flow[:, 1, 0] = -da_dy.squeeze(1) * push_strength
        flow[:, 1, 1] = -da_dx.squeeze(1) * push_strength
        
        # 4. Advection (Operator Split Step 1)
        # We explicitly advect FIRST so the block moves, then we apply TNF reaction
        advected_state = self.apply_advection(env_state, flow)
        
        # 5. Reaction (Operator Split Step 2)
        new_state = self.local_brain.evolve(
            advected_state,
            n_steps=n_local_steps,
            advection_field=None, 
            return_trajectory=False
        )
        
        return {
            "state": new_state,
            "wave_field": u_wave,
            "guidance": v
        }

    def solve(
        self,
        initial_state: torch.Tensor,
        map_func,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        max_macro_steps: int = 50
    ):
        """
        Run the full close-loop solver.
        
        Args:
            initial_state: Starting tensor.
            map_func: Callable(state) -> map_tensor. Extracts the navigation map from state.
            start, goal: coordinates.
            
        Yields:
            state trajectory
        """
        current_state = initial_state
        curr_pos = start
        
        yield current_state
        
        for step in range(max_macro_steps):
            # 1. Extract map from current state (e.g. walls + blocks)
            current_map = map_func(current_state)
            
            # 2. Plan and Act
            # Note: We need to track agent position 'curr_pos' if it changes.
            # But the 'state' contains the agent. The field moves the agent.
            # We don't explicitly update 'curr_pos' here unless we need to re-center something.
            # For Global Planning, we need the GOAL (fixed). 
            # The Guidance Field is computed over the WHOLE grid, so it works everywhere.
            # We don't strictly need 'curr_pos' for the WaveSolve call if we use Goal as source.
            
            result = self.step(
                env_state=current_state,
                current_map=current_map,
                start=curr_pos, # Unused if solving from goal
                goal=goal,
                n_local_steps=5
            )
            
            current_state = result["state"]
            yield current_state
            
            # Update curr_pos? 
            # If the state is a probability distribution, 'curr_pos' is the argmax.
            # Let's assume the user of this generator handles extraction if needed.
