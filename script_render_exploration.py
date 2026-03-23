#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import sys

sys.path.append(".")

from lib.grid_env.obst_grid_agent_expl_env import ObstGridAgentExplEnv
from lib.grid_env.stepper_wrapper_class import stepper_wrapper_class


def render_exploration_grids(env, title_suffix=""):
    """
    Render both the true grid and the observed grid for an exploration environment.
    
    Parameters:
    -----------
    env : ObstGridAgentExplEnv
        The exploration environment to render
    title_suffix : str
        Additional title suffix for disambiguation
    """
    fig, (ax_true, ax_obs) = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    
    height, width = env.grid.shape
    
    # ===== TRUE GRID (LEFT) =====
    ax_true.imshow(env.grid, cmap='Greys', origin='upper', vmin=0, vmax=255)
    
    # Add grid lines for better visibility
    for i in np.arange(-0.5, height, 1):
        ax_true.axhline(y=i, color='gray', linewidth=0.5, alpha=0.3)
    for j in np.arange(-0.5, width, 1):
        ax_true.axvline(x=j, color='gray', linewidth=0.5, alpha=0.3)
    
    # Plot agent on true grid
    ax_true.scatter(env.agent_pos[0], env.agent_pos[1], c='#FF6B6B', s=300, marker='o', 
                    edgecolors='black', linewidth=2, label='Agent', zorder=4)
    
    ax_true.set_title(f"True Grid {title_suffix}", fontsize=14, fontweight='bold', pad=15)
    ax_true.set_xlabel('X coordinate (cells)', fontsize=11, fontweight='bold')
    ax_true.set_ylabel('Y coordinate (cells)', fontsize=11, fontweight='bold')
    ax_true.set_xlim(-0.5, width - 0.5)
    ax_true.set_ylim(height - 0.5, -0.5)
    ax_true.set_xticks(np.arange(0, width, max(1, width // 10)))
    ax_true.set_yticks(np.arange(0, height, max(1, height // 10)))
    ax_true.tick_params(labelsize=10)
    
    # ===== OBSERVED GRID (RIGHT) =====
    ax_obs.imshow(env.obs_grid, cmap='Greys', origin='upper', vmin=0, vmax=255)
    
    # Add grid lines for better visibility
    for i in np.arange(-0.5, height, 1):
        ax_obs.axhline(y=i, color='gray', linewidth=0.5, alpha=0.3)
    for j in np.arange(-0.5, width, 1):
        ax_obs.axvline(x=j, color='gray', linewidth=0.5, alpha=0.3)
    
    # Plot agent on observed grid
    ax_obs.scatter(env.agent_pos[0], env.agent_pos[1], c='#FF6B6B', s=300, marker='o', 
                   edgecolors='black', linewidth=2, label='Agent', zorder=4)
    
    ax_obs.set_title(f"Observed/Beliefs Grid{title_suffix}", fontsize=14, fontweight='bold', pad=15)
    ax_obs.set_xlabel('X coordinate (cells)', fontsize=11, fontweight='bold')
    ax_obs.set_ylabel('Y coordinate (cells)', fontsize=11, fontweight='bold')
    ax_obs.set_xlim(-0.5, width - 0.5)
    ax_obs.set_ylim(height - 0.5, -0.5)
    ax_obs.set_xticks(np.arange(0, width, max(1, width // 10)))
    ax_obs.set_yticks(np.arange(0, height, max(1, height // 10)))
    ax_obs.tick_params(labelsize=10)
    
    # Create comprehensive legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', 
               markersize=10, markeredgecolor='black', markeredgewidth=1.5, label='Agent'),
        Patch(facecolor='white', edgecolor='black', linewidth=1.5, label='Free cells'),
        Patch(facecolor='lightgray', edgecolor='gray', linewidth=0.5, label='Unknown cells'),
        Patch(facecolor='black', edgecolor='gray', linewidth=0.5, label='Occupied cells')
    ]
    
    # Add legend to both subplots
    # for ax in [ax_true, ax_obs]:
    #     ax.legend(handles=legend_elements, loc='upper right', fontsize=11, 
    #               framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
    ax_obs.legend(handles=legend_elements, loc='upper right', fontsize=11,
                  framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
                  
    plt.tight_layout()
    return fig, (ax_true, ax_obs)


if __name__ == "__main__":
    print("=" * 60)
    print("EXPLORATION ENVIRONMENT VISUALIZATION")
    print("=" * 60)
    
    # ===== ENVIRONMENT 1: 20x20 with 0.05 obstacle probability =====
    print("\n[1/2] Creating Environment 1: 20x20, obstacle_prob=0.05...")
    env1 = stepper_wrapper_class(ObstGridAgentExplEnv)(
        width=20,
        height=20,
        obstacle_prob=0.05,
        perception_range=3,
        render_mode=None,
        
    )
    
    print("Resetting Environment 1...")
    env1.reset(init_agent_pos=np.array([15, 10]))
    
    print("Taking 10 random steps in Environment 1...")
    for i in range(10):
        action = np.random.randint(0, 4)
        env1.step(action)
        if (i + 1) % 5 == 0:
            print(f"  Step {i + 1}/10 completed")
    
    print("Rendering Environment 1...")
    fig1, ax1 = render_exploration_grids(env1, " - 20x20 (obstacle_prob=0.05)")
    
    # ===== ENVIRONMENT 2: 20x40 with 0.2 obstacle probability =====
    print("\n[2/2] Creating Environment 2: 20x40, obstacle_prob=0.2...")
    env2 = stepper_wrapper_class(ObstGridAgentExplEnv)(
        width=40,
        height=20,
        obstacle_prob=0.025,
        perception_range=3,
        render_mode=None,
        
    )
    
    print("Resetting Environment 2...")
    env2.reset(init_agent_pos=np.array([20, 10]))
    
    print("Taking 20 random steps in Environment 2...")
    for i in range(40):
        action = np.random.randint(0, 4)
        env2.step(action)
        if (i + 1) % 5 == 0:
            print(f"  Step {i + 1}/20 completed")
    
    print("Rendering Environment 2...")
    fig2, ax2 = render_exploration_grids(env2, " - 20x40 (obstacle_prob=0.2)")
    
    # ===== SAVE FIGURES =====
    print("\n" + "=" * 60)
    print("SAVING FIGURES...")
    print("=" * 60)
    
    fig1.savefig('images/exploration_env1_20x20.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: exploration_env1_20x20.png")
    
    fig2.savefig('images/exploration_env2_20x40.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: exploration_env2_20x40.png")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Environment 1 - Agent discovered cells: {env1.discovered_cells}")
    print(f"Environment 2 - Agent discovered cells: {env2.discovered_cells}")
    print("\nVisualization complete! Figures saved successfully.")
    
    plt.show()
