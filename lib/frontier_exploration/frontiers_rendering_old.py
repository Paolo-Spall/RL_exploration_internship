#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def render_frontiers(grid, centroids, clusters, agent_pos, rl='right'):
    """
    Render frontier regions and clusters on a grid with professional thesis-quality visualization.
    
    Parameters:
    -----------
    grid : np.ndarray
        2D grid where 0=free, 1=occupied, 255=unknown cells
    centroids : np.ndarray
        Centroids of each cluster [N, 2]
    clusters : list of np.ndarray
        List of cluster arrays, each containing frontier cell coordinates
    agent_pos : np.ndarray
        Agent position [x, y]
    """
    # Create figure with high DPI for thesis quality
    fig, ax_env = plt.subplots(1, 1, figsize=(12, 10), dpi=150)
    
    height, width = grid.shape
    
    # Create custom colormap for the grid background
    # Use a more informative visualization: light gray for free, dark gray for occupied, light blue for unknown
    grid_display = grid.astype(float)
    grid_display[grid == 85] = 1.  # Unknown cells - light
    grid_display[grid == 255] = 0.0  # Agent cell - light
    grid_display[grid == 0] = 0.0   # Free cells - very light
    grid_display[grid == 170] = 0.7   # Occupied cells - very light
    #grid_display[grid > 1] = 0.1     # Occupied cells - very dark
    
    ax_env.imshow(grid_display, cmap='Greys', origin='upper', vmin=0, vmax=1)
    
    # Add grid lines for better visibility
    for i in np.arange(-0.5, height, 1):
        ax_env.axhline(y=i, color='gray', linewidth=0.5, alpha=0.3)
    for j in np.arange(-0.5, width, 1):
        ax_env.axvline(x=j, color='gray', linewidth=0.5, alpha=0.3)
    
    # Define a color palette for clusters
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(clusters))))
    
    # Plot each cluster with its own color
    for idx, cluster in enumerate(clusters):
        if len(cluster) > 0:
            color = colors[idx % len(colors)]
            # Plot frontier cells in this cluster
            ax_env.scatter(cluster[:, 0], cluster[:, 1], c=[color], s=200, 
                          alpha=0.7, edgecolors='black', linewidth=0.5, 
                          marker='o', zorder=3, label=f'Cluster {idx+1}' if idx < 5 else '')
    
    # Plot centroids as crosses with same color as their cluster
    if len(centroids) > 0:
        for idx, centroid in enumerate(centroids):
            color = colors[idx % len(colors)]
            ax_env.scatter(centroid[0], centroid[1], c=[color], s=250, 
                          marker='x', linewidth=2.5, edgecolors='black', zorder=4)
    
    # Plot agent position
    ax_env.scatter(agent_pos[0], agent_pos[1], c='#06A77D', s=300, marker='o', 
                   edgecolors='black', linewidth=2, label='Agent', zorder=5)
    
    # Create comprehensive legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=8, markeredgecolor='black', markeredgewidth=0.5, label='Frontier cells'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='gray', 
               markersize=10, markeredgecolor='black', markeredgewidth=2, label='Centroids (cluster color)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#06A77D', 
               markersize=8, markeredgecolor='black', markeredgewidth=1.5, label='Agent'),
        Patch(facecolor='white', edgecolor='black', linewidth=1.5, label='Free cells'),
        Patch(facecolor='black', edgecolor='gray', linewidth=0.5, label='Unknown cells'),
        Patch(facecolor='lightgray', edgecolor='gray', linewidth=0.5, label='Occupied cells')
    ]
    
    ax_env.legend(handles=legend_elements, loc=f'upper {rl}', fontsize=13, 
                  framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
    
    # Set title and labels
    #ax_env.set_title("Frontier cells and clusters", fontsize=16, fontweight='bold', pad=20)
    ax_env.set_xlabel('X coordinate (cells)', fontsize=12, fontweight='bold')
    ax_env.set_ylabel('Y coordinate (cells)', fontsize=12, fontweight='bold')
    
    # Set axis limits and ticks for better appearance
    ax_env.set_xlim(-0.5, width - 0.5)
    ax_env.set_ylim(height - 0.5, -0.5)
    ax_env.set_xticks(np.arange(0, width, max(1, width // 10)))
    ax_env.set_yticks(np.arange(0, height, max(1, height // 10)))
    ax_env.tick_params(labelsize=10)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig, ax_env


def render_frontiers_comparison(grid, 
                                centroids_orig, 
                                clusters_orig, 
                                centroids_split, 
                                clusters_split, 
                                agent_pos,
                                legend_rl='right'):
    """
    Render two frontier cluster configurations side-by-side with professional thesis-quality visualization.
    
    Parameters:
    -----------
    grid : np.ndarray
        2D grid where 0=free, 1=occupied, 255=unknown cells
    centroids_orig : np.ndarray
        Centroids of original (unsplitted) clusters [N, 2]
    clusters_orig : list of np.ndarray
        List of original cluster arrays
    centroids_split : np.ndarray
        Centroids of splitted clusters [M, 2]
    clusters_split : list of np.ndarray
        List of splitted cluster arrays
    agent_pos : np.ndarray
        Agent position [x, y]
    """
    # Create figure with high DPI for thesis quality
    fig, (ax_orig, ax_split) = plt.subplots(1, 2, figsize=(18, 8), dpi=150)
    
    height, width = grid.shape
    
    # Define a color palette for clusters
    max_clusters = max(len(clusters_orig), len(clusters_split))
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, max_clusters)))
    
    # ===== ORIGINAL CLUSTERS (LEFT) =====
    grid_display = grid.astype(float)
    grid_display[grid == 255] = 0.3  # Unknown cells
    grid_display[grid == 0] = 0.95   # Free cells
    grid_display[grid > 1] = 0.1     # Occupied cells
    
    ax_orig.imshow(grid_display, cmap='Greys', origin='upper', vmin=0, vmax=1)
    
    # Add grid lines
    for i in np.arange(-0.5, height, 1):
        ax_orig.axhline(y=i, color='gray', linewidth=0.5, alpha=0.3)
    for j in np.arange(-0.5, width, 1):
        ax_orig.axvline(x=j, color='gray', linewidth=0.5, alpha=0.3)
    
    # Plot clusters
    for idx, cluster in enumerate(clusters_orig):
        if len(cluster) > 0:
            color = colors[idx % len(colors)]
            ax_orig.scatter(cluster[:, 0], cluster[:, 1], c=[color], s=60, 
                           alpha=0.7, edgecolors='black', linewidth=0.5, 
                           marker='o', zorder=3)
    
    # Plot centroids with cluster colors
    if len(centroids_orig) > 0:
        for idx, centroid in enumerate(centroids_orig):
            color = colors[idx % len(colors)]
            ax_orig.scatter(centroid[0], centroid[1], c=[color], s=250, 
                           marker='x', linewidth=2.5, edgecolors='black', zorder=4)
    
    # Plot agent
    ax_orig.scatter(agent_pos[0], agent_pos[1], c='#06A77D', s=300, marker='o', 
                   edgecolors='black', linewidth=2, label='Agent', zorder=5)
    
    ax_orig.set_title("Original Clusters", fontsize=14, fontweight='bold', pad=15)
    ax_orig.set_xlabel('X coordinate (cells)', fontsize=11, fontweight='bold')
    ax_orig.set_ylabel('Y coordinate (cells)', fontsize=11, fontweight='bold')
    ax_orig.set_xlim(-0.5, width - 0.5)
    ax_orig.set_ylim(height - 0.5, -0.5)
    ax_orig.set_xticks(np.arange(0, width, max(1, width // 10)))
    ax_orig.set_yticks(np.arange(0, height, max(1, height // 10)))
    ax_orig.tick_params(labelsize=10)
    
    # ===== SPLITTED CLUSTERS (RIGHT) =====
    ax_split.imshow(grid_display, cmap='Greys', origin='upper', vmin=0, vmax=1)
    
    # Add grid lines
    for i in np.arange(-0.5, height, 1):
        ax_split.axhline(y=i, color='gray', linewidth=0.5, alpha=0.3)
    for j in np.arange(-0.5, width, 1):
        ax_split.axvline(x=j, color='gray', linewidth=0.5, alpha=0.3)
    
    # Plot clusters
    for idx, cluster in enumerate(clusters_split):
        if len(cluster) > 0:
            color = colors[idx % len(colors)]
            ax_split.scatter(cluster[:, 0], cluster[:, 1], c=[color], s=60, 
                            alpha=0.7, edgecolors='black', linewidth=0.5, 
                            marker='o', zorder=3)
    
    # Plot centroids with cluster colors
    if len(centroids_split) > 0:
        for idx, centroid in enumerate(centroids_split):
            color = colors[idx % len(colors)]
            ax_split.scatter(centroid[0], centroid[1], c=[color], s=250, 
                            marker='x', linewidth=2.5, edgecolors='black', zorder=4)
    
    # Plot agent
    ax_split.scatter(agent_pos[0], agent_pos[1], c='#06A77D', s=300, marker='o', 
                    edgecolors='black', linewidth=2, label='Agent', zorder=5)
    
    ax_split.set_title("Splitted Clusters", fontsize=14, fontweight='bold', pad=15)
    ax_split.set_xlabel('X coordinate (cells)', fontsize=11, fontweight='bold')
    ax_split.set_ylabel('Y coordinate (cells)', fontsize=11, fontweight='bold')
    ax_split.set_xlim(-0.5, width - 0.5)
    ax_split.set_ylim(height - 0.5, -0.5)
    ax_split.set_xticks(np.arange(0, width, max(1, width // 10)))
    ax_split.set_yticks(np.arange(0, height, max(1, height // 10)))
    ax_split.tick_params(labelsize=10)
    
    # Create comprehensive legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=8, markeredgecolor='black', markeredgewidth=0.5, label='Frontier cells'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='gray', 
               markersize=10, markeredgecolor='black', markeredgewidth=2, label='Centroids (cluster color)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#06A77D', 
               markersize=8, markeredgecolor='black', markeredgewidth=1.5, label='Agent'),
        Patch(facecolor='white', edgecolor='black', linewidth=1.5, label='Free cells'),
        Patch(facecolor='lightgray', edgecolor='gray', linewidth=0.5, label='Unknown cells'),
        Patch(facecolor='black', edgecolor='gray', linewidth=0.5, label='Occupied cells')
    ]
    
    # Add legend to both subplots 
    # for ax in [ax_orig, ax_split]:
    #     ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
    #               framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
    ax_orig.legend(handles=legend_elements, loc=f'upper {legend_rl}', fontsize=10, 
                framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
    
    plt.tight_layout()
    return fig, (ax_orig, ax_split)