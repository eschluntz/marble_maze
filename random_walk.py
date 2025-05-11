#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Random walk parameters
grid_size = 16
moves = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # right, down, left, up

# Global state for the random walk
grid = np.zeros((grid_size, grid_size))
path = [(0, 0)]
current_x = 0
current_y = 0
is_complete = False

# Initialize the grid
grid[0, 0] = 1  # Mark starting position as visited

def find_valid_moves(x, y):
    """Find all valid moves from current position."""
    valid_moves = []
    for dx, dy in moves:
        new_x, new_y = x + dx, y + dy
        if (0 <= new_x < grid_size and 
            0 <= new_y < grid_size and 
            grid[new_y, new_x] == 0):
            valid_moves.append((dx, dy))
    return valid_moves

def make_random_step():
    """Make one step in the random walk."""
    global current_x, current_y, is_complete
    
    if is_complete:
        return
    
    valid_moves = find_valid_moves(current_x, current_y)
    
    if valid_moves:
        # Choose a random valid move
        dx, dy = random.choice(valid_moves)
        current_x += dx
        current_y += dy
        grid[current_y, current_x] = 1
        path.append((current_x, current_y))
    else:
        # No valid moves - walk is complete
        is_complete = True
        print(f"Path length: {len(path)}")

# Rendering functions
def setup_render():
    """Set up the matplotlib figure and axis for rendering."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1, grid_size)
    ax.set_ylim(-1, grid_size)
    ax.invert_yaxis()  # To match array coordinates
    ax.set_aspect('equal', adjustable='box')
    
    # Add custom gridlines offset by 0.5
    for i in range(grid_size + 1):
        ax.axvline(x=i - 0.5, color='lightgray', linewidth=0.5)
        ax.axhline(y=i - 0.5, color='lightgray', linewidth=0.5)
    
    ax.set_title('Random Walk Animation')
    
    # Initialize line
    line, = ax.plot([], [], 'b-', alpha=0.7, linewidth=2)
    
    return fig, ax, line

def render_frame(frame, ax, line):
    """Render one frame of the animation."""
    # Make one step in the random walk
    make_random_step()
    
    # Update the visualization
    if len(path) > 0:
        x_coords, y_coords = zip(*path)
        line.set_data(x_coords, y_coords)
        
        if is_complete:
            ax.set_title(f'Random Walk Complete (Path Length: {len(path)})')
        else:
            ax.set_title(f'Random Walk Animation (Step {len(path)})')
    
    return line,

def run_animation():
    """Run the random walk animation."""
    fig, ax, line = setup_render()
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, 
        lambda frame: render_frame(frame, ax, line),
        interval=100, 
        blit=False, 
        repeat=False,
        cache_frame_data=False
    )
    
    plt.show()
    
    # Optional: Save the animation
    # anim.save('random_walk_animation.gif', writer='pillow', fps=10)

if __name__ == "__main__":
    run_animation()