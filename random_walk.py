#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Random walk parameters
GRID_SIZE = 16
MOVES = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # right, down, left, up

# Global state for the random walk
grid = np.zeros((GRID_SIZE, GRID_SIZE))
path = [(0, 0)]
current_x = 0
current_y = 0
is_complete = False

# Initialize the grid
grid[0, 0] = 1  # Mark starting position as visited

def find_valid_moves(x: int, y: int, grid: np.ndarray) -> list[tuple[int, int]]:
    """Find all valid moves from current position."""
    valid_moves = []
    grid_size = len(grid)

    for dx, dy in MOVES:
        new_x = x + dx
        new_y = y + dy
        if (0 <= new_x < grid_size and
            0 <= new_y < grid_size and
            grid[new_y, new_x] == 0):
            valid_moves.append((dx, dy))
    return valid_moves

def make_random_step() -> None:
    """Make one step in the random walk."""
    global current_x, current_y, is_complete

    if is_complete:
        return

    valid_moves = find_valid_moves(current_x, current_y, grid)

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
def setup_render(grid_dimensions: int) -> tuple[plt.Figure, plt.Axes, plt.Line2D]:
    """Set up the matplotlib figure and axis for rendering."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1, grid_dimensions)
    ax.set_ylim(-1, grid_dimensions)
    ax.invert_yaxis()  # To match array coordinates
    ax.set_aspect('equal', adjustable='box')

    # Add custom gridlines offset by 0.5
    for i in range(grid_dimensions + 1):
        ax.axvline(x=i - 0.5, color='lightgray', linewidth=0.5)
        ax.axhline(y=i - 0.5, color='lightgray', linewidth=0.5)

    ax.set_title('Random Walk Animation')

    # Initialize line
    line, = ax.plot([], [], 'b-', alpha=0.7, linewidth=2)

    return fig, ax, line

def render_frame(frame: int, ax: plt.Axes, line: plt.Line2D, current_path: list[tuple[int, int]], walk_complete: bool) -> tuple[plt.Line2D]:
    """Render one frame of the animation."""
    # Make one step in the random walk
    make_random_step()

    # Update the visualization
    if current_path:
        x_coords, y_coords = zip(*current_path)
        line.set_data(x_coords, y_coords)

    return line,

def run_animation() -> None:
    """Run the random walk animation."""
    fig, ax, line = setup_render(GRID_SIZE)

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        lambda frame: render_frame(frame, ax, line, path, is_complete),
        interval=0,
        blit=False,
        repeat=False,
        cache_frame_data=False
    )

    plt.show()

    # Optional: Save the animation
    # anim.save('random_walk_animation.gif', writer='pillow', fps=10)

if __name__ == "__main__":
    run_animation()