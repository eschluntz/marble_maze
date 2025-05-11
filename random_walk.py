#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import time

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

def reset_walk() -> None:
    """Reset the random walk to start a new one."""
    global grid, path, current_x, current_y, is_complete

    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    path = [(0, 0)]
    current_x = 0
    current_y = 0
    is_complete = False
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

    valid_moves = find_valid_moves(current_x, current_y, grid)

    if not valid_moves:
        is_complete = True
        return

    # Choose a random valid move
    dx, dy = random.choice(valid_moves)
    current_x += dx
    current_y += dy
    grid[current_y, current_x] = 1
    path.append((current_x, current_y))

def make_curvy_step() -> None:
    """Make one step, preferring turns over straight lines."""
    global current_x, current_y, is_complete

    valid_moves = find_valid_moves(current_x, current_y, grid)

    if not valid_moves:
        is_complete = True
        return

    # Get the last direction if we have more than one point in path
    if len(path) > 1:
        last_dx = path[-1][0] - path[-2][0]
        last_dy = path[-1][1] - path[-2][1]

        # Filter out moves that go straight (same direction as last move)
        turning_moves = []
        straight_move = None

        for dx, dy in valid_moves:
            if dx == last_dx and dy == last_dy:
                straight_move = (dx, dy)
            else:
                turning_moves.append((dx, dy))

        # Prefer turning moves if available
        if turning_moves:
            dx, dy = random.choice(turning_moves)
        elif straight_move:
            dx, dy = straight_move
        else:
            # Shouldn't happen, but fallback to random
            dx, dy = random.choice(valid_moves)
    else:
        # First move, choose randomly
        dx, dy = random.choice(valid_moves)

    # Make the move
    current_x += dx
    current_y += dy
    grid[current_y, current_x] = 1
    path.append((current_x, current_y))

def make_weighted_step() -> None:
    """Make one step with weighted random choice to prefer moves toward (0,0)."""
    global current_x, current_y, is_complete

    valid_moves = find_valid_moves(current_x, current_y, grid)

    if not valid_moves:
        is_complete = True
        return

    # Calculate weights based on distance to origin (0,0)
    weights = []
    for dx, dy in valid_moves:
        new_x = current_x + dx
        new_y = current_y + dy
        # Calculate current distance and new distance to origin
        current_dist = (current_x ** 2 + current_y ** 2) ** 0.5
        new_dist = (new_x ** 2 + new_y ** 2) ** 0.5

        # Weight: higher weight for moves that reduce distance
        # Add 1 to avoid zero weights and ensure all moves have some chance
        if new_dist < current_dist:
            # Moving closer to origin - higher weight
            weight = 20.0
        elif new_dist > current_dist:
            # Moving away from origin - lower weight
            weight = 0.5
        else:
            # Same distance - neutral weight
            weight = 1.0

        weights.append(weight)

    # Choose move using weighted random selection
    dx, dy = random.choices(valid_moves, weights=weights, k=1)[0]

    # Make the move
    current_x += dx
    current_y += dy
    grid[current_y, current_x] = 1
    path.append((current_x, current_y))

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

    # Remove tick marks
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title('Random Walk Animation')

    # Initialize line
    line, = ax.plot([], [], 'b-', alpha=0.7, linewidth=2)

    return fig, ax, line

def render_frame(frame: int, ax: plt.Axes, line: plt.Line2D) -> tuple[plt.Line2D]:
    """Render one frame of the animation."""
    global is_complete

    # If the walk is complete, wait a bit then reset
    if is_complete:
        # Show the complete path for a short time
        if frame % 100 == 19:  # After 20 frames, reset
            reset_walk()
    else:
        # Make one step in the random walk (using weighted preference)
        make_weighted_step()

    # Update the visualization
    if path:
        x_coords, y_coords = zip(*path)
        line.set_data(x_coords, y_coords)

    return line,

def run_animation() -> None:
    """Run the random walk animation."""
    fig, ax, line = setup_render(GRID_SIZE)

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        lambda frame: render_frame(frame, ax, line),
        interval=1,  # Slow down the animation
        blit=False,
        cache_frame_data=False
    )

    plt.show()

    # Optional: Save the animation
    # anim.save('random_walk_animation.gif', writer='pillow', fps=10)

def find_longest_path(num_attempts: int = 100) -> tuple[list[tuple[int, int]], int]:
    """Find the longest possible path by running multiple attempts.

    Args:
        num_attempts: Number of paths to try generating

    Returns:
        Tuple containing the longest path and its length
    """
    global is_complete, path

    longest_path = []
    longest_length = 0
    max_possible_length = GRID_SIZE * GRID_SIZE

    for attempt in range(num_attempts):
        # Reset for a new attempt
        reset_walk()

        # Generate a complete path
        while not is_complete:
            make_weighted_step()

        # Check if this path is longer than our current best
        if len(path) > longest_length:
            longest_length = len(path)
            longest_path = path.copy()  # Make a copy to preserve it

            # If we've found a path that fills the entire grid, we can't do better
            if longest_length == max_possible_length:
                print(f"\nFound a perfect path that fills the entire grid! (Length: {longest_length})")
                return longest_path, longest_length

        print(f"Attempt {attempt+1}/{num_attempts}: Path length = {len(path)}, Best so far = {longest_length}")

    print(f"\nLongest path found: {longest_length} steps")
    return longest_path, longest_length

def plot_longest_path(path: list[tuple[int, int]]) -> None:
    """Plot the longest path found."""
    # Reuse setup_render function for consistent styling
    fig, ax, line = setup_render(GRID_SIZE)

    # Update title for static plot
    ax.set_title(f'Longest Random Walk (Length: {len(path)} steps)')

    # Plot the path
    x_coords, y_coords = zip(*path)
    line.set_data(x_coords, y_coords)

    # Mark start and end points
    ax.plot(x_coords[0], y_coords[0], 'go', markersize=10)  # Start (green)
    ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10)  # End (red)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Choose one of these options:
    # Option 1: Run the animation
    # run_animation()

    # Option 2: Find the longest path and plot it
    longest_path, path_length = find_longest_path(5000)
    plot_longest_path(longest_path)