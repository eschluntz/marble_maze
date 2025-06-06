#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import time
import pickle
import os

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

def reset_detour_walk() -> None:
    """Initialize the walk for detour algorithm - straight line from top-left to top-right."""
    global grid, path, current_x, current_y, is_complete

    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    path = []

    # Create initial path from top-left to top-right
    for x in range(GRID_SIZE):
        path.append((x, 0))
        grid[0, x] = 1

    current_x = GRID_SIZE - 1
    current_y = 0
    is_complete = False

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

def make_detour_step() -> bool:
    """Try to add a detour to the existing path using shuffled indices.

    Returns:
        True if a detour was added, False if no more detours possible
    """
    global path, grid, is_complete

    # Check if there are no more valid edge positions to create detours
    if len(path) <= 1:
        is_complete = True
        return False

    # Create a shuffled list of all edge indices
    shuffled_indices = list(range(len(path) - 1))
    random.shuffle(shuffled_indices)

    # Try each index in the shuffled list
    for index in shuffled_indices:
        # Get the edge
        p_chosen = path[index]
        p_next = path[index + 1]

        # Determine the direction from p_chosen to p_next
        direction = (p_next[0] - p_chosen[0], p_next[1] - p_chosen[1])

        # Determine valid detour directions based on path direction
        detour_directions = []
        if direction[1] == 0:  # horizontal movement
            detour_directions = [(0, -1), (0, 1)]  # up and down
        elif direction[0] == 0:  # vertical movement
            detour_directions = [(-1, 0), (1, 0)]  # left and right

        # Try each detour direction
        for detour_dir in detour_directions:
            # Calculate the two new points for the detour
            first_detour_point = (p_chosen[0] + detour_dir[0], p_chosen[1] + detour_dir[1])
            second_detour_point = (p_next[0] + detour_dir[0], p_next[1] + detour_dir[1])

            # Check if both points are within bounds
            if (0 <= first_detour_point[0] < GRID_SIZE and
                0 <= first_detour_point[1] < GRID_SIZE and
                0 <= second_detour_point[0] < GRID_SIZE and
                0 <= second_detour_point[1] < GRID_SIZE):

                # Check if both spots are empty
                is_first_empty = grid[first_detour_point[1], first_detour_point[0]] == 0
                is_second_empty = grid[second_detour_point[1], second_detour_point[0]] == 0

                if is_first_empty and is_second_empty:
                    # Insert the detour
                    path[index + 1:index + 1] = [first_detour_point, second_detour_point]
                    grid[first_detour_point[1], first_detour_point[0]] = 1
                    grid[second_detour_point[1], second_detour_point[0]] = 1
                    return True  # Detour was added successfully

    # We've tried all indices and found no valid detour
    is_complete = True
    return False

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
    anim.save('random_walk_animation.gif', writer='pillow', fps=10)

def render_detour_frame(frame: int, ax: plt.Axes, line: plt.Line2D) -> tuple[plt.Line2D]:
    """Render one frame of the detour algorithm animation."""
    global is_complete

    # Try to make a detour every frame
    if not is_complete:
        make_detour_step()

    # Update the visualization
    if path:
        x_coords, y_coords = zip(*path)
        line.set_data(x_coords, y_coords)

    return line,


def is_complete_frame_generator():
    frame = 0
    while not is_complete:
        yield frame
        frame += 1
    # Add a few more frames to show the final state
    for _ in range(10):
        yield frame
        frame += 1


def run_detour_animation() -> None:
    """Run the detour path animation."""
    global is_complete
    reset_detour_walk()
    fig, ax, line = setup_render(GRID_SIZE)

    # Create animation with frame generator
    anim = animation.FuncAnimation(
        fig,
        lambda frame: render_detour_frame(frame, ax, line),
        frames=is_complete_frame_generator(),
        interval=1,  # Slower animation for detours
        blit=False,
        cache_frame_data=False,
        repeat=False  # Don't repeat the animation
    )
    # plt.show()
    print("Saving animation...")
    anim.save('detour_animation.gif', writer='pillow', fps=30)
    print("Animation saved as detour_animation.gif")
    plt.close()  # Close the figure to prevent display

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

    # Plot the path
    x_coords, y_coords = zip(*path)
    line.set_data(x_coords, y_coords)

    # Mark start and end points
    ax.plot(x_coords[0], y_coords[0], 'go', markersize=10)  # Start (green)
    ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10)  # End (red)

    plt.tight_layout()
    plt.show()

def create_composite_animation() -> None:
    """Create a composite animation with 3 random walks and the longest path."""
    global is_complete
    
    fig, ax, line = setup_render(GRID_SIZE)
    frames = []
    all_paths = []
    
    # Part 1: Three random walks
    completions = 0
    
    print("Generating 3 random walks...")
    while completions < 2:
        if is_complete:
            # Store the completed path
            all_paths.append(path.copy())
            reset_walk()
            completions += 1
        else:
            make_weighted_step()
        
        # Store frame data
        frames.append({
            'path': path.copy(),
            'phase': 'random_walk',
            'completion': completions
        })
    
    # Part 2: Find longest path
    print("Finding longest path...")
    longest_path, path_length = find_longest_path(5000)
    
    # Part 3: Animate longest path growth
    print("Adding longest path animation...")
    for i in range(1, len(longest_path) + 1):
        frames.append({
            'path': longest_path[:i],
            'phase': 'longest_path',
            'completion': 3
        })
    
    # Add some frames to show the complete longest path
    for _ in range(30):
        frames.append({
            'path': longest_path,
            'phase': 'longest_path_complete',
            'completion': 3
        })
    
    # Animation function for composite
    def animate_composite(frame_num):
        if frame_num >= len(frames):
            return line,
            
        frame_data = frames[frame_num]
        current_path = frame_data['path']
        phase = frame_data['phase']
        
        if current_path:
            x_coords, y_coords = zip(*current_path)
            line.set_data(x_coords, y_coords)
            
            # Update title based on phase
            if phase == 'random_walk':
                ax.set_title(f'Weighted Random Walk: Attempt {frame_data["completion"] + 1}')
            elif phase == 'longest_path':
                ax.set_title(f'Weighted Random Walk: Attempt 1652')
            else:  # longest_path_complete
                ax.set_title(f'Weighted Random Walk: Attempt 1652')
        
        return line,
    
    # Create the animation
    anim = animation.FuncAnimation(
        fig,
        animate_composite,
        frames=len(frames),
        interval=5,  # 50ms per frame
        blit=False,
        repeat=False
    )
    # plt.show()
    
    print("Saving composite animation...")
    anim.save('composite_random_walk.gif', writer='pillow', fps=30)
    print("Animation saved as composite_random_walk.gif")
    plt.close()

def save_points(method='weighted'):
    """Save points generated using the specified method.

    Args:
        method: Either 'weighted' for weighted random walk or 'detour' for detour algorithm
    """
    if method == 'weighted':
        longest_path, path_length = find_longest_path(5000)
        pickle_file = 'longest_path_weighted.pickle'
    elif method == 'detour':
        global is_complete, path
        reset_detour_walk()
        # Generate complete detour path
        while not is_complete:
            make_detour_step()
        longest_path = path.copy()
        path_length = len(longest_path)
        pickle_file = 'longest_path_detour.pickle'
    else:
        raise ValueError("Method must be either 'weighted' or 'detour'")

    # Save the path
    with open(pickle_file, 'wb') as f:
        pickle.dump(longest_path, f)
    print(f"Path (method={method}) of length {path_length} saved to {pickle_file}")


if __name__ == "__main__":
    # Create composite animation
    # create_composite_animation()

    # Save points using weighted random walk method
    # save_points('weighted')

    # Or save points using detour method
    save_points('detour')