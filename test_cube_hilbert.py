import numpy as np
import matplotlib.pyplot as plt
from cube_hilbert import get_cube_hilbert_curve

def test_connections(points, tolerance=0.1):
    """Test if consecutive points are reasonably close (good connections)"""
    distances = []
    for i in range(len(points) - 1):
        dist = np.linalg.norm(points[i+1] - points[i])
        distances.append(dist)
    
    distances = np.array(distances)
    print(f"Connection distances: min={distances.min():.4f}, max={distances.max():.4f}, mean={distances.mean():.4f}")
    
    # Find large jumps (potential connection issues)
    large_jumps = distances > tolerance
    if np.any(large_jumps):
        jump_indices = np.where(large_jumps)[0]
        print(f"Large jumps found at indices: {jump_indices}")
        for idx in jump_indices:
            print(f"  Jump {idx}: {points[idx]} -> {points[idx+1]} (distance: {distances[idx]:.4f})")
    else:
        print("All connections look good!")
    
    return distances

def plot_cube_faces(points):
    """Plot the points colored by which face they belong to"""
    fig = plt.figure(figsize=(15, 10))
    
    # 3D plot
    ax = fig.add_subplot(111, projection='3d')
    
    # Assume 6 faces with equal number of points each
    n_per_face = len(points) // 6
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']
    face_names = ['Front', 'Bottom', 'Right', 'Top', 'Back', 'Left']
    
    for i in range(6):
        start_idx = i * n_per_face
        end_idx = (i + 1) * n_per_face if i < 5 else len(points)
        face_points = points[start_idx:end_idx]
        
        ax.scatter(face_points[:, 0], face_points[:, 1], face_points[:, 2], 
                  c=colors[i], label=f'{face_names[i]} face', alpha=0.7, s=20)
    
    # Draw the path
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'k-', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Cube Hilbert Curve - All 6 Faces')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test with small order first and margin
    print("Testing cube Hilbert curve...")
    cube_points = get_cube_hilbert_curve(order=3, cube_size=1.0, margin=0.1)
    
    print(f"Generated {len(cube_points)} points")
    print(f"Points per face: {len(cube_points) // 6}")
    
    # Test connections
    distances = test_connections(cube_points, tolerance=0.5)
    
    # Show first and last few points
    print("\nFirst 5 points:")
    for i in range(min(5, len(cube_points))):
        print(f"  {i}: {cube_points[i]}")
    
    print("\nLast 5 points:")
    for i in range(max(0, len(cube_points)-5), len(cube_points)):
        print(f"  {i}: {cube_points[i]}")
    
    # Plot
    plot_cube_faces(cube_points)