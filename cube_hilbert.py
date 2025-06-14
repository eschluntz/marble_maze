import numpy as np
from hilbertcurve import HilbertCurve

def get_cube_hilbert_curve(order, cube_size=1.0, margin=0.1):
    """
    Generate a 3D Hilbert curve that covers all 6 faces of a cube with perfect connections.
    
    Face sequence for seamless loop:
    1. Front face: front bottom left -> front bottom right
    2. Bottom face: front bottom right -> back bottom right  
    3. Right face: back bottom right -> back top right
    4. Top face: back top right -> back top left
    5. Back face: back top left -> back bottom left
    6. Left face: back bottom left -> front bottom left [closes loop]
    
    Args:
        order: Hilbert curve order (determines resolution)
        cube_size: Size of the cube (default 1.0)
        margin: Margin/padding around curves on each face (default 0.1)
                The curves will be scaled to fit within the inner area
    
    Returns:
        numpy array of shape (N, 3) containing all points
    """
    
    # Generate base 2D Hilbert curve 
    dimensions = 2
    hilbert_curve = HilbertCurve(order, dimensions)
    distances = list(range(2 ** (dimensions * order)))
    base_points = np.array(hilbert_curve.points_from_distances(distances), dtype=float)
    
    # Normalize to [0, 1] range
    max_coord = 2**order - 1
    base_points = base_points / max_coord
    
    # Apply margin: scale curves to fit within (margin, 1-margin) range
    curve_scale = 1 - 2 * margin
    base_points = base_points * curve_scale + margin
    
    # Standard curve: starts at (margin, margin), ends at (1-margin, margin)
    print(f"Base curve with margin {margin}: start={base_points[0]}, end={base_points[-1]}")
    
    all_points = []
    
    # Face 1: Front face (z=0)
    # Place curve ON the front face (z=0) with margins from edges
    front_3d = np.column_stack((base_points[:,0], base_points[:,1], np.zeros(len(base_points))))
    all_points.extend(front_3d)
    
    # Face 2: Bottom face (y=0) 
    # Place curve ON the bottom face (y=0) with margins from edges
    bottom_3d = np.column_stack((1 - base_points[:,1], np.zeros(len(base_points)), base_points[:,0]))
    all_points.extend(bottom_3d)
    
    # Face 3: Right face (x=1)
    # Place curve ON the right face (x=1) with margins from edges
    right_3d = np.column_stack((np.ones(len(base_points)), base_points[:,0], 1 - base_points[:,1]))
    all_points.extend(right_3d)
    
    # Face 4: Top face (y=1)
    # Place curve ON the top face (y=1) with margins from edges
    # Flip in Z direction
    top_3d = np.column_stack((1 - base_points[:,0], np.ones(len(base_points)), 1 - base_points[:,1]))
    all_points.extend(top_3d)
    
    # Face 5: Back face (z=1)
    # Place curve ON the back face (z=1) with margins from edges
    back_3d = np.column_stack((base_points[:,1], 1 - base_points[:,0], np.ones(len(base_points))))
    all_points.extend(back_3d)
    
    # Face 6: Left face (x=0)
    # Place curve ON the left face (x=0) with margins from edges
    # Flip in Z direction
    left_3d = np.column_stack((np.zeros(len(base_points)), base_points[:,1], 1 - base_points[:,0]))
    all_points.extend(left_3d)
    
    # Scale by cube_size
    all_points = np.array(all_points) * cube_size
    
    return all_points

if __name__ == "__main__":
    # Test the cube Hilbert curve with margins
    cube_points = get_cube_hilbert_curve(order=3, cube_size=1.0, margin=0.1)
    print(f"Generated {len(cube_points)} points covering all 6 cube faces with margin 0.1")
    print(f"Point range: x=[{cube_points[:,0].min():.2f}, {cube_points[:,0].max():.2f}], y=[{cube_points[:,1].min():.2f}, {cube_points[:,1].max():.2f}], z=[{cube_points[:,2].min():.2f}, {cube_points[:,2].max():.2f}]")