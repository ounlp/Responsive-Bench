"""
Box utility functions for layout similarity computation.

Functions for checking box relationships, merging boxes, and converting to geometric shapes.
"""

from shapely.geometry import box


def boxes_adjacent(box1, box2, align_tolerance=8, adj_tolerance=4):
    """
    Determine if two boxes are adjacent considering a horizontal and vertical tolerance.
    
    Boxes are considered adjacent if:
    - They are nearly aligned vertically (their vertical centers are close within y_tolerance)
    - They are horizontally sequential without a large gap (distance between them does not exceed x_tolerance)
    - OR they are vertically sequential where one box is directly above or below the other within y_tolerance
    
    Args:
        box1: Dict with 'x', 'y', 'width', 'height'
        box2: Dict with 'x', 'y', 'width', 'height'
        align_tolerance: Tolerance for alignment check (default 8)
        adj_tolerance: Tolerance for adjacency check (default 4)
    
    Returns:
        True if boxes are adjacent, False otherwise
    """
    # Calculate vertical centers and horizontal centers
    vertical_center1 = box1['y'] + box1['height'] / 2
    vertical_center2 = box2['y'] + box2['height'] / 2

    horizontal_center1 = box1['x'] + box1['width'] / 2
    horizontal_center2 = box2['x'] + box2['width'] / 2

    # Check vertical alignment
    vertically_aligned = abs(vertical_center1 - vertical_center2) <= align_tolerance

    # Check horizontal adjacency
    horizontally_adjacent = (box1['x'] + box1['width'] + adj_tolerance >= box2['x'] and box1['x'] < box2['x']) or \
                            (box2['x'] + box2['width'] + adj_tolerance >= box1['x'] and box2['x'] < box1['x'])

    # Check horizontal alignment for vertical adjacency
    horizontally_aligned = abs(horizontal_center1 - horizontal_center2) <= align_tolerance

    # Check vertical adjacency
    vertically_adjacent = (box1['y'] + box1['height'] + adj_tolerance >= box2['y'] and box1['y'] < box2['y']) or \
                          (box2['y'] + box2['height'] + adj_tolerance >= box1['y'] and box2['y'] < box1['y'])

    return (vertically_aligned and horizontally_adjacent) or (horizontally_aligned and vertically_adjacent)


def merge_boxes(box1, box2):
    """
    Merge two adjacent boxes into a larger box.
    
    Args:
        box1: Dict with 'x', 'y', 'width', 'height'
        box2: Dict with 'x', 'y', 'width', 'height'
    
    Returns:
        Dict with merged box coordinates
    """
    x1 = min(box1['x'], box2['x'])
    y1 = min(box1['y'], box2['y'])
    x2 = max(box1['x'] + box1['width'], box2['x'] + box2['width'])
    y2 = max(box1['y'] + box1['height'], box2['y'] + box2['height'])
    return {
        'x': x1,
        'y': y1,
        'width': x2 - x1,
        'height': y2 - y1
    }


def is_within(box1, box2):
    """
    Check if box1 is within box2.
    
    Args:
        box1: Dict with 'x', 'y', 'width', 'height'
        box2: Dict with 'x', 'y', 'width', 'height'
    
    Returns:
        True if box1 is completely within box2, False otherwise
    """
    return (box1['x'] >= box2['x'] and
            box1['y'] >= box2['y'] and
            box1['x'] + box1['width'] <= box2['x'] + box2['width'] and
            box1['y'] + box1['height'] <= box2['y'] + box2['height'])


def intersection(boxA, boxB):
    """
    Compute intersection area of two boxes.
    
    Args:
        boxA: Dict with 'x', 'y', 'width', 'height'
        boxB: Dict with 'x', 'y', 'width', 'height'
    
    Returns:
        Intersection area (float)
    """
    xA = max(boxA['x'], boxB['x'])
    yA = max(boxA['y'], boxB['y'])
    xB = min(boxA['x'] + boxA['width'], boxB['x'] + boxB['width'])
    yB = min(boxA['y'] + boxA['height'], boxB['y'] + boxB['height'])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)

    return inter_width * inter_height


def bounding_box_to_polygon(bbox):
    """
    Convert bounding box dict to shapely Polygon.
    
    Args:
        bbox: Dict with 'x', 'y', 'width', 'height'
    
    Returns:
        shapely.geometry.Polygon
    """
    return box(bbox['x'], bbox['y'], bbox['x'] + bbox['width'], bbox['y'] + bbox['height'])
