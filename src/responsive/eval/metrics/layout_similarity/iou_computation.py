"""
IoU (Intersection over Union) computation functions for layout similarity.

Provides both basic and shapely-based implementations for computing IoU scores
between element lists and weighted IoU across multiple element types.
"""

from shapely.ops import unary_union

from .box_utils import intersection, bounding_box_to_polygon


def compute_list_iou(listA, listB):
    """
    Compute IoU between two element lists (original method).
    
    Args:
        listA: List of element dicts with 'box' key
        listB: List of element dicts with 'box' key
    
    Returns:
        Tuple of (total_iou, total_union_area)
    """
    total_intersection = 0
    total_area_A = 0
    total_area_B = 0

    for elemA in listA:
        boxA = elemA['box']
        total_area_A += boxA['width'] * boxA['height']
        for elemB in listB:
            boxB = elemB['box']
            total_intersection += intersection(boxA, boxB)

    for elemB in listB:
        boxB = elemB['box']
        total_area_B += boxB['width'] * boxB['height']

    total_union = total_area_A + total_area_B - total_intersection

    # Calculate the Intersection over Union (IoU)
    total_iou = total_intersection / total_union if total_union > 0 else 0

    return total_iou, total_area_A + total_area_B


def compute_weighted_iou(elementsA, elementsB):
    """
    Compute weighted IoU across multiple element types (original method).
    
    Args:
        elementsA: Dict mapping element type to list of elements
        elementsB: Dict mapping element type to list of elements (reference)
    
    Returns:
        Weighted IoU score (float)
    """
    areas = {}
    ious = {}

    all_keys = set(elementsA.keys()).union(set(elementsB.keys()))

    for key in all_keys:       # elementsB is the reference layout
        if key not in elementsA:
            elementsA[key] = []
        if key not in elementsB:
            elementsB[key] = []

        ious[key], areas[key] = compute_list_iou(elementsA[key], elementsB[key])

    total_area = sum(areas[key] for key in all_keys)
    weighted_iou = sum(areas[key] * ious[key] for key in all_keys) / total_area if total_area > 0 else 0

    return weighted_iou


def compute_list_iou_shapely(listA, listB):
    """
    Compute IoU between two element lists using shapely (preferred method).
    
    Args:
        listA: List of element dicts with 'box' key
        listB: List of element dicts with 'box' key
    
    Returns:
        Tuple of (total_iou, union_area)
    """
    if not listA and not listB:
        # If both lists are empty, IoU is 0 and union area is 0
        return 0.0, 0.0
    elif not listA:
        # If listA is empty, IoU is 0 and union area is the area of listB
        polygonsB = [bounding_box_to_polygon(elem['box']) for elem in listB]
        unionB = unary_union(polygonsB)
        union_area = unionB.area
        return 0.0, union_area
    elif not listB:
        # If listB is empty, IoU is 0 and union area is the area of listA
        polygonsA = [bounding_box_to_polygon(elem['box']) for elem in listA]
        unionA = unary_union(polygonsA)
        union_area = unionA.area
        return 0.0, union_area

    # Convert bounding boxes to shapely polygons
    polygonsA = [bounding_box_to_polygon(elem['box']) for elem in listA]
    polygonsB = [bounding_box_to_polygon(elem['box']) for elem in listB]

    # Create a union of all polygons in listA and listB
    unionA = unary_union(polygonsA)
    unionB = unary_union(polygonsB)

    # Compute the total intersection and union areas
    intersection_area = unionA.intersection(unionB).area
    union_area = unionA.union(unionB).area

    total_iou = intersection_area / union_area if union_area > 0 else 0

    return total_iou, union_area


def compute_weighted_iou_shapely(elementsA, elementsB):
    """
    Compute weighted IoU across multiple element types using shapely (preferred method).
    
    Args:
        elementsA: Dict mapping element type to list of elements
        elementsB: Dict mapping element type to list of elements (reference)
    
    Returns:
        Tuple of (weighted_iou, multi_score_dict)
        - weighted_iou: Overall weighted IoU score (float)
        - multi_score_dict: Dict mapping element type to (iou_score, weight) tuple
    """
    areas = {}
    ious = {}

    all_keys = set(elementsA.keys()).union(set(elementsB.keys()))

    for key in all_keys:       # elementsB is the reference layout
        if key not in elementsA:
            elementsA[key] = []
        if key not in elementsB:
            elementsB[key] = []

        ious[key], areas[key] = compute_list_iou_shapely(elementsA[key], elementsB[key])

    total_area = sum(areas[key] for key in all_keys)
    weighted_iou = sum(areas[key] * ious[key] for key in all_keys) / total_area if total_area > 0 else 0

    multi_score = {}
    for key in all_keys:
        multi_score[key] = (ious[key], areas[key] / total_area)     # score, weight

    return weighted_iou, multi_score
