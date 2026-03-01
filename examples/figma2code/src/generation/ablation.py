"""
Ablation study utilities for Figma-to-Code.

This module provides functions to systematically remove specific information
from Figma JSON metadata for ablation studies, evaluating the importance
of different information components for code generation.

Ablation types:
- geometry: Remove layout information (x, y, width, height, transforms)
- style: Remove visual styles (colors, fonts, effects, etc.)
- image_refs: Remove image references
- structure: Flatten node hierarchy
- text: Remove text content

Usage:
    from src.generation.ablation import ablate_geometry, ABLATION_FUNCTIONS
    
    # Apply single ablation
    ablated_data = ablate_geometry(figma_json)
    
    # Get ablation function by name
    func = ABLATION_FUNCTIONS["geometry"]
    ablated_data = func(figma_json)
"""

import copy
from collections import Counter
from typing import Dict, Any, Set, Callable

from ..utils.console_logger import logger


def _count_keys_recursive(obj: Any, keys: Set[str]) -> Dict[str, int]:
    """
    Count occurrences of specified keys in entire JSON tree.
    
    Args:
        obj: JSON-serializable object
        keys: Set of key names to count
    
    Returns:
        Dictionary mapping key names to counts
    """
    counter = Counter()
    
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in keys:
                    counter[k] += 1
                walk(v)
        elif isinstance(o, list):
            for it in o:
                walk(it)
    
    walk(obj)
    return {k: c for k, c in counter.items() if c}


def _get_root_node(data: dict) -> dict:
    """
    Get the traversable root node from Figma JSON.
    
    Handles different JSON structures (direct node, document wrapper, etc.)
    
    Args:
        data: Figma JSON data
    
    Returns:
        Root node dictionary
    """
    if isinstance(data, dict):
        for key in ("document", "node"):
            if key in data and isinstance(data[key], dict) and "children" in data[key]:
                return data[key]
    return data


def ablate_geometry(data: dict) -> dict:
    """
    Remove all geometry information from Figma JSON.
    
    Removes: x, y, width, height, absoluteBoundingBox, absoluteRenderBounds,
    absoluteTransform, relativeTransform, rotation
    
    Args:
        data: Original Figma JSON data
    
    Returns:
        Copy of data with geometry information removed
    """
    data_copy = copy.deepcopy(data)
    
    geom_keys = {
        'x', 'y', 'width', 'height',
        'absoluteBoundingBox', 'absoluteRenderBounds',
        'absoluteTransform', 'relativeTransform', 'rotation'
    }
    
    def prune_geometry(obj):
        if isinstance(obj, dict):
            # Delete geometry keys
            for k in list(obj.keys()):
                if k in geom_keys:
                    del obj[k]
            
            # Clean overrides references
            ovs = obj.get('overrides')
            if isinstance(ovs, list):
                for ov in ovs:
                    if isinstance(ov, dict) and isinstance(ov.get('overriddenFields'), list):
                        ov['overriddenFields'] = [
                            s for s in ov['overriddenFields'] if s not in geom_keys
                        ]
            
            # Recurse
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    prune_geometry(v)
        elif isinstance(obj, list):
            for it in obj:
                prune_geometry(it)
    
    prune_geometry(data_copy)
    
    stats = _count_keys_recursive(data_copy, geom_keys)
    logger.info(f"[ablate_geometry] Remaining geometry keys: {stats}")
    
    return data_copy


def ablate_style(data: dict) -> dict:
    """
    Remove all style information from Figma JSON.
    
    Removes: fills, strokes, effects, fonts, colors, opacity, etc.
    
    Args:
        data: Original Figma JSON data
    
    Returns:
        Copy of data with style information removed
    """
    data_copy = copy.deepcopy(data)
    
    # Defines a broad set of style-related keys, excluding layout geometry
    style_keys = {
        'fills', 'strokes', 'strokeWeight', 'strokeAlign', 'cornerRadius',
        'rectangleCornerRadii', 'effects', 'fontName', 'fontSize', 'fontWeight',
        'letterSpacing', 'lineHeight', 'textAlignHorizontal', 'textAlignVertical',
        'textAutoResize', 'textCase', 'textDecoration',
        # Container or additional styles
        'style', 'styles', 'opacity', 'blendMode', 'cornerSmoothing',
        # Background-related
        'background', 'backgroundColor',
        # Additional text and paragraph styles
        'paragraphSpacing', 'paragraphIndent', 'fontFamily', 'fontPostScriptName', 'lineHeightPx',
        # Grid and stroke supplements
        'layoutGrids', 'gridStyleId', 'strokeCap', 'strokesCap', 'strokeJoin', 'strokeMiterAngle', 'strokeDashes',
        # Override tables
        'characterStyleOverrides', 'styleOverrideTable', 'fillOverrideTable',
    }
    
    def prune_styles(obj):
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                if key in style_keys:
                    del obj[key]
                    continue
                # Delete style reference IDs
                if key == 'styleId' or (isinstance(key, str) and key.endswith('StyleId')):
                    del obj[key]
                    continue
            
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    prune_styles(v)
        elif isinstance(obj, list):
            for it in obj:
                prune_styles(it)
    
    prune_styles(data_copy)
    
    stats = _count_keys_recursive(data_copy, style_keys)
    logger.info(f"[ablate_style] Remaining style keys: {stats}")
    
    return data_copy


def ablate_image_refs(data: dict) -> dict:
    """
    Remove all image references from Figma JSON.
    
    Removes: imageRef, gifRef, imageHash, imageURL, and IMAGE type fills
    
    Args:
        data: Original Figma JSON data
    
    Returns:
        Copy of data with image references removed
    """
    data_copy = copy.deepcopy(data)
    
    image_keys = {
        'imageRef', 'gifRef', 'imageHash', 'imageURL', 'imageUrl', 'previewImage', 'images'
    }
    
    def prune_image_refs(obj):
        if isinstance(obj, dict):
            # Remove image keys
            for key in list(obj.keys()):
                if key in image_keys:
                    del obj[key]
            
            # Filter IMAGE type fills
            if isinstance(obj.get('fills'), list):
                obj['fills'] = [
                    fill for fill in obj['fills'] 
                    if not (isinstance(fill, dict) and fill.get('type') == 'IMAGE')
                ]
            if isinstance(obj.get('background'), list):
                obj['background'] = [
                    bg for bg in obj['background'] 
                    if not (isinstance(bg, dict) and bg.get('type') == 'IMAGE')
                ]
            if isinstance(obj.get('strokes'), list):
                obj['strokes'] = [
                    st for st in obj['strokes'] 
                    if not (isinstance(st, dict) and st.get('type') == 'IMAGE')
                ]
            
            # Filter fillOverrideTable
            if isinstance(obj.get('fillOverrideTable'), dict):
                fot = obj['fillOverrideTable']
                for k, v in list(fot.items()):
                    if isinstance(v, list):
                        fot[k] = [
                            fill for fill in v 
                            if not (isinstance(fill, dict) and fill.get('type') == 'IMAGE')
                        ]
            
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    prune_image_refs(v)
        elif isinstance(obj, list):
            for it in obj:
                prune_image_refs(it)
    
    prune_image_refs(data_copy)
    
    stats = _count_keys_recursive(data_copy, image_keys)
    logger.info(f"[ablate_image_refs] Remaining image-ref keys: {stats}")
    
    return data_copy


def ablate_structure(data: dict) -> dict:
    """
    Flatten the node tree structure.
    
    Collects all nodes and places them as direct children of the root,
    removing nested hierarchy.
    
    Args:
        data: Original Figma JSON data
    
    Returns:
        Copy of data with flattened structure
    """
    data_copy = copy.deepcopy(data)
    root = _get_root_node(data_copy)
    
    if not isinstance(root, dict):
        logger.warning("Not a valid root node to operate on")
        return data_copy
    
    all_nodes_map = {}
    
    def collect_nodes_recursively(obj):
        """Collects all node objects by ID into `all_nodes_map` without modification."""
        if isinstance(obj, dict):
            if 'id' in obj and 'type' in obj:
                if obj['id'] not in all_nodes_map:
                    all_nodes_map[obj['id']] = obj
            
            for value in obj.values():
                collect_nodes_recursively(value)
        elif isinstance(obj, list):
            for item in obj:
                collect_nodes_recursively(item)
    
    collect_nodes_recursively(data_copy)
    
    # Rebuild flattened structure
    flattened_list = [
        node for node_id, node in all_nodes_map.items() 
        if node_id != root.get('id')
    ]
    
    # Clear all children
    for node in flattened_list:
        if isinstance(node, dict):
            node['children'] = []
    
    root['children'] = flattened_list
    
    # Log statistics
    def compute_children_depth_and_nonroot(o, is_root=True, depth=0):
        max_depth = depth
        nonroot_holders = 0
        if isinstance(o, dict):
            ch = o.get('children')
            if isinstance(ch, list) and ch:
                if is_root:
                    max_depth = max(max_depth, depth + 1)
                else:
                    nonroot_holders += 1
                    max_depth = max(max_depth, depth + 1)
                for c in ch:
                    d, n = compute_children_depth_and_nonroot(c, is_root=False, depth=depth + 1)
                    max_depth = max(max_depth, d)
                    nonroot_holders += n
        elif isinstance(o, list):
            for it in o:
                d, n = compute_children_depth_and_nonroot(it, is_root=is_root, depth=depth)
                max_depth = max(max_depth, d)
                nonroot_holders += n
        return max_depth, nonroot_holders
    
    max_depth, nonroot_holders = compute_children_depth_and_nonroot(root, is_root=True, depth=0)
    root_children = root.get('children') if isinstance(root, dict) else []
    root_children_count = len(root_children) if isinstance(root_children, list) else 0
    seen_ids = set()
    duplicate_child_ids = 0
    if isinstance(root_children, list):
        for n in root_children:
            if isinstance(n, dict) and 'id' in n:
                nid = n['id']
                if nid in seen_ids:
                    duplicate_child_ids += 1
                else:
                    seen_ids.add(nid)
    logger.info(f"[ablate_structure] Flatten summary: {{'root_children': {root_children_count}, 'nonroot_children_holders': {nonroot_holders}, 'max_children_depth': {max_depth}, 'duplicate_child_ids': {duplicate_child_ids}}}")
    return data_copy


def ablate_text(data: dict) -> dict:
    """
    Removes text content information:
    - Deletes `characters` field from all nodes.
    - For `TEXT` nodes:
        - Deletes optional `text` field.
        - Anonymizes the `name` field to "TEXT" to prevent semantic leakage.
    - Cleans text-related field names from `overrides.overriddenFields`.

    Note: This ablation aims to preserve structure and style while removing only semantic text.
    
    Args:
        data: Original Figma JSON data
    
    Returns:
        Copy of data with text content removed
    """
    data_copy = copy.deepcopy(data)
    
    text_value_keys = {'characters'}
    text_optional_keys = {'text'}
    
    def prune_text(obj):
        if isinstance(obj, dict):
            node_type = obj.get('type')
            
            # Delete text content keys
            for k in list(obj.keys()):
                if k in text_value_keys:
                    del obj[k]
                    continue
                # Only delete 'text' on TEXT nodes
                if node_type == 'TEXT' and k in text_optional_keys:
                    del obj[k]
                    continue
            
            # Anonymize TEXT node names
            if node_type == 'TEXT' and 'name' in obj and isinstance(obj['name'], str):
                obj['name'] = 'TEXT'
            
            # Clear componentProperties TEXT values
            if 'componentProperties' in obj and isinstance(obj['componentProperties'], dict):
                for ck, cv in obj['componentProperties'].items():
                    if isinstance(cv, dict) and cv.get('type') == 'TEXT' and 'value' in cv:
                        cv['value'] = ""
            
            # Clean overrides
            ovs = obj.get('overrides')
            if isinstance(ovs, list):
                for ov in ovs:
                    if isinstance(ov, dict) and isinstance(ov.get('overriddenFields'), list):
                        ov['overriddenFields'] = [
                            s for s in ov['overriddenFields'] 
                            if s not in (text_value_keys | text_optional_keys)
                        ]
            
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    prune_text(v)
        elif isinstance(obj, list):
            for it in obj:
                prune_text(it)
    
    prune_text(data_copy)
    
    # Statistics: Count remaining text content keys context-sensitively.
    # - Global count of 'characters'.
    # - 'text' key on TEXT nodes (excluding style contexts like styles.text).
    # - Non-empty TEXT values in componentProperties.

    # 1) Global 'characters' count
    char_stats = _count_keys_recursive(data_copy, text_value_keys)

    # 2) 'text' key count on TEXT nodes
    def count_text_on_text_nodes(obj):
        cnt = 0
        def walk(o, parent_type=None, parent_key=None):
            nonlocal cnt
            if isinstance(o, dict):
                node_type = o.get('type', parent_type)
                for k, v in o.items():
                    # Count 'text' key only if the node type is TEXT and parent is not 'styles'
                    if k == 'text' and node_type == 'TEXT' and parent_key != 'styles':
                        cnt += 1
                    walk(v, node_type, k)
            elif isinstance(o, list):
                for it in o:
                    walk(it, parent_type, parent_key)
        walk(obj)
        return cnt
    
    text_on_text = count_text_on_text_nodes(data_copy)

    # 3) Non-empty TEXT value count in componentProperties
    def count_component_prop_text_values(obj):
        cnt = 0
        def walk(o):
            nonlocal cnt
            if isinstance(o, dict):
                if 'componentProperties' in o and isinstance(o['componentProperties'], dict):
                    for cv in o['componentProperties'].values():
                        if isinstance(cv, dict) and cv.get('type') == 'TEXT':
                            val = cv.get('value')
                            if isinstance(val, str) and val.strip() != "":
                                cnt += 1
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for it in o:
                    walk(it)
        walk(obj)
        return cnt

    comp_text_vals = count_component_prop_text_values(data_copy)

    # Log statistics
    stats = {
        'characters': char_stats,
        'text_on_text': text_on_text,
        'component_prop_text_values': comp_text_vals
    }
    logger.info(f"[ablate_text] Remaining text-content keys: {stats}")
    
    return data_copy


# Registry of ablation functions by name
ABLATION_FUNCTIONS: Dict[str, Callable[[dict], dict]] = {
    "geometry": ablate_geometry,
    "style": ablate_style,
    "image_refs": ablate_image_refs,
    "structure": ablate_structure,
    "text": ablate_text,
}


def get_ablation_function(ablation_type: str) -> Callable[[dict], dict]:
    """
    Get ablation function by type name.
    
    Args:
        ablation_type: One of "geometry", "style", "image_refs", "structure", "text"
    
    Returns:
        Ablation function
    
    Raises:
        ValueError: If ablation_type is unknown
    """
    if ablation_type not in ABLATION_FUNCTIONS:
        available = ", ".join(ABLATION_FUNCTIONS.keys())
        raise ValueError(f"Unknown ablation type: {ablation_type}. Available: {available}")
    
    return ABLATION_FUNCTIONS[ablation_type]


def list_ablation_types() -> list:
    """Return list of available ablation type names."""
    return list(ABLATION_FUNCTIONS.keys())
