"""
Figma JSON to Intermediate Representation (IR) converter.

Converts Figma REST API JSON to an intermediate AltNode format
suitable for code generation.

Main features include:

* **Relative Position and Size Calculation**: Converts the absolute coordinates in a
  child node's `absoluteBoundingBox` to coordinates relative to its parent, while
  also copying the width and height [116893540116888†L471-L501].
* **Empty Frame to Rectangle Conversion**: When nodes like `FRAME`, `COMPONENT`,
  `INSTANCE`, or `COMPONENT_SET` have no children, their type is converted to
  `RECTANGLE` [116893540116888†L306-L316].
* **Group Flattening**: The original plugin flattens all `GROUP` nodes, meaning
  the group container is discarded, and its children are directly attached to the
  grandparent node [116893540116888†L331-L374]. This script also flattens all `GROUP`s
  during conversion and recalculates relative coordinates.
* **Unique Naming**: Uses a counter to generate a `uniqueName` for each node,
  ensuring uniqueness [116893540116888†L387-L399].
* **Layout Defaults**: Fills in default values for auto-layout related properties
  (like `layoutMode`, `layoutGrow`) when they are missing, facilitating subsequent
  code generation [116893540116888†L551-L562].
* **Rotation Handling**: When a node has a `rotation` field or a `relativeTransform`
  matrix, it uses `calculate_rectangle_from_bounding_box` to compute the pre-rotation
  dimensions and offset, but does not fully support cumulative rotation (nested
  parent-child rotations) or more complex matrix cases [2941448318545†L47-L89].
* **Text Style Splitting**: Simulates the behavior of `getStyledTextSegments` based
  on `characterStyleOverrides` and `styleOverrideTable` from the REST API, splitting
  text into multiple segments and merging styles [640185572000812†L764-L797].
* **Stroke Weight**: Preserves `strokeWeight`, `strokeWeights`, and
  `individualStrokeWeights` fields, allowing for accurate border rendering later
  [116893540116888†L519-L528].
* **Auto Layout Spacing and Padding Preservation**: Copies `itemSpacing`, `paddingLeft`,
  `paddingRight`, `paddingTop`, `paddingBottom`, etc., to the IRNode, enabling
  the mapping of these values to Tailwind's `gap-*` and `p-*` classes during code
  generation.
* **Layer Order (zIndex)**: Generates a `zIndex` field based on the order of the
  parent's `children` array. According to the Figma plugin API documentation, the
  `children` array is ordered from **back to front**: elements with a higher index
  are positioned further forward [361601496675411†L6543-L6546]. For containers with
  auto layout enabled (`layoutMode` is `HORIZONTAL` or `VERTICAL`), if their
  `itemReverseZIndex` property is true, the order is reversed, meaning elements
  with a lower index are positioned further forward. Top-level nodes are also
  assigned a zIndex based on their list order.
* **Instance Node Expansion**: When a `components_map` parameter is provided and an
  `INSTANCE` node has no children of its own, it looks up the corresponding
  component's `document.children` in the `components_map` using the `componentId`
  and expands it as the instance's subtree. This is useful for remote component
  data that has already been downloaded and appended to the top level of the JSON,
  avoiding further external API calls.

### Known Limitations and Differences from the Original

The following features are still missing or simplified, due to dependencies on the
Figma plugin API or difficulties in implementing them in an offline environment:

* **Cumulative Rotation and Transform Matrices**: Although single-layer rotation
  correction is implemented, cumulative rotation from parent to child is not
  supported, nor is the full `relativeTransform` matrix parsed [2941448318545†L47-L89].
* **Complex Rules for Mixing Absolute Positioning and Auto Layout**: The original
  implementation adjusts `HUG` and `FIXED` dimensions based on whether there are
  absolutely positioned children; this script only converts `HUG` to `FIXED` when
  there are no children [116893540116888†L565-L577].
* **Color Variables, Gradients, and Shadows**: The original plugin resolves color
  variable mappings, gradients, and shadow effects [695566843972746†L124-L132]; this
  information is incomplete in the REST API and is therefore not implemented.
* **Component Instances and Remote Resources**: This script provides basic instance
  expansion capabilities via the `components_map` parameter—if an instance has no
  children and its `componentId` exists in `components_map`, it expands the
  component's `document.children`. However, if the component definition has no
  children or if a component set is needed to select a specific variant, the full
  logic for variant handling and property overrides is not yet implemented.
  * **Layer Sorting**: The original plugin reconstructs the correct stacking order
    by combining `y`/`x` coordinates and the `itemReverseZIndex` of auto layout
    containers. This script generates zIndex based on the `children` array order
    as described in the Figma plugin API documentation: a higher index means the
    element is further forward on the canvas; the order is only reversed if the

    container is an auto layout and `itemReverseZIndex` is true [361601496675411†L6543-L6546].
    However, it does not reorder based on `y`/`x` coordinates, so in some complex
    layouts or hierarchies, the stacking order may still differ slightly from
    Figma's actual rendering.
* **Style Inheritance (Optional)**: If a `styles_map` (usually from the `styles`
  dictionary at the top level of the Figma JSON) is passed when initializing the
  converter, it will attempt to expand style definitions directly into the nodes.
  - When a node lacks fill or stroke colors, the corresponding properties will be
    completed from the `fills` or `strokes` in the style definition; existing
    image fills will not be overwritten.
  - When a text node lacks a default style, it will inherit the `style` dictionary
    and fill color from the style definition, ensuring the text's font, size,
    weight, etc., are consistent with the style.
  - After filling in the style properties, the node's `styles` field is deleted
    to prevent downstream code generation or large model calls from processing
    style IDs again, thus making the IRNode completely independent of style maps.

In summary, this script is suitable for generating an approximate IRNode structure
in an offline environment for use as input in automated workflows. When higher
fidelity or support for more visual effects is required, the official implementation
or an extended script should be run in the Figma plugin environment.
"""

import json
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple

from ...utils.console_logger import logger
from ...utils.files import load_json


def calculate_relative_pos(node_bbox: Dict[str, Any], parent_bbox: Optional[Dict[str, Any]]) -> Tuple[float, float]:
    """
    Calculates relative coordinates based on the absolute bounding boxes of a parent and child node.

    If there is no parent (root node), it returns the absolute coordinates directly; otherwise,
    it returns the child's x and y minus the parent's x and y.

    Args:
        node_bbox: The child's absoluteBoundingBox.
        parent_bbox: The parent's absoluteBoundingBox or None.

    Returns:
        An (x, y) tuple representing the relative coordinates.
    """
    if parent_bbox is None:
        return node_bbox.get('x', 0.0), node_bbox.get('y', 0.0)
    return (
        node_bbox.get('x', 0.0) - parent_bbox.get('x', 0.0),
        node_bbox.get('y', 0.0) - parent_bbox.get('y', 0.0),
    )

# The following helper replicates the behaviour of FigmaToCode's
# `calculateRectangleFromBoundingBox` function.  Given a bounding box
# (representing the rotated frame) and a rotation in degrees, it
# computes the original width and height of the unrotated rectangle and
# returns the top/left coordinate of that rectangle.  See
# packages/backend/src/common/commonPosition.ts in the FigmaToCode repo
# for the original implementation【2941448318545†L47-L89】.
def calculate_rectangle_from_bounding_box(bbox: Dict[str, Any], rotation_deg: float) -> Dict[str, float]:
    """
    Calculates the original rectangle's width, height, and top-left corner from a rotated bounding box.

    FigmaToCode uses the `calculateRectangleFromBoundingBox` function to convert a rotated
    bounding box back to its unrotated dimensions and offset [2941448318545†L47-L89].
    This function is a rewrite based on the mathematical formula from the original implementation.
    It takes a rotated bounding box and a rotation angle as input and outputs the width,
    height, and top-left coordinates of the unrotated rectangle.

    Args:
        bbox: A dictionary containing `x`, `y`, `width`, `height` for the rotated box.
        rotation_deg: The rotation angle, positive for clockwise.

    Returns:
        A dictionary with `width`, `height`, `left`, `top` fields.
    """
    # In FigmaToCode, the rotation angle is negated when passed in
    css_rotation = -rotation_deg
    theta = css_rotation * 3.141592653589793 / 180.0
    # Only absolute values are needed for calculation
    cos_theta = abs(__import__('math').cos(theta))
    sin_theta = abs(__import__('math').sin(theta))
    w_b = bbox.get('width', 0.0)
    h_b = bbox.get('height', 0.0)
    x_b = bbox.get('x', 0.0)
    y_b = bbox.get('y', 0.0)
    # When the rotation angle is close to 0 or 90 degrees, the denominator approaches 0, so return original values
    denominator = cos_theta * cos_theta - sin_theta * sin_theta
    if abs(denominator) < 1e-6:
        return {
            'width': w_b,
            'height': h_b,
            'left': x_b,
            'top': y_b,
        }
    # Solve equations to get original width and height
    h = (w_b * sin_theta - h_b * cos_theta) / -denominator
    w = (w_b - h * sin_theta) / cos_theta
    # Calculate the four vertices of the unrotated rectangle
    from math import cos as cosf, sin as sinf
    rotated_corners = [
        (0, 0),
        (w, 0),
        (w, h),
        (0, h),
    ]
    # Rotate each vertex to find the minimum offset
    rotated = []
    for x, y in rotated_corners:
        rx = x * cosf(theta) + y * sinf(theta)
        ry = -x * sinf(theta) + y * cosf(theta)
        rotated.append((rx, ry))
    min_x = min(c[0] for c in rotated)
    min_y = min(c[1] for c in rotated)
    left = x_b - min_x
    top = y_b - min_y
    return {
        'width': w,
        'height': h,
        'left': left,
        'top': top,
    }


class IRNodeConverter:
    """
    An IRNode converter that mimics the JSON to IRNode conversion logic in the FigmaToCode plugin.

    Each conversion resets the name counter to generate unique names for each node.
    """

    def __init__(self, components_map: Optional[Dict[str, Any]] = None, styles_map: Optional[Dict[str, Any]] = None):
        """
        Initializes the converter.

        Args:
            components_map: A dictionary where keys are component IDs and values are the full
                component definitions (usually from the `components` dictionary at the top
                level of the Figma JSON). If provided, it is used to expand the subtree
                of an `INSTANCE` node when the node itself has no children.

            styles_map: A dictionary where keys are style IDs and values are style definition
                objects (usually from the `styles` dictionary at the top level of the Figma
                JSON). During node conversion, if an IRNode is missing fill or stroke
                colors, this map is used to supplement the color information based on styles.
        """
        # Maintain counters to generate unique names
        self.name_counters: defaultdict[str, int] = defaultdict(int)
        # Store the component map for looking up componentId in instance nodes
        self.components_map = components_map or {}
        # Store the style map to supplement missing fill or stroke based on style ID
        self.styles_map: Dict[str, Any] = styles_map or {}

    def assign_unique_name(self, name: str) -> str:
        """
        Generates a unique name based on the given name: returns the name directly if it's the
        first occurrence, otherwise appends an incrementing number with zero-padding.

        Args:
            name: The original name.

        Returns:
            A unique name.
        """
        clean_name = name.strip()
        count = self.name_counters[clean_name]
        self.name_counters[clean_name] += 1
        if count == 0:
            return clean_name
        return f"{clean_name}_{str(count).zfill(2)}"
    
    def convert_node(self, json_node: Dict[str, Any], parent: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Recursively converts a Figma JSON node to an IRNode dictionary.

        This method handles the following cases:
          * Group expansion: All GROUP nodes are expanded, and their children are directly
            assigned to the grandparent.
          * Empty frames to rectangles: FRAME/COMPONENT/INSTANCE/COMPONENT_SET without children
            are converted to RECTANGLE type.
          * Relative position calculation: Calculates x/y based on the absoluteBoundingBox
            and the parent's box.
          * Rotation correction: If a node has rotation or relativeTransform, it calculates
            the pre-rotation dimensions and offset.
          * Unique naming: Generates a uniqueName for each node.
          * Text style splitting: Splits text into multiple segments based on
            characterStyleOverrides and styleOverrideTable, with each segment having
            merged styles.
          * Stroke weight: Copies strokeWeight/individualStrokeWeights and other related fields.

        Unsupported types (like SLICES) return None. Nodes marked as invisible are also skipped.

        Args:
            json_node: The original JSON node.
            parent: The parent's IRNode, used for calculating relative coordinates; None for root nodes.

        Returns:
            An IRNode dictionary, a list (when expanding a group), or None (if filtered).
        """
        node_type = json_node.get('type')
        # Skip hidden nodes
        if json_node.get('visible') is False:
            return None

        # Convert Frame/Instance/Component without children to a rectangle
        if node_type in {'FRAME', 'INSTANCE', 'COMPONENT', 'COMPONENT_SET'}:
            children = json_node.get('children', [])
            if not children:
                node_type = 'RECTANGLE'

        # Expand groups to match FigmaToCode's behavior: the original plugin always
        # inlines GROUP containers, removing the container and attaching children to
        # the grandparent.
        # This implementation also expands all GROUPs, regardless of the number of children.
        # During expansion, we pass the grandparent node when calling convert_node,
        # so the children's relative coordinates are based on the grandparent. This may
        # ignore rotation or transforms on the group, but it is consistent with the original implementation.
        if node_type == 'GROUP':
            # Use the parent (grandparent of children) for relative positioning
            result_children: List[Dict[str, Any]] = []
            for child in json_node.get('children', []) or []:
                if child.get('visible') is False:
                    continue
                alt_child = self.convert_node(child, parent)
                if alt_child is None:
                    continue
                if isinstance(alt_child, list):
                    result_children.extend(alt_child)
                else:
                    result_children.append(alt_child)
            return result_children

        # Build the IRNode dictionary
        node: Dict[str, Any] = {
            'id': json_node.get('id'),
            'name': json_node.get('name', ''),
            'type': node_type,
            'visible': json_node.get('visible', True),
            'children': [],
        }

        # Assign unique name [116893540116888†L387-L399]
        node['uniqueName'] = self.assign_unique_name(node['name'])

        # Parent-child relationship: To avoid circular references when serializing IRNode,
        # we don't store the parent object directly, but rather the parent's id. This makes
        # it easy to find the parent in subsequent processing without causing circular
        # dependency issues during JSON serialization.
        if parent is not None:
            node['parentId'] = parent.get('id')

      # ---------- Relative Positioning and Sizing (prioritize absoluteRenderBounds) ----------
        def _pick_bounds(n: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            """
            Prioritizes absoluteRenderBounds from a node; falls back to absoluteBoundingBox if it
            doesn't exist; returns None if neither is present.
            """
            if not n:
                return None
            rb = n.get('absoluteRenderBounds')
            if isinstance(rb, dict) and 'x' in rb and 'y' in rb:
                return rb
            return n.get('absoluteBoundingBox')
        bbox_used = _pick_bounds(json_node)                 # Child node prioritizes renderBounds
        parent_bbox_used = _pick_bounds(parent) if parent else None  # Parent also prioritizes renderBounds

        if bbox_used:
            # Calculate rotation angle (your original way: use rotation if present; otherwise estimate from relativeTransform)
            rotation_deg = None
            rot_field = json_node.get('rotation')
            if rot_field is not None:
                rotation_deg = rot_field or 0
            else:
                rt = json_node.get('relativeTransform')
                if rt and len(rt) == 2 and len(rt[0]) >= 2 and len(rt[1]) >= 2:
                    import math
                    m00 = rt[0][0]
                    m10 = rt[1][0]
                    rotation_deg = math.degrees(math.atan2(-m10, m00))
                else:
                    rotation_deg = 0

            if rotation_deg:
                rect = calculate_rectangle_from_bounding_box(bbox_used, rotation_deg)
                if parent_bbox_used:
                    node['x'] = rect['left'] - parent_bbox_used.get('x', 0.0)
                    node['y'] = rect['top']  - parent_bbox_used.get('y', 0.0)
                else:
                    node['x'] = rect['left']
                    node['y'] = rect['top']
                node['width']  = rect['width']
                node['height'] = rect['height']
            else:
                x, y = calculate_relative_pos(bbox_used, parent_bbox_used)
                node['x'] = x
                node['y'] = y
                node['width']  = bbox_used.get('width')
                node['height'] = bbox_used.get('height')

            # Also save both bboxes in the IRNode for downstream selection
            if json_node.get('absoluteRenderBounds'):
                node['absoluteRenderBounds'] = json_node['absoluteRenderBounds']
            if json_node.get('absoluteBoundingBox'):
                node['absoluteBoundingBox'] = json_node['absoluteBoundingBox']
        else:
            # Fallback to your original logic if no bbox is available
            size = json_node.get('size')
            if size and isinstance(size, dict):
                node['width'] = size.get('x')
                node['height'] = size.get('y')
            else:
                node['width'] = json_node.get('width')
                node['height'] = json_node.get('height')
            node['x'] = 0
            node['y'] = 0

        # Default layout properties [116893540116888†L551-L562]
        node['layoutMode'] = json_node.get('layoutMode', 'NONE')
        node['layoutGrow'] = json_node.get('layoutGrow', 0)
        node['layoutSizingHorizontal'] = json_node.get('layoutSizingHorizontal', 'FIXED')
        node['layoutSizingVertical'] = json_node.get('layoutSizingVertical', 'FIXED')
        node['primaryAxisAlignItems'] = json_node.get('primaryAxisAlignItems', 'MIN')
        node['counterAxisAlignItems'] = json_node.get('counterAxisAlignItems', 'MIN')

        # ------------------------------------------------------------------
        # Preserve auto layout spacing and padding properties
        #
        # Figma's auto layout allows setting `itemSpacing` for gaps between
        # child elements, and `paddingLeft`, `paddingRight`, `paddingTop`,
        # `paddingBottom` to define container padding. These values are
        # crucial for generating Tailwind's `gap-*` and `p-*` classes,
        # so they need to be preserved in the IRNode. If these fields
        # are not present in the node, they are ignored.
        for padding_key in [
            'itemSpacing', 'paddingLeft', 'paddingRight', 'paddingTop', 'paddingBottom'
        ]:
            if padding_key in json_node:
                node[padding_key] = json_node[padding_key]

        # Convert 'HUG' sizing to 'FIXED' when there are no children, similar
        # to the original plugin logic [116893540116888†L565-L577].
        has_children = bool(json_node.get('children'))
        if not has_children:
            if node['layoutSizingHorizontal'] == 'HUG':
                node['layoutSizingHorizontal'] = 'FIXED'
            if node['layoutSizingVertical'] == 'HUG':
                node['layoutSizingVertical'] = 'FIXED'

        # Handle text nodes: copy characters and style if available [116893540116888†L401-L466]
        if node_type == 'TEXT':
            node['characters'] = json_node.get('characters', '')
            # Default text style, applied to all characters unless
            # `characterStyleOverrides` specifies an override.
            default_style = json_node.get('style', {}) or {}
            node['style'] = default_style
            
                # These need to be added
            if 'textAutoResize' in json_node:
                node['textAutoResize'] = json_node['textAutoResize']
            if 'textAlignHorizontal' in default_style:
                node['style']['textAlignHorizontal'] = default_style['textAlignHorizontal']
            if 'textAlignVertical' in default_style:
                node['style']['textAlignVertical'] = default_style['textAlignVertical']
            # Build text style segments from REST API's `characterStyleOverrides`
            # and `styleOverrideTable`. They correspond to the TypePropertiesTrait
            # in the API documentation [640185572000812†L764-L797].
            # `characterStyleOverrides` is an integer array, where each element is
            # a key in the style table (needs to be converted to a string); 0 means
            # use the default style. If the array is shorter than the text, missing
            # parts are defaulted to 0.
            # A new segment is started when the override index changes, and each
            # segment merges the default style with the override style.
            overrides = json_node.get('characterStyleOverrides', []) or []
            style_table = json_node.get('styleOverrideTable', {}) or {}

            # Only split text if style overrides exist or the style table is not empty,
            # otherwise use the default style directly. This roughly simulates the
            # behavior of `figma.getStyledTextSegments()` but is implemented with
            # static JSON data [116893540116888†L401-L466].
            if overrides or style_table:
                # Build a list of tuples: (start_index, end_index, style_dict)
                segments: List[Dict[str, Any]] = []
                chars: str = node['characters']
                # To avoid index out of bounds, pad overrides with 0 if it's shorter than the text.
                if len(overrides) < len(chars):
                    overrides = overrides + [0] * (len(chars) - len(overrides))
                prev_index = overrides[0] if overrides else 0
                start = 0
                for idx, override_idx in enumerate(overrides):
                    if override_idx != prev_index:
                        # End the previous style segment
                        segment_text = chars[start:idx]
                        # Merge default style with override style
                        segment_style = default_style.copy()
                        if prev_index != 0:
                            # Keys in styleOverrideTable are strings
                            override_key = str(prev_index)
                            override_style = style_table.get(override_key, {})
                            # Override default style
                            segment_style.update(override_style)
                        segments.append({
                            'text': segment_text,
                            'style': segment_style,
                            'start': start,
                            'end': idx
                        })
                        # Start a new style segment
                        start = idx
                        prev_index = override_idx
                # Append the last style segment
                if start < len(chars):
                    segment_text = chars[start:]
                    segment_style = default_style.copy()
                    if prev_index != 0:
                        override_style = style_table.get(str(prev_index), {})
                        segment_style.update(override_style)
                    segments.append({
                        'text': segment_text,
                        'style': segment_style,
                        'start': start,
                        'end': len(chars)
                    })
                # Generate a unique ID for each style segment, similar to the original implementation.
                # Combine the node's uniqueName and a sequence number.
                base = node['uniqueName'].replace(" ", "").lower()
                styled_segments = []
                for i, seg in enumerate(segments):
                    unique_id = f"{base}_span" + (f"_{str(i+1).zfill(2)}" if len(segments) > 1 else "")
                    styled_segments.append({
                        'uniqueId': unique_id,
                        'text': seg['text'],
                        'style': seg['style'],
                        'start': seg['start'],
                        'end': seg['end']
                    })
                node['styledTextSegments'] = styled_segments

        # Copy stroke weight related fields: the REST API may provide a uniform
        # `strokeWeight`, or individual `strokeWeights`/`individualStrokeWeights`
        # for each side. We attach them directly to the IRNode for later use in
        # rendering precise borders.
        if 'strokeWeight' in json_node:
            node['strokeWeight'] = json_node['strokeWeight']
        if 'strokeWeights' in json_node:
            node['strokeWeights'] = json_node['strokeWeights']
        if 'individualStrokeWeights' in json_node:
            node['strokeTopWeight'] = json_node['individualStrokeWeights'].get('top')
            node['strokeBottomWeight'] = json_node['individualStrokeWeights'].get('bottom')
            node['strokeLeftWeight'] = json_node['individualStrokeWeights'].get('left')
            node['strokeRightWeight'] = json_node['individualStrokeWeights'].get('right')

        # If the node defines layout constraints, copy them to the IRNode.
        # Constraints describe how a child node behaves when its parent container is resized,
        # with horizontal/vertical options like MIN, CENTER, MAX, SCALE, STRETCH; older
        # versions had LEFT/RIGHT/TOP/BOTTOM [640185572000812†L203-L238]. Preserving this
        # information helps in generating responsive layouts downstream.
        if 'constraints' in json_node:
            node['constraints'] = json_node['constraints']

        # Handle instance nodes: if this node is an INSTANCE and has no children, try to
        # expand its component definition's document from the pre-loaded component map.
        # This helps restore the internal structure of components in a simplified JSON.
        children_to_process = json_node.get('children', []) or []
        if node_type == 'INSTANCE':
            # If the instance itself has no children, try to use the component definition's document
            if not children_to_process:
                comp_id = json_node.get('componentId')
                if comp_id and comp_id in self.components_map:
                    remote_component = self.components_map[comp_id]
                    # The remote_component structure might contain a document node
                    remote_doc = remote_component.get('document', {}) if isinstance(remote_component, dict) else {}
                    remote_children = remote_doc.get('children', []) or []
                    children_to_process = remote_children
        # Calculate zIndex for children and process them recursively
        #
        # Figma's rendering order is determined by the order of the parent's children array [361601496675411†L6543-L6546].
        # By default, elements with a higher index in the children array are rendered further
        # forward on the canvas (i.e., have a higher zIndex).
        # For auto layout containers (layoutMode is HORIZONTAL or VERTICAL), if
        # itemReverseZIndex is true, the order is reversed, meaning elements with a lower
        # index are on top.
        total_children = len(children_to_process)
        # Check if the current parent is an auto layout container with reverse stacking enabled
        reverse_z = False
        layout_mode = json_node.get('layoutMode')
        if layout_mode in {'HORIZONTAL', 'VERTICAL'} and json_node.get('itemReverseZIndex'):
            reverse_z = True
        for idx, child in enumerate(children_to_process):
            alt_child = self.convert_node(child, node)
            if alt_child is None:
                continue
            # Calculate zIndex: determine index mapping based on whether it's reversed
            if reverse_z:
                z_val = total_children - 1 - idx
            else:
                z_val = idx
            # If expanding a group returns an array, set the same zIndex for each child
            if isinstance(alt_child, list):
                for sub in alt_child:
                    if isinstance(sub, dict):
                        sub['zIndex'] = z_val
                node['children'].extend(alt_child)
            else:
                if isinstance(alt_child, dict):
                    alt_child['zIndex'] = z_val
                node['children'].append(alt_child)

        # Determine canBeFlattened (icon detection). Without runtime API we
        # approximate icons by primitive types used for icons [605162198890559†L8-L15].
        # The original implementation also checks export settings and size; here
        # we consider nodes with types VECTOR, BOOLEAN_OPERATION, POLYGON or STAR as icons.
        if node_type in {'VECTOR', 'BOOLEAN_OPERATION', 'POLYGON', 'STAR'}:
            node['canBeFlattened'] = True
        else:
            # When a node is of type IMAGE and contains an imageRef (usually from external
            # tools merging vector groups into an image), we also consider it flattenable
            # as a single resource. Otherwise, default to False.
            if node_type == 'IMAGE' and (
                ('imageRef' in node) or
                any(isinstance(f, dict) and f.get('type') == 'IMAGE' for f in node.get('fills', []) if isinstance(node.get('fills'), list))
            ):
                node['canBeFlattened'] = True
            else:
                node['canBeFlattened'] = False
                
        # Add more properties in the "Additional Field Copying" section
        # Copy corner radius
        if 'cornerRadius' in json_node:
            node['cornerRadius'] = json_node['cornerRadius']
        if 'rectangleCornerRadii' in json_node:
            node['rectangleCornerRadii'] = json_node['rectangleCornerRadii']

        # Copy effects (shadows, blurs, etc.)
        if 'effects' in json_node and json_node['effects']:
            try:
                node['effects'] = json.loads(json.dumps(json_node['effects']))
            except Exception:
                node['effects'] = json_node['effects']

        # Copy blend mode
        if 'blendMode' in json_node:
            node['blendMode'] = json_node['blendMode']

        # Copy opacity
        if 'opacity' in json_node:
            node['opacity'] = json_node['opacity']

        # Copy clipping settings
        if 'clipsContent' in json_node:
            node['clipsContent'] = json_node['clipsContent']

        # Copy export settings
        if 'exportSettings' in json_node:
            try:
                node['exportSettings'] = json.loads(json.dumps(json_node['exportSettings']))
            except Exception:
                node['exportSettings'] = json_node['exportSettings']

        # ------------------------------------------------------------------
        # Additional Field Copying
        #
        # In the user-provided simplified Figma JSON, some resources (images/vector groups)
        # have already been rendered as local files and placed within the fills array
        # via `imageRef`. To preserve this information in the IRNode, we need to
        # copy fields like fill and stroke as-is, and retain the relative path
        # specified by imageRef.
        #
        # 1. Copy fills: This may contain type="IMAGE" with an imageRef. This
        #    array is used later to set background images or SVGs when generating
        #    HTML/Tailwind.
        # 2. Copy strokes: Used to preserve stroke color and style. If not copied,
        #    the border color cannot be restored during rendering.
        # 3. Copy strokeAlign: Indicates the position of the stroke on the element (inside/outside).
        # 4. Copy componentId/componentProperties/overrides: These fields can be
        #    used to locate the master component or override properties when parsing
        #    component instances.
        #
        # Note: The presence of these fields depends on whether the input JSON contains
        # them; missing fields will not affect the IRNode data structure.
        if 'fills' in json_node:
            # Deep copy fills to avoid modifying the original data later
            try:
                node['fills'] = json.loads(json.dumps(json_node['fills']))
            except Exception:
                # If fills is not a JSON serializable structure, reference it directly
                node['fills'] = json_node['fills']
        if 'strokes' in json_node:
            try:
                node['strokes'] = json.loads(json.dumps(json_node['strokes']))
            except Exception:
                node['strokes'] = json_node['strokes']
        # Copy stroke alignment mode, e.g., INSIDE/OUTSIDE/CENTER
        if 'strokeAlign' in json_node:
            node['strokeAlign'] = json_node['strokeAlign']
        # Component instance identifier and override table
        if 'componentId' in json_node:
            node['componentId'] = json_node['componentId']
        if 'componentProperties' in json_node and json_node['componentProperties']:
            node['componentProperties'] = json_node['componentProperties']
        if 'overrides' in json_node and json_node['overrides']:
            node['overrides'] = json_node['overrides']

        # Some nodes may contain a complete styles object, used to define a list
        # of color or text styles. These style entries are already expanded in the
        # simplified JSON, and this implementation copies them directly to the
        # IRNode for easier processing later.
        if 'styles' in json_node and json_node['styles']:
            try:
                node['styles'] = json.loads(json.dumps(json_node['styles']))
            except Exception:
                node['styles'] = json_node['styles']

            # If a style map is provided, try to complete missing fill or stroke
            # colors based on the style definitions. Many simplified nodes may only
            # have a style ID; this logic looks up the definition in the styles_map
            # using the style ID and injects color information when the node lacks
            # explicit fills or strokes.
            if self.styles_map:
                # Iterate through the style map and try to inherit missing style properties from the style definition
                for style_key, style_id in node['styles'].items():
                    # style_key could be fill/fills/stroke/strokes/text, etc.
                    if not isinstance(style_id, str):
                        continue
                    style_def = self.styles_map.get(style_id)
                    if not style_def:
                        continue
                    style_doc = style_def.get('document', {}) if isinstance(style_def, dict) else {}
                    # Process by style type
                    key_lower = style_key.lower()
                    # 1. Fill style
                    if key_lower.startswith('fill'):
                        # If the node itself does not specify fills, inherit fills from the style definition
                        if not node.get('fills'):
                            style_fills = style_doc.get('fills')
                            if style_fills:
                                try:
                                    node['fills'] = json.loads(json.dumps(style_fills))
                                except Exception:
                                    node['fills'] = style_fills
                    # 2. Stroke style
                    elif key_lower.startswith('stroke'):
                        # If the node itself does not define strokes, inherit them from the style
                        if not node.get('strokes'):
                            style_strokes = style_doc.get('strokes')
                            if style_strokes:
                                try:
                                    node['strokes'] = json.loads(json.dumps(style_strokes))
                                except Exception:
                                    node['strokes'] = style_strokes
                            # Also inherit stroke weight and alignment
                            if 'strokeWeight' in style_doc and 'strokeWeight' not in node:
                                node['strokeWeight'] = style_doc['strokeWeight']
                            if 'strokeAlign' in style_doc and 'strokeAlign' not in node:
                                node['strokeAlign'] = style_doc['strokeAlign']
                    # 3. Text style
                    elif key_lower.startswith('text'):
                        # Only inherit styles on text nodes
                        if node_type == 'TEXT':
                            # If the default style is empty, use the style from the style definition
                            if not node.get('style'):
                                style_style = style_doc.get('style')
                                if style_style:
                                    try:
                                        node['style'] = json.loads(json.dumps(style_style))
                                    except Exception:
                                        node['style'] = style_style
                            # If the text lacks a fill color, inherit fills from the style
                            if not node.get('fills'):
                                style_fills = style_doc.get('fills')
                                if style_fills:
                                    try:
                                        node['fills'] = json.loads(json.dumps(style_fills))
                                    except Exception:
                                        node['fills'] = style_fills
                # After inheriting styles, remove the styles field to prevent downstream from processing style IDs again
                del node['styles']

        # If the node itself directly provides an imageRef (usually for merged vector groups),
        # also preserve it in the IRNode. This property is not typically found in the
        # original REST API, but a user might add it during simplification.
        # [Note: This field has not been found in our test files so far].
        if 'imageRef' in json_node:
            node['imageRef'] = json_node['imageRef']

        return node

    def convert(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Converts a list of top-level Figma JSON nodes to a list of IRNodes.

        This method resets the name counter and calls `convert_node()` in a loop.

        Args:
            nodes: The list of top-level nodes to convert.

        Returns:
            A list of IRNodes. If expanding a group returns a list, it will be merged into the result.
        """
        self.name_counters.clear()
        result: List[Dict[str, Any]] = []
        for idx, node in enumerate(nodes):
            ir_node = self.convert_node(node, None)
            if ir_node is None:
                continue
            # Flatten arrays (for inlined group results)
            if isinstance(ir_node, list):
                # Top-level expanded groups are uncommon in this scenario, but are still added to the result in order
                result.extend(ir_node)
            else:
                result.append(ir_node)
        # Set zIndex for top-level nodes: higher index means higher layer
        total_top = len(result)
        for idx, node in enumerate(result):
            # If the top-level page has itemReverseZIndex set, the order can be adjusted
            # Since top-level nodes don't have a parent json_node, we process in default order here
            node['zIndex'] = idx
        return result


def select_top_level_nodes(figma_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Selects the top-level nodes to be converted, compatible with two export formats:
    1) document.type == 'DOCUMENT', where children are PAGEs;
    2) The document itself is a FRAME/canvas, and the real elements are in document.children.
    """
    document = figma_json.get('document', {}) or {}
    children = document.get('children', []) or []

    if not children:
        # No children, return empty
        return []

    # Format A: Has PAGEs
    has_page = any(ch.get('type') == 'PAGE' for ch in children)
    if has_page:
        top_nodes: List[Dict[str, Any]] = []
        for page in children:
            if page.get('type') == 'PAGE':
                top_nodes.extend(page.get('children', []) or [])
        return top_nodes

    # Format B: The document is a Frame/canvas, and the real elements are in children
    return children

def FigmatoIR(metda_json_path: str, save_path : str) -> List[Dict[str, Any]]:
    figma_data = load_json(metda_json_path)
    nodes = select_top_level_nodes(figma_data)
    # Pass the component map to expand components when processing INSTANCE nodes
    components_map = figma_data.get('components', {}) if isinstance(figma_data, dict) else {}
    # Pass the style map to complete missing colors based on styleId during conversion
    styles_map = figma_data.get('styles', {}) if isinstance(figma_data, dict) else {}
    converter = IRNodeConverter(components_map=components_map, styles_map=styles_map)
    alt_nodes = converter.convert(nodes)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(alt_nodes, f, indent=2, ensure_ascii=False)
    logger.info(f"Conversion completed for {len(alt_nodes)} nodes, saved to {save_path}")
    
    return alt_nodes
