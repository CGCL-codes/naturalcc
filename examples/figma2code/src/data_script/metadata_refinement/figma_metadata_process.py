"""
The FigmaMetaDataProcess class is used to process Figma data, implementing a complete workflow for data acquisition, preprocessing, resource collection, and downloading.

Main functionalities include:
- Data Acquisition: Get raw JSON data and rendered images of the root node.
- Data Preprocessing: Node optimization (attribute filtering, removal of useless nodes, merging of asset nodes, coordinate normalization).
- Layer Optimization: Redundant nesting filtering, layout recognition.
- Resource Collection: Collect various resource IDs and keys.
- Resource Downloading: Download all necessary resource files.
- Data Statistics: Generate detailed processing reports.
- Batch Processing: Support batch processing via whitelist.csv.
- Staged Cleanup: Level 1 cleanup for processed_metadata.json + report.json, Level 2 for assets, Level 3 for the entire folder.

Output Structure:
```
output/processed/
└── {file_key}_{safe_filename(node_id)}/
    ├── raw.json                    # Raw Figma data
    ├── processed_metadata.json     # Processed metadata
    ├── report.json                 # Detailed processing report
    ├── root.png                    # Rendered image of the root node
    ├── assets/
    |   ├── components_json/        # Component JSON data
    |   ├── component_sets_json/    # Component set JSON data
    |   ├── styles_json/            # Style JSON data
    |   ├── image_refs/             # imageRef images
    |   └── svg_assets/             # SVG asset files
    └── render/
        ├── components_img/         # Rendered component images
        └── nodes_img/              # Rendered node images
```
"""

import os
import shutil
from typing import Dict, List, Any
from ...utils.figma_utils import (
    FigmaSession,
    find_imageref_in_json,
    safe_filename,
    compare_images,
    get_node_statics,
)
from ...utils.files import save_json, load_json
from ...utils.console_logger import *
import csv
from copy import deepcopy
import traceback
import sys

process_config = {
    "move_to_error_dir": False,             # Move failed data to the error directory (set to False for debugging)
    "remove_useless_attributes": True,      # Remove useless attributes
    "remove_system_components": False,      # Remove system components
    "remove_empty_attributes": False,        # Remove empty attributes
    "add_layout_inference": False,          # Infer layout for nodes without auto-layout (not yet implemented)
    "add_render_path": False,               # Add paths for rendered images
    "enable_resource_deduplication": True,  # Enable resource deduplication
    "prune_duplicate_nodes": False,          # Simplify duplicate nodes
    "remove_dup_file":False,                # Actually delete duplicate resource files
    "add_details": True,                    # Add details for component, componentSet, and style nodes
    "precision": 3,                         # Numerical precision for floats (None to keep original)
    "change_svg_node_to_rectangle": True,  # Convert SVG nodes to RECTANGLE nodes and place as imageRef in fills
}


class FigmaMetaDataProcess:
    def __init__(self, figma_session: FigmaSession = None, output_dir: str = "output/processed"):
        """
        Initializes the Figma metadata processor.
        
        Args:
            figma_session: Figma session object; a new one is created if None.
            output_dir: The output directory.
        """
        self.figma_session = figma_session or FigmaSession()
        self.output_dir = output_dir
        self.error_dir = "output/error"
        setup_logging(logger, output_dir=output_dir, log_name="figma_metadata_process")
        self.logger = logger

        self.resource_doc = {}   # Mapping from component/componentSet/style node ID to its JSON data
        self.processed_data = {} # The processed data
        self.page_node = {}      # The page node

        self.resource_info = {} # Information about resources to be downloaded
        self.resource_downloaded = {} # Mapping from ID/key to the downloaded resource path (relative to the page directory)

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)

        self.container_types = ['GROUP', 'FRAME']
        
        # List of useless attributes not required for code generation
        # Based on Figma API documentation: https://www.figma.com/developers/api#files
        # TODO: To be revised
        self.useless_attributes = {
            # === Editor-specific attributes ===
            'locked',                 # Locked
            'exportSettings',         # Export settings

            'visible',                # Visibility (invisible nodes are already removed)

            # === Deprecated attributes ===
            'background',
            'backgroundColor'         # Kept in JSON but fills are used in practice
        }
        
        # Keywords for identifying and removing system-level UI components
        # TODO: To be revised
        self.system_component_keywords = [
            'status bar', 'statusbar', 'iphone status', 'android status',
            'navigation bar', 'navbar', 'android navigation', 'home indicator',
            'safe area', 'notch', 'dynamic island', 'system ui', 'battery',
            'signal', 'wifi', 'time', 'carrier'
        ]
        
        # Complete list of basic shape and vector node types
        self.base_node_types = {'VECTOR', 'STAR', 'LINE', 'ELLIPSE', 'REGULAR_POLYGON', 'RECTANGLE'}
        
        # Shape node types to be merged into SVG
        self.svg_merge_types = {'VECTOR', 'STAR', 'LINE', 'ELLIPSE', 'REGULAR_POLYGON',
                                'BOOLEAN_OPERATION'}
        self.svg_merge_keywords = ['merge']

        # Useless attributes to be removed from merged nodes
        # TODO: To be supplemented
        self.merge_filter_attributes1 = ['children', 'rotation', 'opacity'] # Attributes removed upon any merge
        self.merge_filter_attributes2 = ['fills', 'strokes', 'strokeWeight', 'strokeAlign'] # Additional attributes removed if not converted to RECTANGLE
        
        # Attributes to keep during deduplication
        self.duplicate_attributes = ['id', 'name', 'type', 'absoluteBoundingBox', 'absoluteRenderBounds', 'asset_path', 'render_path']
        # Resource deduplication mapping
        # Format: {resource_file_path: [resource_id_1, resource_id_2, ...]}
        self.duplicate_resource_mapping = {}

        # Check if the configuration is reasonable
        if process_config['prune_duplicate_nodes'] or process_config['remove_dup_file']:
            assert process_config['enable_resource_deduplication'], "Resource deduplication must be enabled for this operation."

    def _move_to_error_folder(self, output_dir: str, task_id: str) -> None:
        """Moves failed data to the error folder."""
        try:
            if os.path.exists(output_dir):
                error_dir = os.path.join(self.error_dir, os.path.basename(output_dir))
                if os.path.exists(error_dir):
                    shutil.rmtree(error_dir)
                shutil.move(output_dir, error_dir)
                self.logger.info(f"{task_id} data moved to error folder: {error_dir}")
        except Exception as e:
            self.logger.error(f"{task_id} failed to move to error folder: {e}")

    def process_figma_metadata(self, file_key: str, node_id: str = None, annotation: Dict[str, Any] = None) -> None:
        """
        Main function: The complete workflow for processing Figma metadata.
        
        Args:
            file_key: The key of the Figma file.
            node_id: The specific node ID; if None, the entire file is processed.
            annotation: Annotation properties.
        """
        task_id = safe_filename(f"{file_key}_{node_id}")
        output_dir = os.path.join(self.output_dir, task_id)
        self.logger.info(f"{task_id} starting processing...")
        
        try:
            os.makedirs(output_dir, exist_ok=True)

            # Data acquisition
            raw_data = self._acquire_data(file_key, node_id, output_dir)
            self.processed_data = deepcopy(raw_data)
            self.page_node = self.processed_data.get('document')
            
            # Page compression
            self._compress_node(self.page_node)
            
            # Process resources
            self._process_resources(file_key, output_dir)
            
            # Simplify duplicate nodes (optional)
            if process_config['prune_duplicate_nodes']:
                self._simplify_duplicate_nodes()
            
            # Generate statistics
            statistics = self._generate_statistics()
            
            # Merge page and resource_doc
            if process_config['add_details']:
                self._merge_page_and_resource_doc()

            # Save processed data
            processed_file = os.path.join(output_dir, "processed_metadata.json")
            save_json(self.processed_data, processed_file)
            
            # Save statistics report
            report = {
                'file_key': file_key,
                'node_id': node_id,
                'page_url': f'https://www.figma.com/design/{file_key}/?node-id={node_id}',
                'annotation': annotation,
                'statistics': statistics,
                'resource_info': self.resource_info,
                'downloaded_resources': self.resource_downloaded,
                'duplicate_mapping': self.duplicate_resource_mapping if process_config['enable_resource_deduplication'] else None
            }
            report_file = os.path.join(output_dir, "report.json")
            save_json(report, report_file)
            
            self.logger.info(f"{task_id} processing completed successfully.")
            return True
            
        except Exception as e:
            error_msg = f"An error occurred during processing: {str(e)}, {traceback.format_exc()}"
            self.logger.error(f"{task_id} {error_msg}")
            if process_config['move_to_error_dir']:
                self._move_to_error_folder(output_dir, task_id)
            return False

    def _acquire_data(self, file_key: str, node_id: str, output_dir: str) -> Dict[str, Any]:
        """Data Acquisition: Get raw JSON data and rendered image of the root node."""
        
        # Get raw JSON data
        raw_json_path = os.path.join(output_dir, "raw.json")
        if not os.path.exists(raw_json_path):
            raw_data = self.figma_session.get_node_info(file_key, node_id)
            if raw_data.get('document',{}).get('type') != 'FRAME':
                raise ValueError("Failed to get FRAME node information.")
            save_json(raw_data, raw_json_path)
        else:
            raw_data = load_json(raw_json_path)
                    
        # Download rendered image of the root node
        root_image_path = os.path.join(output_dir, "root.png")
        if not os.path.exists(root_image_path):
            target_node_id = node_id if node_id else list(raw_data.get('document', {}).get('children', [{}]))[0].get('id')
            if target_node_id:
                download_urls, failed_ids = self.figma_session.get_render_image_urls(file_key, [target_node_id])
                if target_node_id in download_urls:
                    file_ext = self.figma_session.download_image_from_url(
                        download_urls[target_node_id], output_dir, "root"
                    )
                else:
                    raise ValueError("Failed to get root node image.")
        
        return raw_data

    def _compress_node(self, root_node: Dict[str, Any], is_page: bool = True) -> None:
        """Node compression, including node optimization and layer optimization."""
        def remove_useless_nodes(root_node: Dict[str, Any]) -> None:
            """Remove useless nodes."""
            remove_node_id = []
            
            def has_rotation(node_data: Dict[str, Any]) -> bool:
                """Check if the node has rotation."""
                return abs(node_data.get('rotation', 0)) > 1e-6
            
            def has_styles(node_data: Dict[str, Any]) -> bool:
                """Check if the node has any styles."""
                # Note: No need to consider effects here, as they require fills or strokes to be active.
                return node_data.get('fills', []) or node_data.get('strokes', []) 
            
            def get_bbox(node_data: Dict[str, Any]) -> Dict[str, float]:
                """Get the bounding box of the node, preferring render bounds over theoretical bounds."""
                bbox = node_data.get('absoluteRenderBounds', {})
                if not bbox:
                    bbox = node_data.get('absoluteBoundingBox', {})
                return {
                    'x': bbox.get('x', 0),
                    'y': bbox.get('y', 0),
                    'width': bbox.get('width', 0),
                    'height': bbox.get('height', 0)
                }
            
            def bbox_completely_outside(child_bbox: Dict[str, float], parent_bbox: Dict[str, float]) -> bool:
                """Check if the child's bounding box is completely outside the parent's."""
                child_right = child_bbox['x'] + child_bbox['width']
                child_bottom = child_bbox['y'] + child_bbox['height']
                parent_right = parent_bbox['x'] + parent_bbox['width']
                parent_bottom = parent_bbox['y'] + parent_bbox['height']
                
                return (child_right <= parent_bbox['x'] or 
                    child_bbox['x'] >= parent_right or
                    child_bottom <= parent_bbox['y'] or
                    child_bbox['y'] >= parent_bottom)
            
            def bbox_completely_covered(lower_bbox: Dict[str, float], upper_bbox: Dict[str, float]) -> bool:
                """Check if the lower node's bounding box is completely covered by the upper one."""
                return (upper_bbox['x'] <= lower_bbox['x'] and
                    upper_bbox['y'] <= lower_bbox['y'] and
                    upper_bbox['x'] + upper_bbox['width'] >= lower_bbox['x'] + lower_bbox['width'] and
                    upper_bbox['y'] + upper_bbox['height'] >= lower_bbox['y'] + lower_bbox['height'])
            
            def is_opaque_solid_fill(fill: Dict[str, Any]) -> bool:
                """Check if the fill is an opaque SOLID fill."""
                return (fill.get('type') == 'SOLID' and
                        fill.get('blendMode', 'NORMAL') == 'NORMAL' and
                        fill.get('opacity', 1) == 1)

            def has_opaque_solid_fill(node_data: Dict[str, Any]) -> bool:
                """Check if the node has an opaque SOLID fill."""
                fills = node_data.get('fills', [])
                if not isinstance(fills, list):
                    return False
                
                for fill in fills:
                    if is_opaque_solid_fill(fill):
                        return True
                return False
            
            def can_cover_other_nodes(node_data: Dict[str, Any]) -> bool:
                """Check if the node can cover other nodes."""
                node_type = node_data.get('type')
                return (not has_rotation(node_data) and
                    node_type in ['GROUP', 'FRAME', 'RECTANGLE'] and
                    has_opaque_solid_fill(node_data))
            
            def can_be_covered(node_data: Dict[str, Any]) -> bool:
                """Check if the node can be covered by other nodes."""
                return not has_rotation(node_data)
            
            def remove_invisible_fills(node_data: Dict[str, Any]) -> None:
                """Remove invisible or covered fills."""
                fills = node_data.get('fills', [])
                if not fills:
                    return
                
                # Traverse from back to front (top to bottom layer) to remove covered fills
                visible_fills = []
                has_opaque_cover = False
                
                for fill in reversed(fills):
                    if not isinstance(fill, dict):
                        continue
                    
                    # Check if the fill is visible
                    if (fill.get('visible', True) is False or 
                        fill.get('opacity', 1) == 0):
                        continue
                    
                    # If there's an opaque cover, skip lower fills
                    if has_opaque_cover:
                        continue
                    
                    # Lower layer, rendered first, insert at the beginning of the list
                    visible_fills.insert(0, fill)
                    
                    # Check if the current fill forms an opaque cover
                    if is_opaque_solid_fill(fill):
                        has_opaque_cover = True
                
                if len(visible_fills) != len(fills):
                    node_data['fills'] = visible_fills
                    self.logger.debug(f"Removed invisible fills: {node_data.get('id', '')}")
            
            def should_remove_node(node_data: Dict[str, Any], parent_data: Dict[str, Any] = None, upper_cover_nodes: List[Dict[str, Any]] = None) -> bool:
                # Remove system components (optional)
                if process_config['remove_system_components']:
                    node_name = node_data.get('name', '').lower()
                    for keyword in self.system_component_keywords:
                        if keyword in node_name:
                            return True
                # Remove empty container nodes (no styles, no children)
                if node_data.get('type') in self.container_types and not node_data.get('children', []) and not has_styles(node_data):
                    return True
                
                # Remove invisible or transparent nodes
                if node_data.get('visible') is False or node_data.get('opacity') == 0:
                    return True
                
                # Remove nodes without a bounding box (these are abnormal and should be removable)
                if node_data.get('absoluteBoundingBox') is None:
                    return True
                
                # Remove shape or VECTOR nodes without any fills/strokes/effects
                if node_data.get('type') in self.base_node_types and not has_styles(node_data):
                    return True
                
                # Remove nodes completely outside their parent container
                if parent_data is not None:
                    node_bbox = get_bbox(node_data)
                    parent_bbox = get_bbox(parent_data)
                    if bbox_completely_outside(node_bbox, parent_bbox):
                        return True
                
                # Remove nodes completely obscured by upper sibling nodes
                if upper_cover_nodes and can_be_covered(node_data):
                    node_bbox = get_bbox(node_data)
                    for sibling in upper_cover_nodes:
                        sibling_bbox = get_bbox(sibling)
                        if bbox_completely_covered(node_bbox, sibling_bbox):
                            return True
                
                return False
            
            def should_reverse_index(node_data: Dict[str, Any]) -> bool:
                """Check if node traversal needs to be reversed."""
                layout_mode = node_data.get('layoutMode', 'NONE')
                if (layout_mode == 'HORIZONTAL' or layout_mode == 'VERTICAL'):
                    # When itemReverseZIndex is true, the first layer will be drawn on top. 
                    # This property is only applicable for auto-layout frames.
                    return not node_data.get('itemReverseZIndex', False)
                return True
        
            def remove_nodes_recursive(node_data: Dict[str, Any]) -> None:
                if not isinstance(node_data, dict) or 'children' not in node_data:
                    return
                
                # Filter child nodes
                original_children = node_data['children']
                upper_cover_nodes = []
                filtered_children = []

                should_reverse = should_reverse_index(node_data)
                if should_reverse:
                    original_children.reverse()

                # From later-rendered upper nodes to earlier-rendered lower nodes
                for child in original_children:
                    # First, remove invisible fills
                    remove_invisible_fills(child)

                    tmp_upper_cover_nodes = upper_cover_nodes.copy()
                    if not should_remove_node(child, node_data, upper_cover_nodes):
                        remove_nodes_recursive(child)

                        # Re-check after recursion to ensure it doesn't need to be removed
                        # e.g., a container node with no styles whose children have all been removed
                        if not should_remove_node(child, node_data, tmp_upper_cover_nodes):
                            filtered_children.append(child)
                            if can_cover_other_nodes(child):
                                upper_cover_nodes.append(child)
                            continue
                    remove_node_id.append(child.get('id', ''))
                if should_reverse:
                    filtered_children.reverse()
                node_data['children'] = filtered_children
            
            remove_nodes_recursive(root_node)
            self.logger.debug(f"Removed nodes: {remove_node_id}")

        def merge_asset_nodes(root_node: Dict[str, Any]) -> None:
            """Merge asset nodes."""
            
            def should_merge_as_svg(node_data: Dict[str, Any]) -> bool:
                """Determine if a node should be merged as an SVG."""
                node_type = node_data.get('type')
                node_name = node_data.get('name', '').lower()
                
                # Single non-mask shape node
                if node_type in self.svg_merge_types and not node_data.get('isMask',False):
                    return True
                
                # Name contains merge keyword
                for keyword in self.svg_merge_keywords:
                    if keyword in node_name:
                        return True
                
                # GROUP or FRAME containing multiple non-mask shape nodes
                if node_type in self.container_types:
                    children = node_data.get('children', [])
                    if children and all(child.get('type') in self.svg_merge_types and not child.get('isMask',False)  for child in children):
                        return True
                
                return False
            
            def filter_node_attributes(node_data: Dict[str, Any]) -> None:
                """Filter node attributes."""
                for key in self.merge_filter_attributes1:
                    node_data.pop(key, None)
                if not process_config['change_svg_node_to_rectangle']:
                    for key in self.merge_filter_attributes2:
                        node_data.pop(key, None)
            
            def merge_nodes_recursive(node_data: Dict[str, Any]) -> None:
                # Check if the current node needs to be merged
                if should_merge_as_svg(node_data):
                    node_data['type'] = 'SVG_ASSET'
                    filter_node_attributes(node_data)
                else:
                    # Recursively process child nodes
                    if 'children' in node_data:
                        for child in node_data['children']:
                            merge_nodes_recursive(child)
            
            merge_nodes_recursive(root_node)

        def filter_node_attributes(root_node: Dict[str, Any]) -> None:
            """Filter node attributes: remove useless attributes (optional)."""
            
            def filter_node_recursive(node_data: Dict[str, Any]) -> Dict[str, Any]:
                if not isinstance(node_data, dict):
                    return node_data
                    
                filtered_node = {}
                for key, value in node_data.items():
                    if key in self.useless_attributes:
                        continue
                    if process_config['remove_empty_attributes'] and not value:
                        continue
                    if key == 'children' and isinstance(value, list):
                        filtered_node[key] = [filter_node_recursive(child) for child in value]
                    else:
                        filtered_node[key] = value
                
                return filtered_node
            
            # Recursively process the node tree
            if process_config['remove_useless_attributes']:
                filtered_data = filter_node_recursive(root_node)
                root_node.clear()
                root_node.update(filtered_data)

        def normalize_num_value(root_node: Dict[str, Any]) -> None:
            """
            Normalize numerical values:
            1) Convert absoluteBoundingBox and absoluteRenderBounds coordinates to be relative to the root node.
            2) Round all floats in the JSON according to process_config["precision"] (if precision is not None).

            Args:
                root_node: The node data to process, serving as the root.
            """
            # Use the root node's absoluteBoundingBox as the origin
            root_bounds = root_node.get('absoluteBoundingBox', {'x': 0, 'y': 0, 'width': 0, 'height': 0}).copy()

            def normalize_node_recursive(node_data: Dict[str, Any]) -> None:
                # Normalize absoluteBoundingBox coordinates
                if 'absoluteBoundingBox' in node_data:
                    bbox = node_data['absoluteBoundingBox']
                    if isinstance(bbox, dict) and 'x' in bbox and 'y' in bbox:
                        bbox['x'] = bbox['x'] - root_bounds['x']
                        bbox['y'] = bbox['y'] - root_bounds['y']

                # Normalize absoluteRenderBounds coordinates
                if 'absoluteRenderBounds' in node_data:
                    render_bounds = node_data['absoluteRenderBounds']
                    # absoluteRenderBounds can be null, so check
                    if isinstance(render_bounds, dict) and 'x' in render_bounds and 'y' in render_bounds:
                        render_bounds['x'] = render_bounds['x'] - root_bounds['x']
                        render_bounds['y'] = render_bounds['y'] - root_bounds['y']
                
                # Recursively process child nodes
                if 'children' in node_data:
                    for child in node_data['children']:
                        normalize_node_recursive(child)

            def round_all_floats_inplace(value: Any, precision: int) -> Any:
                # Recursively round floats in dicts/lists in-place and return the value
                if isinstance(value, float):
                    return round(value, precision)
                if isinstance(value, list):
                    for i in range(len(value)):
                        value[i] = round_all_floats_inplace(value[i], precision)
                    return value
                if isinstance(value, dict):
                    for k in list(value.keys()):
                        value[k] = round_all_floats_inplace(value[k], precision)
                    return value
                return value

            # 1) Coordinate normalization
            normalize_node_recursive(root_node)

            # 2) Round all floats to the specified precision
            precision = process_config.get('precision', None)
            if precision is not None:
                round_all_floats_inplace(root_node, precision)

        def filter_redundant_nesting(root_node: Dict[str, Any]) -> None:
            """Filter redundant nesting: remove layers with no style information and merge multi-level styles."""
            
            def has_meaningful_styles(node_data: Dict[str, Any]) -> bool:
                """Check if a node has meaningful style information."""

                # Check for fills, strokes, effects, corner radius, corner smoothing, rectangle corner radii
                if (node_data.get('fills', []) or node_data.get('strokes', []) or node_data.get('effects', []) or node_data.get('cornerRadius', 0) > 0 or node_data.get('cornerSmoothing', 0) > 0 or any(radius > 0 for radius in node_data.get('rectangleCornerRadii', []))):
                    return True
                
                return False
            
            def merge_styles(parent: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
                """Merge style information from parent and child nodes."""
                merged = child.copy()
                
                # Merge fills - child's fills take precedence, parent's act as background
                merged['fills'] = child.get('fills', []) + parent.get('fills', [])
                
                # Merge strokes - child's strokes take precedence
                merged['strokes'] = child.get('strokes', []) + parent.get('strokes', [])
                
                # Merge effects - child's effects take precedence
                merged['effects'] = child.get('effects', []) + parent.get('effects', [])
                
                # Merge corner radius - take the maximum
                parent_corner_radius = parent.get('cornerRadius', 0)
                child_corner_radius = child.get('cornerRadius', 0)
                if parent_corner_radius > 0 or child_corner_radius > 0:
                    merged['cornerRadius'] = max(parent_corner_radius, child_corner_radius)
                
                # Merge corner smoothing - take the maximum
                parent_corner_smoothing = parent.get('cornerSmoothing', 0)
                child_corner_smoothing = child.get('cornerSmoothing', 0)
                if parent_corner_smoothing > 0 or child_corner_smoothing > 0:
                    merged['cornerSmoothing'] = max(parent_corner_smoothing, child_corner_smoothing)
                
                # Merge rectangle corner radii - take the maximum at each position
                parent_rectangle_corner_radii = parent.get('rectangleCornerRadii', [0, 0, 0, 0])
                child_rectangle_corner_radii = child.get('rectangleCornerRadii', [0, 0, 0, 0])
                merged_radii = [
                        max(parent_rectangle_corner_radii[i], child_rectangle_corner_radii[i])
                        for i in range(4)
                    ]
                if any(radius > 0 for radius in merged_radii):
                    merged['rectangleCornerRadii'] = merged_radii
                
                return merged
            
            def flatten_node_recursive(node_data: Dict[str, Any]) -> Dict[str, Any]:
                """Recursively flatten nodes and merge styles."""
                if not isinstance(node_data, dict):
                    return node_data
                
                # First, recursively process child nodes
                if 'children' in node_data:
                    node_data['children'] = [flatten_node_recursive(child) for child in node_data['children']]
                
                # Check if it's a mergeable container
                children = node_data.get('children', [])
                if (node_data.get('type') in self.container_types and len(children) == 1):
                    # If the node has no style information, return the child directly
                    if not has_meaningful_styles(node_data):
                        return children[0]
                    child = children[0]
                    # If it has styles and the single child is also a container, merge styles
                    if child.get('type') in self.container_types:
                        return merge_styles(node_data, child)
                
                return node_data
            
            flatten_node_recursive(root_node)

        def recognize_layout(root_node: Dict[str, Any]) -> None:
            """Recognize layer layout: re-identify layout information for container nodes lacking it by calculating child element positions."""
            
            def infer_flex_direction(children: List[Dict[str, Any]]) -> str:
                """Infer flex-direction based on child element positions."""
                if len(children) < 2:
                    return 'column'
                
                # Calculate the distribution of child element positions
                positions = []
                for child in children:
                    if isinstance(child, dict) and 'absoluteBoundingBox' in child:
                        bbox = child['absoluteBoundingBox']
                        positions.append((bbox.get('x', 0), bbox.get('y', 0)))
                
                if len(positions) < 2:
                    return 'column'
                
                # Calculate variance in horizontal and vertical directions
                x_variance = max(pos[0] for pos in positions) - min(pos[0] for pos in positions)
                y_variance = max(pos[1] for pos in positions) - min(pos[1] for pos in positions)
                
                return 'row' if x_variance > y_variance else 'column'
            
            def calculate_spacing(children: List[Dict[str, Any]], direction: str) -> float:
                """Calculate spacing between child elements."""
                if len(children) < 2:
                    return 0
                
                spacings = []
                sorted_children = sorted(children, key=lambda x: x.get('absoluteBoundingBox', {}).get('x' if direction == 'row' else 'y', 0))
                
                for i in range(len(sorted_children) - 1):
                    current = sorted_children[i]
                    next_child = sorted_children[i + 1]
                    
                    current_bbox = current.get('absoluteBoundingBox', {})
                    next_bbox = next_child.get('absoluteBoundingBox', {})
                    
                    if direction == 'row':
                        current_end = current_bbox.get('x', 0) + current_bbox.get('width', 0)
                        next_start = next_bbox.get('x', 0)
                    else:
                        current_end = current_bbox.get('y', 0) + current_bbox.get('height', 0)
                        next_start = next_bbox.get('y', 0)
                    
                    spacing = max(0, next_start - current_end)
                    spacings.append(spacing)
                
                return sum(spacings) / len(spacings) if spacings else 0
            
            def recognize_layout_recursive(node_data: Dict[str, Any]) -> None:
                """Recursively recognize layout."""
                if not isinstance(node_data, dict):
                    return
                
                # Recursively process child nodes
                children = node_data.get('children', [])
                for child in children:
                    recognize_layout_recursive(child)
                
                # Only process container nodes
                if node_data.get('type') not in self.container_types or not children:
                    return
                
                # If layoutMode already exists, convert to standard format
                if 'layoutMode' in node_data:
                    layout_mode = node_data['layoutMode']
                    if layout_mode == 'HORIZONTAL':
                        node_data['flex_direction'] = 'row'
                    elif layout_mode == 'VERTICAL':
                        node_data['flex_direction'] = 'column'
                    
                    # Convert other Auto Layout properties
                    if 'itemSpacing' in node_data:
                        node_data['gap'] = node_data['itemSpacing']
                    if 'primaryAxisAlignItems' in node_data:
                        align_map = {
                            'MIN': 'flex-start',
                            'CENTER': 'center', 
                            'MAX': 'flex-end',
                            'SPACE_BETWEEN': 'space-between'
                        }
                        node_data['justify_content'] = align_map.get(node_data['primaryAxisAlignItems'], 'flex-start')
                    if 'counterAxisAlignItems' in node_data:
                        align_map = {
                            'MIN': 'flex-start',
                            'CENTER': 'center',
                            'MAX': 'flex-end'
                        }
                        node_data['align_items'] = align_map.get(node_data['counterAxisAlignItems'], 'flex-start')
                else:
                    # Infer layout mode
                    flex_direction = infer_flex_direction(children)
                    spacing = calculate_spacing(children, flex_direction)
                    
                    node_data['flex_direction'] = flex_direction
                    node_data['gap'] = spacing
                    node_data['display'] = 'flex'
            
            recognize_layout_recursive(root_node)

        if not is_page:
            # For component, componentSet, and style nodes, only perform these compression steps
            merge_asset_nodes(root_node)
            filter_node_attributes(root_node)
            normalize_num_value(root_node)
            return
        
        # Node Optimization

        ## Remove useless nodes
        remove_useless_nodes(root_node)

        ## Merge asset nodes
        merge_asset_nodes(root_node)
        
        ## Filter node attributes
        filter_node_attributes(root_node)
        
        ## Normalize coordinates
        normalize_num_value(root_node)

        # Layer Optimization

        ## Filter redundant nesting
        filter_redundant_nesting(root_node)
        
        ## Recognize layer layout
        # TODO: To be optimized
        if process_config['add_layout_inference']:
            recognize_layout(root_node)
    
    def _process_resources(self, file_key: str, output_dir: str) -> None:
        """Resource processing: handle various types of resources."""
        # Directory setup
        asset_dir = os.path.join(output_dir, 'assets')
        render_dir = os.path.join(output_dir, 'render')
        os.makedirs(asset_dir, exist_ok=True)
        os.makedirs(render_dir, exist_ok=True)
        
        def build_file_mappings(processed_dir_path: str) -> Dict[str, str]:
            """
            Builds a mapping from key/id to a relative path from the page directory.
            
            Args:
                processed_dir_path (str): The page directory path.
                
            Returns:
                Dict[str, str]: 
                    - key_id_to_relative_page_path: Mapping from key/id to a relative path.
            """
            # Categorize by type
            key_id_to_relative_page_path = {
                'assets/components_json': {},
                'assets/component_sets_json': {},
                'assets/styles_json': {},
                'assets/svg_assets': {},
                'assets/image_refs': {},
                'render/components_img': {},
                'render/nodes_img': {}
            }
            
            # Define subdirectories to process
            asset_subdirs = [
                'assets/components_json',
                'assets/component_sets_json', 
                'assets/styles_json',
                'assets/svg_assets',
                'assets/image_refs',
                'render/components_img',
                'render/nodes_img'
            ]
            
            for subdir in asset_subdirs:
                full_subdir_path = os.path.join(processed_dir_path, subdir)
                
                if not os.path.exists(full_subdir_path):
                    continue
                    
                # Iterate over all files in the subdirectory
                for filename in os.listdir(full_subdir_path):
                    file_path = os.path.join(full_subdir_path, filename)
                    relative_path = os.path.join(subdir, filename)
                    
                    # Remove file extension
                    filename = filename.split('.')[0]
                    # Handle key/id mapping based on subdirectory type
                    if subdir in ['assets/components_json', 'assets/component_sets_json', 'assets/styles_json', 'assets/image_refs', 'render/components_img']:
                        # These directories use the filename as the key
                        key = filename
                        key_id_to_relative_page_path[subdir][key] = relative_path
                        
                    elif subdir in ['assets/svg_assets', 'render/nodes_img']:
                        # These directories use a processed ID as the filename, which needs decoding
                        # decoded_id = unquote(filename)
                        decoded_id = filename.replace('_', ':').replace('-', ';')
                        key_id_to_relative_page_path[subdir][decoded_id] = relative_path
            
            return key_id_to_relative_page_path

        def collect_resources(root_node: Dict[str, Any]) -> Dict[str, Any]:
            """Resource Collection: Collect various resource IDs and keys."""
            
            # Collect imageRefs
            image_refs = find_imageref_in_json(root_node)
            
            # Collect SVG asset nodes and container nodes to be rendered, and get their names
            svg_assets = {}
            render_nodes = {}
            
            def collect_assets_and_render_recursive(node: Dict[str, Any]) -> None:
                node_type = node.get('type')
                node_id = node.get('id', '')
                node_name = node.get('name', f'{node_type}_{node_id}')
                
                if node_type == 'SVG_ASSET':
                    svg_assets[node_id] = node_name
                elif node_type in ['FRAME', 'GROUP']:
                    render_nodes[node_id] = node_name
                # Recursively process child nodes
                for child in node.get('children', []):
                    collect_assets_and_render_recursive(child)
            
            collect_assets_and_render_recursive(root_node)
            
            return {
                'image_refs': image_refs,
                'svg_assets': svg_assets,
                'render_nodes': render_nodes
            }

        def process_duplicate_resource(output_dir: str, resource_info: Dict[str, Any], resource_files: Dict[str, str], resource_type: str) -> Dict[str, str]:
            """
            Detects duplicate resources and returns a deduplicated mapping.
            
            Args:
                resource_info: Resource information (for resource names).
                resource_files: Mapping from resource ID to file path.
                resource_type: The type of resource (for logging).
                
            Returns:
                Dict[str, str]: A deduplicated mapping from resource ID to file path.
            """
            # Detect duplicate resources
            deduplicated_resources = {}
            
            # Group by name
            name_groups = {}
            full_path_map = {}
            for resource_id, file_path in resource_files.items():
                if file_path is None:
                    deduplicated_resources[resource_id] = None
                    continue
                full_path = os.path.join(output_dir, file_path)
                assert os.path.exists(full_path), f"Resource file does not exist: {full_path}"
                full_path_map[file_path] = full_path
                resource_name = resource_info[resource_id]
                if resource_name not in name_groups:
                    name_groups[resource_name] = []
                name_groups[resource_name].append((resource_id, file_path))
            
            for name, assets in name_groups.items():
                resource_id, file_path = assets[0]
                if len(assets) == 1:
                    # Single resource, keep it directly
                    deduplicated_resources[resource_id] = file_path
                    continue
                
                # Multiple resources with the same name, compare them pairwise
                self.logger.debug(f"Found {len(assets)} same-named {resource_type}: {name}")
                
                # List of deduplicated resources
                deduplicated_groups = {file_path:[resource_id]}
                deduplicated_resources[resource_id] = file_path # The first one must be added
                
                # From the second resource, compare with previous groups; if new, add a new group
                for i in range(1, len(assets)):
                    resource_id, file_path = assets[i]
                    resource_name = resource_info[resource_id]
                    for deduplicated_file, deduplicated_ids in deduplicated_groups.items():
                        if compare_images(full_path_map[file_path], full_path_map[deduplicated_file]):
                            # Delete duplicate resource file
                            if process_config['remove_dup_file']:
                                os.remove(full_path_map[file_path])
                            deduplicated_ids.append(resource_id)
                            deduplicated_resources[resource_id] = deduplicated_file
                            break
                    else:
                        deduplicated_groups[file_path] = [resource_id]
                        deduplicated_resources[resource_id] = file_path
                
                # Update the deduplication mapping
                for file_path, resource_ids in deduplicated_groups.items():
                    if len(resource_ids) > 1:
                        self.duplicate_resource_mapping[file_path] = resource_ids
                    
            return deduplicated_resources
        
        def update_resources(root_node: Dict[str, Any]) -> None:
            """
            Adds resource information (path backfilling, component/componentSet/style node info).
            
            Processing includes:
            1. Add `asset_path` to SVG_ASSET nodes, pointing to the downloaded SVG file.
            2. Modify `imageRef` values in fills to the actual saved image path.
            3. Add `render_path` to container nodes, pointing to the downloaded rendered image (optional).
            4. Add information for component, componentSet, and style nodes.
            
            Args:
                node: The processed node data.
            """
            
            def update_node_recursive(node: Dict[str, Any]) -> None:
                node_type = node.get('type')
                node_id = node.get('id', '')
                
                # Process imageRef in fills
                if 'fills' in node:
                    fills = node['fills']
                    if isinstance(fills, list):
                        for fill in fills:
                            if isinstance(fill, dict) and fill.get('type') == 'IMAGE':
                                image_ref = fill.get('imageRef')
                                if image_ref and image_ref in self.resource_downloaded.get('assets/image_refs', {}):
                                    # Modify imageRef value to the saved image path
                                    fill['imageRef'] = self.resource_downloaded['assets/image_refs'][image_ref]
                
                # Add asset_path for SVG_ASSET
                if node_type == 'SVG_ASSET' and node_id in self.resource_downloaded.get('assets/svg_assets', {}):                
                    if process_config['change_svg_node_to_rectangle']:
                        node['type'] = 'RECTANGLE'
                        node['blendMode'] = 'PASS_THROUGH'
                        """
                        scaleMode: 
                        - FILL: Fills the container, cropping the image if necessary.
                        - FIT: Fits the image within the container, preserving aspect ratio and possibly leaving empty space.
                        - TILE: Repeats the image.
                        - STRETCH: Stretches the image to fill the container.
                        """
                        node['fills'] = [{
                            'blendMode': 'NORMAL',
                            'type': 'IMAGE',
                            'scaleMode': 'FIT',
                            'imageRef': self.resource_downloaded['assets/svg_assets'][node_id]
                        }]

                        # Restore default properties
                        node['strokeWeight'] = 1.0
                        node['strokeAlign'] = 'INSIDE'
                        if not process_config['remove_empty_attributes']:
                            node['strokes'] = []
                    else:
                        node['asset_path'] = self.resource_downloaded['assets/svg_assets'][node_id]
                
                # Add render_path for container nodes (optional)
                if process_config['add_render_path'] and node_type in ['FRAME', 'GROUP'] and node_id in self.resource_downloaded.get('render/nodes_img', {}):
                    node['render_path'] = self.resource_downloaded['render/nodes_img'][node_id]
                
                # Recursively process child nodes
                if 'children' in node:
                    for child in node['children']:
                        update_node_recursive(child)
            
            # Path backfilling
            update_node_recursive(root_node)
        
        # Get downloaded resource information
        self.resource_downloaded = build_file_mappings(output_dir)

        # Process components (json+png)
        if components_info := self.processed_data.get('components'):
            components_json_dir = os.path.join(asset_dir, 'components_json')
            components_img_dir = os.path.join(render_dir, 'components_img')
            os.makedirs(components_json_dir, exist_ok=True)
            os.makedirs(components_img_dir, exist_ok=True)
            
            for comp_id, comp_info in components_info.items():
                comp_key = comp_info['key']
                # Download JSON
                json_path = os.path.join(components_json_dir, f"{comp_key}.json")
                if comp_key not in self.resource_downloaded['assets/components_json']:
                    comp_data = self.figma_session.get_node_info(file_key, comp_id)
                    if comp_data:
                        save_json(comp_data, json_path)
                        self.resource_downloaded['assets/components_json'][comp_key] = f"assets/components_json/{comp_key}.json"
                    else:
                        raise Exception(f"Failed to get component data for {comp_id}")
                else:
                    comp_data = load_json(json_path)
                self.resource_doc[comp_id] = comp_data.get('document', {})
                self._compress_node(self.resource_doc[comp_id], is_page=False)
                
                # Download image
                if comp_key not in self.resource_downloaded['render/components_img']:
                    if comp_data.get('document', {}).get('visible', False): # The component's image can only be downloaded when it is visible
                        self.resource_downloaded['render/components_img'][comp_key] = None
                    else:
                        img_urls, failed_ids = self.figma_session.get_render_image_urls(file_key, [comp_id])
                        if comp_id in img_urls:
                            file_ext = self.figma_session.download_image_from_url(
                                img_urls[comp_id], components_img_dir, comp_key
                            )
                            self.resource_downloaded['render/components_img'][comp_key] = f"render/components_img/{comp_key}{file_ext}" if file_ext else None
                        else:
                            raise Exception(f"Failed to get component image for {comp_id}")
        # Process componentSets (json)
        if component_set_info := self.processed_data.get('componentSets'):
            component_sets_json_dir = os.path.join(asset_dir, 'component_sets_json')
            os.makedirs(component_sets_json_dir, exist_ok=True)

            for cs_id, cs_info in component_set_info.items():
                cs_key = cs_info['key']
                json_path = os.path.join(component_sets_json_dir, f"{cs_key}.json")
                if cs_key not in self.resource_downloaded['assets/component_sets_json']:
                    cs_data = self.figma_session.get_node_info(file_key, cs_id)
                    if cs_data:
                        save_json(cs_data, json_path)
                        self.resource_downloaded['assets/component_sets_json'][cs_key] = f"assets/component_sets_json/{cs_key}.json"
                    else:
                        raise Exception(f"Failed to get component set data for {cs_id}")
                else:
                    cs_data = load_json(json_path)
                self.resource_doc[cs_id] = cs_data.get('document', {})
                self._compress_node(self.resource_doc[cs_id], is_page=False)
        # Process styles (json)
        if style_info := self.processed_data.get('styles'):
            styles_json_dir = os.path.join(asset_dir, 'styles_json')
            os.makedirs(styles_json_dir, exist_ok=True)

            for style_id, style_info in style_info.items():
                style_key = style_info['key']
                json_path = os.path.join(styles_json_dir, f"{style_key}.json")
                if style_key not in self.resource_downloaded['assets/styles_json']:
                    style_data = self.figma_session.get_node_info(file_key, style_id)
                    if style_data:
                        save_json(style_data, json_path)
                        self.resource_downloaded['assets/styles_json'][style_key] = f"assets/styles_json/{style_key}.json"
                    else:
                        raise Exception(f"Failed to get style data for {style_id}")
                else:
                    style_data = load_json(json_path)
                self.resource_doc[style_id] = style_data.get('document', {})
                self._compress_node(self.resource_doc[style_id], is_page=False)
        
        # Collect resource information
        self.resource_info = collect_resources(self.page_node)

        error_messages = [] # Problems encountered during the download process. When a problem occurs, record it and try to download others. Raise an error only after traversing all items.

        # Process imageRef (png)
        if image_refs := self.resource_info['image_refs']:
            # Check if there are any undownloaded images, note that image_refs here is a list
            image_refs = [ref for ref in image_refs if ref not in self.resource_downloaded['assets/image_refs']]
            if image_refs:
                image_refs_dir = os.path.join(asset_dir, 'image_refs')
                os.makedirs(image_refs_dir, exist_ok=True)
                try:
                    image_urls = self.figma_session.get_image_urls(file_key, image_refs)
                    for image_ref in image_refs:
                        if image_ref in image_urls:
                            try:
                                file_ext = self.figma_session.download_image_from_url(
                                    image_urls[image_ref], image_refs_dir, image_ref
                                )
                                self.resource_downloaded['assets/image_refs'][image_ref] = f"assets/image_refs/{image_ref}{file_ext}" if file_ext else None
                            except Exception as e:
                                error_messages.append(f"{e}")
                        else:
                            error_messages.append(f"Failed to get imageRef for {image_ref}")
                except Exception as e:
                    error_messages.append(f"Failed to get imageRef: {e}")
        # Process svg assets (svg)
        if svg_assets := self.resource_info['svg_assets']:
            # Check for undownloaded assets
            svg_ids = list(svg_assets.keys())
            svg_ids = [svg_id for svg_id in svg_ids if svg_id not in self.resource_downloaded['assets/svg_assets']]
            if svg_ids:
                svg_dir = os.path.join(asset_dir, 'svg_assets')
                os.makedirs(svg_dir, exist_ok=True)
                try:
                    svg_urls, failed_ids = self.figma_session.get_svg_urls(file_key, svg_ids)
                    for svg_id in svg_ids:
                        if svg_id in svg_urls:
                            file_name = safe_filename(svg_id)
                            try:
                                file_ext = self.figma_session.download_svg_from_url(
                                    svg_urls[svg_id], svg_dir, file_name
                                )
                                self.resource_downloaded['assets/svg_assets'][svg_id] = f"assets/svg_assets/{file_name}.svg" if file_ext else None
                            except Exception as e:
                                error_messages.append(f"Failed to download svg: {e}")
                        else:
                            error_messages.append(f"Failed to get svg for {svg_id}")
                except Exception as e:
                    error_messages.append(f"Failed to get svg: {e}")
        # Process container nodes to be rendered (png)
        if render_nodes := self.resource_info['render_nodes']:
            # Check for undownloaded nodes
            render_ids = list(render_nodes.keys())
            render_ids = [render_id for render_id in render_ids if render_id not in self.resource_downloaded['render/nodes_img']]
            if render_ids:
                render_nodes_dir = os.path.join(render_dir, 'nodes_img')
                os.makedirs(render_nodes_dir, exist_ok=True)
                try:
                    render_urls, failed_ids = self.figma_session.get_render_image_urls(file_key, render_ids)
                    for render_id in render_ids:
                        if render_id in render_urls:
                            file_name = safe_filename(render_id)
                            try:
                                file_ext = self.figma_session.download_image_from_url(
                                    render_urls[render_id], render_nodes_dir, file_name
                                )
                                self.resource_downloaded['render/nodes_img'][render_id] = f"render/nodes_img/{file_name}{file_ext}" if file_ext else None
                            except Exception as e:
                                error_messages.append(f"Failed to download render image: {e}")
                        else:
                            error_messages.append(f"Failed to get render image for {render_id}")
                except Exception as e:
                    error_messages.append(f"Failed to get render image: {e}")

        # Check if there are any None cases
        downloaded_none_messages = {}
        for resource_type, resource_dict in self.resource_downloaded.items():
            for resource_id, resource_path in resource_dict.items():
                if resource_path is None:
                    if resource_type not in downloaded_none_messages:
                        downloaded_none_messages[resource_type] = []
                    downloaded_none_messages[resource_type].append(resource_id)
        if downloaded_none_messages:
            self.logger.warning(f"None cases occurred during download: {downloaded_none_messages}")
        
        if error_messages:
            raise Exception(f"Errors occurred during download: {error_messages}")
        
        # Resource deduplication (optional)
        if process_config['enable_resource_deduplication']:
            # Detect duplicate resources of various types (note that only those with id as filename need deduplication)
            self.resource_downloaded['assets/svg_assets'] = process_duplicate_resource(output_dir,self.resource_info['svg_assets'], self.resource_downloaded['assets/svg_assets'], 'svg_assets'
            )
            self.resource_downloaded['render/nodes_img'] = process_duplicate_resource(output_dir,self.resource_info['render_nodes'], self.resource_downloaded['render/nodes_img'], 'render_nodes'
            )

        # Supplement resource information
        update_resources(self.page_node)
        for node_data in self.resource_doc.values():
            update_resources(node_data)

    def _simplify_duplicate_nodes(self) -> None:
        """
        Simplify duplicate nodes.
        
        For nodes representing the same resource:
        - The first node is kept complete.
        - Duplicate nodes only retain the following attributes: name, type, id, absoluteBoundingBox, absoluteRenderBounds, asset_path/render_path
        
        Args:
            processed_data: The processed node data.
        """
        if not self.duplicate_resource_mapping:
            return
        
        # Create a mapping from resource ID to main resource ID
        # Format: {duplicate_resource_ID: main_resource_ID}
        resource_id_mapping = {}
        
        for file_path, resource_ids in self.duplicate_resource_mapping.items():
            # The first ID is the main ID, others are duplicate IDs
            assert len(resource_ids) > 1, f"Resource ID list length must be greater than 1: {resource_ids}"
            main_id = resource_ids[0]
            for duplicate_id in resource_ids[1:]:
                resource_id_mapping[duplicate_id] = main_id 
        
        if not resource_id_mapping:
            return
        
        # Recursively process nodes to simplify duplicate nodes
        def simplify_nodes_recursive(node: Dict[str, Any]) -> int:
            node_id = node.get('id', '')
            simplified_count = 0
            
            # Check if the current node is a duplicate
            if node_id in resource_id_mapping:
                main_id = resource_id_mapping[node_id]
                self.logger.debug(f"Simplifying duplicate node {node_id} -> {main_id}")
                simplified_count += 1
                
                # Create a new simplified node
                simplified_node = {}
                for attr in self.duplicate_attributes:
                    if attr in node:
                        simplified_node[attr] = node[attr]
                # Add simplification flag
                simplified_node['_duplicate_from'] = main_id
                
                # Replace the original node
                node.clear()
                node.update(simplified_node)
            
            # Recursively process child nodes
            if 'children' in node:
                for child in node['children']:
                    simplified_count += simplify_nodes_recursive(child)
            
            return simplified_count
        
        # Start simplification process
        found_simplified_nodes = simplify_nodes_recursive(self.page_node)
        self.logger.debug(f"Simplified {found_simplified_nodes} duplicate nodes in {self.page_node.get('id', '')}")
        for node_data in self.resource_doc.values():
            found_simplified_nodes = simplify_nodes_recursive(node_data)
            if found_simplified_nodes:
                self.logger.debug(f"Simplified {found_simplified_nodes} duplicate nodes in {node_data.get('id', '')}")

    def _generate_statistics(self) -> Dict[str, Any]:
        """Data statistics: generate a detailed processing report"""
        statics = get_node_statics(self.page_node)
        
        # Count the number of resources (before deduplication)
        resource_counts = {
            'components': len(self.page_node.get('components', [])),
            'component_sets': len(self.page_node.get('componentSets', [])),
            'styles': len(self.page_node.get('styles', [])),
            'image_refs': len(self.resource_info['image_refs']),
            'svg_assets': len(self.resource_info['svg_assets'])
        }
        
        # Count the number of downloaded resources (after deduplication)
        downloaded_counts = {
            'components_json': len(self.resource_downloaded['assets/components_json']),
            'components_img': len(self.resource_downloaded['render/components_img']),
            'component_sets_json': len(self.resource_downloaded['assets/component_sets_json']),
            'styles_json': len(self.resource_downloaded['assets/styles_json']),
            'image_refs': len(self.resource_downloaded['assets/image_refs']),
            'svg_assets': len(set(self.resource_downloaded['assets/svg_assets'].values())),
            'render_nodes': len(set(self.resource_downloaded['render/nodes_img'].values()))
        }
        statics.update({
            'resource_counts': resource_counts,
            'downloaded_counts': downloaded_counts,
            'total_resources': sum(resource_counts.values()),
            'total_downloaded': sum(downloaded_counts.values())
        })
        
        return statics

    def _merge_page_and_resource_doc(self) -> None:
        """
        Merge page and resource_doc
        """
        if components := self.processed_data.get('components', {}):
            for id, component in components.items():
                component['document'] = self.resource_doc[id]
        if component_sets := self.processed_data.get('componentSets', {}):
            for id, component_set in component_sets.items():
                component_set['document'] = self.resource_doc[id]
        if styles := self.processed_data.get('styles', {}):
            for id, style in styles.items():
                style['document'] = self.resource_doc[id]

    def process_whitelist(self, csv_path: str) -> None:
        """
        Process the whitelist.csv file and batch process Figma data
        
        Args:
            csv_path: The path to the whitelist.csv file
        """
        success_count = 0
        failed_count = 0
        failed_items = []
        
        # Read the CSV file and calculate the total number of rows
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            total_rows = len(rows)
        
        # Create a progress bar
        progress = create_progress()
        task_id = progress.add_task("Processing whitelist", total=total_rows, status="Preparing...")
        
        with progress:
            for i, row in enumerate(rows):
                file_key = row.get('file_key', '').strip()
                node_id = row.get('node_id', '').strip()

                # Read other annotation attributes
                annotation = {}
                annotation['platform'] = row.get('platform', '').strip()
                annotation['complexity'] = row.get('complexity', '').strip()
                annotation['quality_rating'] = row.get('quality_rating', '').strip()
                annotation['theme'] = row.get('theme', '').strip()
                annotation['language'] = row.get('language', '').strip()
                annotation['content'] = row.get('content', '').strip()
                annotation['description'] = row.get('description', '').strip()
                annotation['content_original'] = row.get('content_original', '').strip()
                
                if not file_key or not node_id:
                    failed_count += 1
                    failed_items.append(f"{file_key}_{node_id}")
                    progress.update(task_id, advance=1, status=f"Skipping invalid row {i+1}")
                    continue
                
                progress.update(task_id, status=f"Processing {file_key}_{node_id}...")
                
                # Call the main processing function
                if self.process_figma_metadata(file_key, node_id, annotation):
                    success_count += 1
                    progress.update(task_id, advance=1, status=f"{file_key}_{node_id} Success")
                else:
                    failed_count += 1
                    failed_items.append(f"{file_key}_{node_id}")
                    progress.update(task_id, advance=1, status=f"{file_key}_{node_id} Failed")
        
        # Print processing result statistics
        console.print(f"\n[bold green]whitelist.csv processing complete![/bold green]")
        console.print(f"[green]Successfully processed: {success_count}[/green]")
        console.print(f"[red]Failed to process: {failed_count}[/red]")
        
        if failed_items:
            console.print(f"[yellow]Failed items: {', '.join(failed_items)}[/yellow]")
            console.print(f"[dim]Failed data has been moved to the {self.error_dir} folder[/dim]")
        
        self.logger.info(f"Whitelist processing complete - Success: {success_count}, Failed: {failed_count}")

    def cleanup_processed(self, level: int, targets: List[str] = None) -> Dict[str, Any]:
        """
        Clean up the processing result directory by level.

        Level description:
        Level 1: Delete processed_metadata.json and report.json
        Level 2: On top of level 1, additionally delete the assets/ folder
        Level 3: Delete the entire {file_key}_{node_id} directory

        Supports passing specific target directories (absolute paths or subdirectory names relative to self.output_dir).
        If targets are not provided, it defaults to iterating through all subdirectories under self.output_dir.

        Args:
            level: Cleanup level, value from {1, 2, 3}
            targets: List of directories to clean up (optional)

        Returns:
            A dictionary containing cleanup statistics.
        """
        if level not in {1, 2, 3}:
            raise ValueError("level can only be 1, 2, or 3")

        processed_root = os.path.abspath(self.output_dir)
        if not os.path.isdir(processed_root):
            return {
                "processed_root": processed_root,
                "deleted_files": [],
                "deleted_dirs": [],
                "skipped": [],
                "total_targets": 0
            }

        # Parse the set of target directories
        candidate_dirs: List[str] = []
        if targets:
            for t in targets:
                abs_path = t if os.path.isabs(t) else os.path.join(processed_root, t)
                candidate_dirs.append(os.path.abspath(abs_path))
        else:
            # Default to cleaning all subdirectories under processed_root
            for name in os.listdir(processed_root):
                full_path = os.path.join(processed_root, name)
                if os.path.isdir(full_path):
                    candidate_dirs.append(os.path.abspath(full_path))

        deleted_files: List[str] = []
        deleted_dirs: List[str] = []
        skipped: List[str] = []

        progress = create_progress()
        task_id = progress.add_task(f"Hierarchical Cleanup (Level {level})", total=len(candidate_dirs), status="Preparing...")
        with progress:
            for d in candidate_dirs:
                if not os.path.isdir(d):
                    skipped.append(d)
                    progress.update(task_id, advance=1, status=f"Skipping (does not exist): {os.path.basename(d)}")
                    continue

                base_name = os.path.basename(d)
                try:
                    if level == 3:
                        shutil.rmtree(d, ignore_errors=True)
                        deleted_dirs.append(d)
                        progress.update(task_id, advance=1, status=f"Deleting directory: {base_name}")
                        continue

                    # File deletion common to Level 1 and Level 2
                    file_candidates = [
                        os.path.join(d, "processed_metadata.json"),
                        os.path.join(d, "report.json"),
                        # Compatible with user-described spelling
                        os.path.join(d, "precessed_metadat.json"),
                    ]
                    for fp in file_candidates:
                        if os.path.exists(fp):
                            try:
                                os.remove(fp)
                                deleted_files.append(fp)
                            except Exception as e:
                                self.logger.warning(f"Failed to delete file {fp}: {e}")

                    # Level 2 additionally deletes the assets directory
                    if level >= 2:
                        assets_dir = os.path.join(d, "assets")
                        if os.path.isdir(assets_dir):
                            shutil.rmtree(assets_dir, ignore_errors=True)
                            deleted_dirs.append(assets_dir)

                    progress.update(task_id, advance=1, status=f"Completed: {base_name}")
                except Exception as e:
                    self.logger.error(f"Cleanup failed {base_name}: {e}")
                    skipped.append(d)
                    progress.update(task_id, advance=1, status=f"Failed: {base_name}")

        result = {
            "processed_root": processed_root,
            "deleted_files": deleted_files,
            "deleted_dirs": deleted_dirs,
            "skipped": skipped,
            "total_targets": len(candidate_dirs)
        }
        # Brief log
        self.logger.info(
            f"Cleanup complete Level {level} | Targets:{len(candidate_dirs)} | Files deleted:{len(deleted_files)} | Dirs deleted:{len(deleted_dirs)} | Skipped:{len(skipped)}"
        )
        return result


if __name__ == "__main__":
    from ...configs.paths import enter_project_root
    enter_project_root()
    # Example usage and simple CLI
    processor = FigmaMetaDataProcess(output_dir="./output/processed_full")

    # processor.process_figma_metadata("He2CR0KvsufdoICJx3MVx1", "40:1255")
    # processor.process_figma_metadata("32Vr8yRCGiZLhAA6JPbfx1", "267:18")

    args = sys.argv[1:]
    if args and args[0] == "cleanup":
        # Usage: python figma_metadata_process.py cleanup <level> [--all | <subdir_name_or_abs_path> ...]
        try:
            level = int(args[1]) if len(args) >= 2 else 1
        except Exception:
            level = 1
        targets: List[str] = None
        if "--all" in args:
            targets = None
        else:
            targets = args[2:] if len(args) >= 3 else None
        result = processor.cleanup_processed(level=level, targets=targets)
        console.print(f"[bold green]Cleanup complete (Level {level})[/bold green]")
        console.print(f"[green]Files deleted: {len(result['deleted_files'])}[/green]")
        console.print(f"[green]Directories deleted: {len(result['deleted_dirs'])}[/green]")
        if result["skipped"]:
            console.print(f"[yellow]Skipped: {len(result['skipped'])}[/yellow]")
        # End
        sys.exit(0)

    elif args and args[0] == "whitelist":
        # Usage: python figma_metadata_process.py whitelist [csv_path]
        csv_path = args[1] if len(args) >= 2 else "./data/whitelist_testdata_final.csv"
        processor.process_whitelist(csv_path)
        sys.exit(0)
    elif args and args[0] == "test":
        # Usage: python figma_metadata_process.py test <file_key> <node_id>
        # python figma_metadata_process.py test K4sCSgcy7ZEewLNWhRg2uh 40:87
        file_key = args[1]
        node_id = args[2]
        processor.process_figma_metadata(file_key, node_id)
        sys.exit(0)
    else:
        # Default: Batch process whitelist.csv (maintaining original default behavior)
        processor.process_whitelist("./data/whitelist_full.csv")
