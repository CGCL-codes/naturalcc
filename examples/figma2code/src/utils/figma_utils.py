"""
Configure Figma-related parameters and functions.
"""
import os
import random
import time
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import hashlib
from typing import Dict, Any
import cv2
import numpy as np

# ==================== Configuration Parameters ====================
try:
    from ..configs.settings import get_settings
    settings = get_settings()
    FIGMA_TOKEN = settings.figma_api_key
except Exception:
    FIGMA_TOKEN = os.getenv("FIGMA_API_KEY", "xxxxxxxx") # Replace with your Token

# Request related
MAX_RETRIES = 3  # Number of failed retries
MIN_DELAY = 0.5  # Minimum request interval (seconds)
MAX_DELAY = 3  # Maximum random delay (seconds) to avoid regular requests
RATE_LIMIT_SLEEP = 30  # Waiting time after triggering rate limit (seconds)

BATCH_SIZE = 30 # Batch request quantity

# ==================== FigmaSession Class ====================
class FigmaSession:
    """Figma API session management class, encapsulating all session-related functions"""
    
    def __init__(self, token: str = None):
        """
        Initialize FigmaSession
        
        Args:
            token (str): Figma API Token. If not provided, the globally configured FIGMA_TOKEN will be used.
        """
        self.token = token or FIGMA_TOKEN
        if not self.token:
            raise ValueError("Figma Token cannot be empty")
        
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create a request session with automatic retries"""
        session = requests.Session()
        # Retry strategy: retry for 5xx errors, 429 (rate limit), 408 (timeout)
        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=1, # Retry interval: {backoff_factor} * (2 **({total_retries} - 1))
            status_forcelist=[429, 500, 502, 503, 504, 408],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers = {"X-Figma-Token": self.token}
        return session

    def _get_response(self, url: str, params: dict = None) -> requests.Response: 
        """
        Get response data, and handle rate limiting, retries, status code checks, etc.

        Args:
            url: API request URL
            params: Request parameters

        Returns:
            requests.Response: Response object
        """
        delay = random.uniform(MIN_DELAY, MAX_DELAY)
        time.sleep(delay)
        response = self.session.get(url, params=params, timeout=20)
        response.raise_for_status()

        # Handle rate limiting
        if response.status_code == 429:
            time.sleep(RATE_LIMIT_SLEEP)

            # Retry once
            response = self.session.get(url, params=params, timeout=20)
            response.raise_for_status()
            
            # If it's still 429, raise an exception
            if response.status_code == 429:
                raise Exception(f"Request failed after triggering rate limit (429), skipping this request")
        # Check response status
        if response.status_code != 200:
            raise Exception(f"HTTP status code {response.status_code} - {response.reason}")
        return response

    def _get_image_from_url(self, image_url: str) -> tuple[bytes, str]:
        """
        Get image bytes from URL and automatically infer file extension.
        
        Returns:
            tuple: (image bytes, file extension)
        """
        response = self._get_response(image_url)
        content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        file_ext = infer_image_ext(content_type, response.content)
        return response.content, file_ext

    def download_image_from_url(self, image_url: str, save_dir: str, file_name: str, file_ext: str = ".png") -> str:
        """
        Download a single image from a URL.
        
        Args:
            image_url (str): Image download URL
            save_dir (str): Save directory
            file_name (str): Filename (without extension)
            file_ext (str): File extension
            
        Returns:
            str: Returns the actual extension
        """
        
        if not image_url: # Indicates that nothing was actually rendered (e.g., invisible)
            return None
            # raise Exception(f"{file_name} download failed, URL is empty")
        
        save_path = os.path.join(save_dir, f"{file_name}{file_ext}")
        
        try:
            # Check if the file already exists
            if os.path.exists(save_path):
                return file_ext
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Download image
            response, file_ext = self._get_image_from_url(image_url)
            save_path = os.path.join(save_dir, f"{file_name}{file_ext}")
            
            with open(save_path, 'wb') as f:
                f.write(response)
            
            return file_ext
            
        except Exception as e:
            raise Exception(f"{file_name} download failed, {e}")

    def download_svg_from_url(self, svg_url: str, save_dir: str, file_name: str) -> str:
        """
        Download an SVG file from a URL.
        
        Args:
            svg_url (str): SVG download URL
            save_dir (str): Save directory
            file_name (str): Filename (without extension)
            
        Returns:
            str: Returns the file extension (.svg)
        """
        
        if not svg_url: # Indicates that nothing was actually rendered (e.g., invisible)
            # raise Exception(f"{file_name} download failed, SVG URL is empty")
            return None
        
        file_ext = ".svg"
        save_path = os.path.join(save_dir, f"{file_name}{file_ext}")
        
        try:
            # Check if the file already exists
            if os.path.exists(save_path):
                return file_ext
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Download SVG
            response = self._get_response(svg_url)
            
            # SVG is a text file, needs to be saved in text mode
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            return file_ext
            
        except Exception as e:
            raise Exception(f"{file_name} download failed, {e}")

    def figma_api_request(self, url: str, params: dict = None) -> dict:
        """
        Unified Figma API request function to handle common logic.
        
        Args:
            url: API request URL
            params: Request parameters
            
        Returns:
            dict: Returns response JSON data
        """
        try:
            response = self._get_response(url, params)
            data = response.json() # Parse JSON
        except json.JSONDecodeError as e:
            raise Exception(f"JSON parsing error - {str(e)}")
        except Exception as e:
            raise Exception(str(e))
        
        # Parse JSON and return
        return data

    def get_raw_json(self, file_key: str) -> dict:
        """
        Get the raw.json of the file.
        """
        url = f"https://api.figma.com/v1/files/{file_key}"
        return self.figma_api_request(url)

    def get_image_urls(self, file_key: str, selected_imageRef: list = None) -> dict:
        """
        Get the download links for imageRefs.
        """
        url = f"https://api.figma.com/v1/files/{file_key}/images"
        data = self.figma_api_request(url)
        imageRef_url = data.get('meta', {}).get('images', {})
        if selected_imageRef:
            imageRef_url = {k: v for k, v in imageRef_url.items() if k in selected_imageRef}
        return imageRef_url

    def get_node_info(self, file_key: str, node_id: str) -> dict:
        """
        Get detailed information about a node.
        """
        url = f"https://api.figma.com/v1/files/{file_key}/nodes"
        params = {"ids": node_id}
        node_info = self.figma_api_request(url, params=params)
        return node_info.get("nodes", {}).get(node_id, {})

    def get_render_image_urls(self, file_key: str, node_ids: list, format: str = 'png', scale: int = 2) -> tuple[dict, set]:
        """
        Get the rendered image download links for a list of node IDs.
        
        Args:
            file_key (str): Figma file key
            node_ids (list): List of node IDs
            format (str): Image format, supports 'png', 'jpg', 'svg', 'pdf'
            scale (int): Scaling factor, only valid for png/jpg
            
        Returns:
            dict: Mapping from node ID to download URL
            set: List of node IDs for which download links were not obtained
        """
        if not node_ids:
            return {}, set()
        
        # Use the unified API request function
        url = f"https://api.figma.com/v1/images/{file_key}"

        download_urls = {}
        # Request in batches
        for i in range(0, len(node_ids), BATCH_SIZE):
            batch_ids = node_ids[i:i+BATCH_SIZE]
            params = {
                'ids': ','.join(batch_ids),
                'format': format
            }
            # Only png and jpg formats support the scale parameter
            if format in ['png', 'jpg']:
                params['scale'] = scale
                
            images_data = self.figma_api_request(url, params=params)
            
            if not images_data:
                continue

            download_urls.update(images_data.get('images', {}))
        
        failed_ids = set(node_ids) - set(download_urls.keys())
        
        return download_urls, failed_ids

    def get_svg_urls(self, file_key: str, node_ids: list) -> tuple[dict, set]:
        """
        Get the SVG download links for VECTOR nodes.
        
        Args:
            file_key (str): Figma file key
            node_ids (list): List of node IDs
            
        Returns:
            dict: Mapping from node ID to SVG download URL
            set: List of node IDs for which download links were not obtained
        """
        return self.get_render_image_urls(file_key, node_ids, format='svg', scale=1)

# ==================== Utility Functions ====================
def safe_filename(filename: str) -> str:
    """Generate a safe filename by replacing special characters."""
    # Replace unsafe characters
    # filename = quote(filename, safe='/-._')
    filename = filename.replace(':', '_').replace(';', '-')

    
    # Limit length and remove leading/trailing spaces
    filename = filename.strip()[:200]
    
    return filename

def find_imageref_in_json(json_data: dict) -> list:
    """Recursively find all imageRefs in JSON."""
    imageref_set = set()
    
    def traverse_node(node):
        if isinstance(node, dict):
            # Check for imageRef in fills
            fills = node.get('fills', [])
            for fill in fills:
                if isinstance(fill, dict) and fill.get('type') == 'IMAGE':
                    image_ref = fill.get('imageRef')
                    if image_ref:
                        imageref_set.add(image_ref)
            
            # Recursively process child nodes
            children = node.get('children', [])
            for child in children:
                traverse_node(child)
        elif isinstance(node, list):
            for item in node:
                traverse_node(item)
    
    traverse_node(json_data)
    return list(imageref_set)


def find_component_instances(json_data: dict) -> set:
    """
    Recursively find all component instances in the node tree and collect their componentIds.
    """
    component_ids_to_download = set()
    
    def traverse_node(node):
        if isinstance(node, dict):
            # Check if the current node is a component instance
            if node.get("type") == "INSTANCE" and "componentId" in node:
                component_id_raw = node["componentId"]
                # Handle local reference format like 'I121:6600' by removing the 'I' prefix
                if component_id_raw.startswith("I"):
                    original_component_id = component_id_raw[1:]
                else:
                    original_component_id = component_id_raw
                
                # Ensure the componentId format is correct, usually "page_id:node_id"
                # Simple validation to prevent empty strings or unexpected IDs
                if ':' in original_component_id:
                    component_ids_to_download.add(original_component_id)
                else:
                    print(f"Warning: Found an abnormal componentId format: {component_id_raw}, skipping.")
            # Recursively process child nodes
            if "children" in node:
                for child in node["children"]:
                    traverse_node(child)
    
    traverse_node(json_data)
    return list(component_ids_to_download)


def compare_images(image_path1: str, image_path2: str, similarity_threshold: float = 0.95) -> bool:
    """
    Compare if two images are the same or highly similar, supports SVG format.
    
    Args:
        image_path1: First image path
        image_path2: Second image path
        similarity_threshold: Similarity threshold, default 0.95
        
    Returns:
        bool: Whether they are the same or highly similar
    """
    try:
        # First, compare file size and hash (quick check)
        if os.path.getsize(image_path1) != os.path.getsize(image_path2):
            return False
        
        # Calculate file hash
        def get_file_hash(file_path: str) -> str:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        
        hash1 = get_file_hash(image_path1)
        hash2 = get_file_hash(image_path2)
        
        if hash1 == hash2:
            return True
        
        # If hashes are different, perform image similarity comparison
        def load_image(file_path: str):
            if file_path.lower().endswith('.svg'):
                # Convert SVG to PNG for comparison
                import cairosvg
                png_data = cairosvg.svg2png(url=file_path)
                image = cv2.imdecode(np.frombuffer(png_data, np.uint8), cv2.IMREAD_UNCHANGED)
            else:
                image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            return image
        
        img1 = load_image(image_path1)
        img2 = load_image(image_path2)
        
        if img1 is None or img2 is None:
            raise Exception("Unable to load image")
        
        if img1.shape != img2.shape:
            return False

        # Handle small images separately
        if img1.shape[0] < 7 or img1.shape[1] < 7:
            # Use mean squared error for small images
            mse = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
            return mse < (1 - similarity_threshold) * 255
        
        # Compute structural similarity
        from skimage.metrics import structural_similarity as ssim
        sim_index, _ = ssim(img1, img2, full=True, multichannel=True, channel_axis=-1)
        
        return sim_index >= similarity_threshold
        
    except Exception as e:
        raise Exception(f"Image comparison failed: {e}")

def get_node_statics(root_node: Dict[str, Any]) -> Dict[str, Any]:
    counts = {}
        
    def count_recursive(node_data: dict[str, Any], depth: int = 0) -> int:
        node_type = node_data.get('type', 'UNKNOWN')
        counts[node_type] = counts.get(node_type, 0) + 1
        
        max_depth = depth
        for child in node_data.get('children', []):
            child_depth = count_recursive(child, depth + 1)
            max_depth = max(max_depth, child_depth)
        
        return max_depth
        
    max_depth = count_recursive(root_node)

    return {
        'type_counts': counts,
        'node_counts': sum(counts.values()),
        'max_depth': max_depth
    }


# ==================== Image Format Inference Related ====================
# Pillow is only used for fallback recognition; if not installed, it can be omitted.
try:
    from PIL import Image, UnidentifiedImageError
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

MIME_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg":  ".jpg",
    "image/png":  ".png",
    "image/gif":  ".gif",
    "image/webp": ".webp",
    "image/bmp":  ".bmp",
    "image/tiff": ".tiff",
    "image/x-icon": ".ico",
    "image/vnd.microsoft.icon": ".ico",
    "image/svg+xml": ".svg",
    "image/heic": ".heic",
    "image/heif": ".heif",
}

PIL_FORMAT_TO_EXT = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "GIF": ".gif",
    "WEBP": ".webp",
    "BMP": ".bmp",
    "TIFF": ".tiff",
    "ICO": ".ico",
    "PPM": ".ppm",
    "PGM": ".pgm",
    "PBM": ".pbm",
    "TGA": ".tga",
}

def infer_image_ext(content_type: str, content_bytes: bytes) -> str:
    """
    Infer extension from Content-Type; if unavailable, use Pillow to determine from bytes.
    If still unrecognizable, return '.png' as the default.
    """
    # 1) First, check Content-Type
    if content_type in MIME_TO_EXT:
        return MIME_TO_EXT[content_type]

    # 2) Use Pillow for fallback recognition (based on file header)
    if PIL_AVAILABLE:
        try:
            import io
            with Image.open(io.BytesIO(content_bytes)) as im:
                fmt = im.format  # e.g., 'PNG', 'JPEG'
                if fmt and fmt in PIL_FORMAT_TO_EXT:
                    return PIL_FORMAT_TO_EXT[fmt]
        except UnidentifiedImageError:
            pass
        except Exception:
            pass

    # 3) If still unrecognizable, provide a default extension
    return ".png"
