"""
HTML to screenshot conversion utilities.

Provides tools for capturing screenshots of HTML files using Playwright,
with support for proper image and font loading.
"""

import io
import os
import time
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Union

from PIL import Image
from playwright.sync_api import sync_playwright, Page

from .console_logger import logger


def _all_images_loaded(page: Page, timeout: float = 3.0) -> bool:
    """
    Check if all images on the page have finished loading.
    
    Args:
        page: Playwright Page object
        timeout: Maximum wait time in seconds
    
    Returns:
        True if all images loaded, False if timeout reached
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            all_loaded = page.evaluate('''
                () => {
                    const images = document.querySelectorAll('img');
                    if (images.length === 0) return true;
                    
                    return Array.from(images).every(img => {
                        if (!img.complete) return false;
                        if (img.naturalWidth === 0 && img.src) return false;
                        return true;
                    });
                }
            ''')
            
            if all_loaded:
                return True
            
            time.sleep(0.1)
            
        except Exception as e:
            logger.debug(f"Error checking image load status: {e}")
            time.sleep(0.1)
    
    return False


def _wait_for_fonts_loaded(page: Page, timeout: float = 2.0) -> bool:
    """
    Wait for web fonts to finish loading.
    
    Args:
        page: Playwright Page object
        timeout: Maximum wait time in seconds
    
    Returns:
        True if fonts loaded, False if timeout reached
    """
    try:
        page.wait_for_function(
            'document.fonts.ready',
            timeout=timeout * 1000
        )
        return True
    except Exception:
        return False


def html2shot(
    html_file_path: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    headless: bool = True,
    wait_time: float = 0.5,
    viewport: Optional[dict] = None,
    use_viewport: bool = False,
    device_scale_factor: float = 1.0,
    downscale_to_css: bool = False,
) -> Image.Image:
    """
    Convert an HTML file to a screenshot image.
    
    Args:
        html_file_path: Path to the HTML file
        output_file: Optional output PNG file path
        headless: Run browser in headless mode
        wait_time: Extra wait time after page load (seconds)
        viewport: Custom viewport size dict (e.g., {'width': 1920, 'height': 1080})
        use_viewport: Whether to constrain to viewport (False = full page)
        device_scale_factor: Device pixel ratio (for high-DPI screenshots)
        downscale_to_css: Scale output to CSS pixels (useful with high DPR)
    
    Returns:
        PIL Image object of the screenshot
    
    Raises:
        FileNotFoundError: If HTML file doesn't exist
    """
    html_file_path = Path(html_file_path)
    
    if not html_file_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_file_path}")
    
    # Build file URL
    abs_path = html_file_path.resolve()
    if os.name == 'nt':  # Windows
        file_url = 'file:///' + str(abs_path).replace('\\', '/')
    else:  # Unix/Linux/Mac
        file_url = 'file://' + str(abs_path)
    
    with sync_playwright() as p:
        launch_args = ['--disable-blink-features=AutomationControlled']
        
        browser = p.chromium.launch(
            headless=headless,
            args=launch_args
        )
        
        context = None
        page = None
        
        try:
            # Create context with options
            context_options = {"device_scale_factor": float(device_scale_factor)}
            if use_viewport:
                context_options['viewport'] = viewport or {'width': 1920, 'height': 1080}
            
            context = browser.new_context(**context_options)
            page = context.new_page()
            
            # Load HTML file
            page.goto(file_url, wait_until='networkidle', timeout=60000)
            
            # Wait for page load states
            page.wait_for_load_state('domcontentloaded')
            page.wait_for_load_state('networkidle')
            
            # Wait for fonts and images
            _wait_for_fonts_loaded(page, timeout=2.0)
            _all_images_loaded(page, timeout=5.0)
            
            # Extra wait for rendering
            if wait_time > 0:
                time.sleep(wait_time)
            
            # Take screenshot
            screenshot_bytes = page.screenshot(
                full_page=True,
                animations="disabled",
                timeout=60000
            )
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(screenshot_bytes))
            
            # Optionally downscale to CSS pixels
            if downscale_to_css and device_scale_factor > 1:
                target_w = max(1, round(image.width / device_scale_factor))
                target_h = max(1, round(image.height / device_scale_factor))
                image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
            
            # Save to file if requested
            if output_file:
                output_file = Path(output_file)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                if not str(output_file).lower().endswith('.png'):
                    output_file = output_file.with_suffix('.png')
                
                image.save(str(output_file), format='PNG', optimize=True)
            
            return image
            
        except Exception as e:
            logger.error(f"Screenshot error: {e}")
            traceback.print_exc()
            
            # Return blank image as fallback
            image = Image.new('RGB', (1920, 1080), color='white')
            if output_file:
                output_file = Path(output_file)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                image.save(str(output_file), format='PNG')
            return image
            
        finally:
            if page:
                page.close()
            if context:
                context.close()
            browser.close()


def batch_html_to_screenshots(
    html_files: List[Union[str, Path]],
    output_dir: Union[str, Path] = "screenshots",
    headless: bool = True,
    wait_time: float = 0.5
) -> Dict[str, Optional[Image.Image]]:
    """
    Batch process multiple HTML files to screenshots.
    
    Uses a shared browser instance for efficiency.
    
    Args:
        html_files: List of HTML file paths
        output_dir: Output directory for PNG files
        headless: Run browser in headless mode
        wait_time: Wait time for each page (seconds)
    
    Returns:
        Dictionary mapping input paths to Image objects (None if failed)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=['--disable-blink-features=AutomationControlled']
        )
        
        try:
            for html_file in html_files:
                html_file = Path(html_file)
                context = None
                page = None
                
                try:
                    output_file = output_dir / f"{html_file.stem}.png"
                    
                    context = browser.new_context()
                    page = context.new_page()
                    
                    # Build file URL
                    abs_path = html_file.resolve()
                    if os.name == 'nt':
                        file_url = 'file:///' + str(abs_path).replace('\\', '/')
                    else:
                        file_url = 'file://' + str(abs_path)
                    
                    page.goto(file_url, wait_until='networkidle', timeout=60000)
                    page.wait_for_load_state('networkidle')
                    _wait_for_fonts_loaded(page, timeout=2.0)
                    _all_images_loaded(page, timeout=5.0)
                    
                    if wait_time > 0:
                        time.sleep(wait_time)
                    
                    screenshot_bytes = page.screenshot(
                        full_page=True,
                        animations="disabled",
                        timeout=60000
                    )
                    
                    image = Image.open(io.BytesIO(screenshot_bytes))
                    image.save(str(output_file), format='PNG', optimize=True)
                    
                    results[str(html_file)] = image
                    
                except Exception as e:
                    logger.error(f"Failed to screenshot {html_file}: {e}")
                    results[str(html_file)] = None
                    
                finally:
                    if page:
                        page.close()
                    if context:
                        context.close()
                    
        finally:
            browser.close()
    
    return results


def quick_shot(
    html_file: Union[str, Path],
    output_name: Optional[str] = None
) -> tuple:
    """
    Quick screenshot with default settings.
    
    Args:
        html_file: HTML file path
        output_name: Output filename (without path or extension)
    
    Returns:
        Tuple of (PIL Image, output file path)
    """
    html_file = Path(html_file)
    
    if output_name:
        output_file = html_file.parent / f"{output_name}.png"
    else:
        output_file = html_file.parent / f"{html_file.stem}_screenshot.png"
    
    image = html2shot(html_file, output_file)
    return image, output_file
