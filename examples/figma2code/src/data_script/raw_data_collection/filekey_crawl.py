"""
1. Get community links.
2. Get all template links under the community links.
3. Get all file keys under the template links.
4. Save the file keys to a file.

This script needs to be run on Windows with Chrome browser installed and a Figma account logged in.
"""


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
import time
import re
import os
import json
import hashlib
import random

# =======================================================
# Configure Figma account information
# =======================================================
try:
    from ...configs.settings import get_settings
    settings = get_settings()
    FIGMA_EMAIL = settings.figma_email
    FIGMA_PASSWORD = settings.figma_password
except Exception:
    FIGMA_EMAIL = os.getenv("FIGMA_EMAIL", "your_email@example.com")
    FIGMA_PASSWORD = os.getenv("FIGMA_PASSWORD", "your_password")

if FIGMA_EMAIL == "your_email@example.com" or FIGMA_PASSWORD == "your_password":
    print("⚠️ Please configure the correct Figma account information!")

DEFAULT_URLS = [
        "https://www.figma.com/community/website-templates?resource_type=files&editor_type=all&price=free",
        "https://www.figma.com/community/website-templates/blog?resource_type=files&editor_type=all&price=free",
        "https://www.figma.com/community/web-ads?resource_type=files&editor_type=all&price=free",
        "https://www.figma.com/community/design-templates?resource_type=files&editor_type=figma&price=free",
        "https://www.figma.com/community/mobile-apps?resource_type=files&editor_type=figma&price=free",
        "https://www.figma.com/community/portfolio-templates?resource_type=files&editor_type=figma&price=free",
        "https://www.figma.com/community/resume-templates?resource_type=files&editor_type=figma&price=free",
]


# =======================================================
# Rate Limit Manager Class
# =======================================================
class RateLimitManager:
    def __init__(self):
        self.request_times = []
        self.max_requests_per_minute = 30  # Max 30 requests per minute
        self.max_requests_per_hour = 500   # Max 500 requests per hour
        self.min_delay = 2                 # Min delay 2 seconds
        self.max_delay = 8                 # Max delay 8 seconds
        self.consecutive_failures = 0      # Consecutive failures
        self.last_request_time = 0
        
    def wait_if_needed(self):
        """Wait if needed based on rate limit policy."""
        current_time = time.time()
        
        # Clean up expired request records (from 1 hour ago)
        self.request_times = [t for t in self.request_times if current_time - t < 3600]
        
        # Check per-minute limit
        recent_requests = [t for t in self.request_times if current_time - t < 60]
        if len(recent_requests) >= self.max_requests_per_minute:
            wait_time = 60 - (current_time - min(recent_requests))
            print(f"⏰ Reached per-minute limit, waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        
        # Check per-hour limit
        if len(self.request_times) >= self.max_requests_per_hour:
            wait_time = 3600 - (current_time - min(self.request_times))
            print(f"⏰ Reached per-hour limit, waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        
        # Base delay (randomized)
        base_delay = random.uniform(self.min_delay, self.max_delay)
        
        # Increase delay for consecutive failures
        if self.consecutive_failures > 0:
            failure_delay = min(self.consecutive_failures * 2, 30)  # Max additional wait of 30 seconds
            base_delay += failure_delay
            print(f"⚠️ {self.consecutive_failures} consecutive failures, increasing delay to {base_delay:.1f} seconds")
        
        # Ensure interval since last request is at least the base delay
        time_since_last = current_time - self.last_request_time
        if time_since_last < base_delay:
            sleep_time = base_delay - time_since_last
            print(f"⏱️ Waiting {sleep_time:.1f} seconds to avoid rate limiting...")
            time.sleep(sleep_time)
        
        # Record request time
        self.request_times.append(time.time())
        self.last_request_time = time.time()
    
    def record_success(self):
        """Record successful request, reset failure count."""
        self.consecutive_failures = 0
    
    def record_failure(self):
        """Record failed request."""
        self.consecutive_failures += 1
        print(f"❌ Failure recorded, consecutive failures: {self.consecutive_failures}")
    
    def is_rate_limited(self, response_text="", status_code=None):
        """Check if rate limited."""
        rate_limit_indicators = [
            "rate limit",
            "too many requests",
            "Request frequency is too high",
            "Too many visits",
            "429",
            "blocked",
            "captcha"
        ]
        
        if status_code == 429:
            return True
            
        if response_text:
            response_lower = response_text.lower()
            return any(indicator in response_lower for indicator in rate_limit_indicators)
        
        return False

# =======================================================
# Cache Manager Class (Enhanced)
# =======================================================
class CacheManager:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        self.templates_cache_file = os.path.join(cache_dir, "templates_cache.json")
        self.filekeys_cache_file = os.path.join(cache_dir, "filekeys_cache.json")
        self.failed_urls_cache_file = os.path.join(cache_dir, "failed_urls_cache.json")
        self.ensure_cache_dir()
        self.load_caches()
    
    def ensure_cache_dir(self):
        """Ensure the cache directory exists."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"Created cache directory: {self.cache_dir}")
    
    def load_caches(self):
        """Load cache data."""
        # Load template link cache
        self.templates_cache = {}
        if os.path.exists(self.templates_cache_file):
            try:
                with open(self.templates_cache_file, 'r', encoding='utf-8') as f:
                    self.templates_cache = json.load(f)
                print(f"Loaded template link cache: {len(self.templates_cache)} community links")
            except Exception as e:
                print(f"Failed to load template link cache: {e}")
        
        # Load file key cache
        self.filekeys_cache = {}
        if os.path.exists(self.filekeys_cache_file):
            try:
                with open(self.filekeys_cache_file, 'r', encoding='utf-8') as f:
                    self.filekeys_cache = json.load(f)
                print(f"Loaded file key cache: {len(self.filekeys_cache)} template links")
            except Exception as e:
                print(f"Failed to load file key cache: {e}")
        
        # Load failed URL cache
        self.failed_urls_cache = {}
        if os.path.exists(self.failed_urls_cache_file):
            try:
                with open(self.failed_urls_cache_file, 'r', encoding='utf-8') as f:
                    self.failed_urls_cache = json.load(f)
                print(f"Loaded failed URL cache: {len(self.failed_urls_cache)}")
            except Exception as e:
                print(f"Failed to load failed URL cache: {e}")
    
    def get_cache_key(self, url):
        """Generate a cache key for the URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def is_recently_failed(self, url, hours=24):
        """Check if the URL has failed within the specified time."""
        cache_key = self.get_cache_key(url)
        if cache_key in self.failed_urls_cache:
            fail_time = self.failed_urls_cache[cache_key].get('timestamp', 0)
            return time.time() - fail_time < hours * 3600
        return False
    
    def cache_failed_url(self, url, error_msg=""):
        """Cache a failed URL."""
        cache_key = self.get_cache_key(url)
        self.failed_urls_cache[cache_key] = {
            'url': url,
            'error': error_msg,
            'timestamp': time.time(),
            'fail_count': self.failed_urls_cache.get(cache_key, {}).get('fail_count', 0) + 1
        }
        self.save_failed_urls_cache()
    
    def save_failed_urls_cache(self):
        """Save the failed URL cache."""
        try:
            with open(self.failed_urls_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.failed_urls_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save failed URL cache: {e}")
    
    def get_cached_templates(self, community_url):
        """Get cached template links."""
        cache_key = self.get_cache_key(community_url)
        if cache_key in self.templates_cache:
            cached_data = self.templates_cache[cache_key]
            print(f"Using cached template links: {len(cached_data['templates'])}")
            return cached_data['templates']
        return None
    
    def cache_templates(self, community_url, templates):
        """Cache template links."""
        cache_key = self.get_cache_key(community_url)
        self.templates_cache[cache_key] = {
            'url': community_url,
            'templates': templates,
            'timestamp': time.time(),
            'count': len(templates)
        }
        self.save_templates_cache()
        print(f"Cached {len(templates)} template links")
    
    def get_cached_filekey(self, template_url):
        """Get cached file key."""
        cache_key = self.get_cache_key(template_url)
        if cache_key in self.filekeys_cache:
            cached_data = self.filekeys_cache[cache_key]
            print(f"Using cached file key: {cached_data['filekey']}")
            return cached_data['filekey']
        return None
    
    def cache_filekey(self, template_url, filekey):
        """Cache file key."""
        cache_key = self.get_cache_key(template_url)
        self.filekeys_cache[cache_key] = {
            'url': template_url,
            'filekey': filekey,
            'timestamp': time.time()
        }
        self.save_filekeys_cache()
        print(f"Cached file key: {filekey}")
    
    def save_templates_cache(self):
        """Save the template link cache."""
        try:
            with open(self.templates_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.templates_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save template link cache: {e}")
    
    def save_filekeys_cache(self):
        """Save the file key cache."""
        try:
            with open(self.filekeys_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.filekeys_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save file key cache: {e}")
    
    def get_cache_stats(self):
        """Get cache statistics."""
        return {
            'templates_cache_count': len(self.templates_cache),
            'filekeys_cache_count': len(self.filekeys_cache),
            'failed_urls_count': len(self.failed_urls_cache),
            'templates_cache_size': os.path.getsize(self.templates_cache_file) if os.path.exists(self.templates_cache_file) else 0,
            'filekeys_cache_size': os.path.getsize(self.filekeys_cache_file) if os.path.exists(self.filekeys_cache_file) else 0
        }

# =======================================================
# Initialize WebDriver (Enhanced)
# =======================================================
def get_chrome_driver_with_options():
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")  # Uncomment this line for debugging
    
    # Basic options
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-popup-blocking")
    
    # Anti-detection options
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    
    # Randomize User-Agent
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ]
    options.add_argument(f"user-agent={random.choice(user_agents)}")
    
    # # Disable WebGL warnings
    # options.add_argument("--log-level=3")
    # options.add_argument("--silent")
    # options.add_argument("--disable-logging")
    # options.add_argument("--disable-extensions")
    # options.add_argument("--disable-web-security")
    # options.add_argument("--disable-webgl")
    # options.add_argument("--disable-webgl2")
    # options.add_argument("--disable-software-rasterizer")

    try:
        service = Service(ChromeDriverManager().install())
        service.log_path = os.devnull if os.name != 'nt' else 'NUL'
        driver = webdriver.Chrome(service=service, options=options)
        
        # Hide automation features
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
                window.chrome = {runtime: {}};
            """
        })
        
        # Set random window size
        widths = [1366, 1920, 1440, 1536]
        heights = [768, 1080, 900, 864]
        width = random.choice(widths)
        height = random.choice(heights)
        driver.set_window_size(width, height)
        
        return driver
    except Exception as e:
        print(f"Failed to initialize WebDriver: {e}")
        raise

# =======================================================
# Login to Figma (Enhanced)
# =======================================================
def login_to_figma(driver, email, password, rate_limiter):
    print("Logging into Figma...")
    
    # Rate limit wait
    rate_limiter.wait_if_needed()
    
    try:
        driver.get("https://www.figma.com/login")
        
        # Random wait to simulate human behavior
        time.sleep(random.uniform(2, 4))
        
        # Enter email
        email_input = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.NAME, "email"))
        )
        # Simulate human typing
        for char in email:
            email_input.send_keys(char)
            time.sleep(random.uniform(0.05, 0.15))
        
        time.sleep(random.uniform(1, 2))
        
        # Enter password
        password_input = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.NAME, "password"))
        )
        # Simulate human typing
        for char in password:
            password_input.send_keys(char)
            time.sleep(random.uniform(0.05, 0.15))
        
        time.sleep(random.uniform(1, 2))
        
        # Click login
        login_btn = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Log in')]"))
        )
        login_btn.click()
        
        # Wait for successful login
        WebDriverWait(driver, 25).until(EC.url_changes("https://www.figma.com/login"))
        print("Login successful!")
        rate_limiter.record_success()
        return True
        
    except Exception as e:
        print(f"Login failed: {e}")
        rate_limiter.record_failure()
        driver.save_screenshot("login_error.png")
        return False

# =======================================================
# Smart Wait Function
# =======================================================
def smart_wait(driver, min_time=1, max_time=3):
    """Smart wait to simulate human behavior."""
    wait_time = random.uniform(min_time, max_time)
    time.sleep(wait_time)

def wait_for_page_load(driver, timeout=20):
    """Wait for the page to load completely."""
    try:
        WebDriverWait(driver, timeout).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        # Extra wait for JavaScript execution
        time.sleep(random.uniform(2, 4))
        return True
    except TimeoutException:
        print("Page load timed out")
        return False

# =======================================================
# Handle WebGL Alert (Enhanced)
# =======================================================
def handle_webgl_alert(driver, timeout=5):
    try:
        WebDriverWait(driver, timeout).until(EC.alert_is_present())
        alert = driver.switch_to.alert
        alert.accept()
        print("Handled WebGL alert")
        return True
    except TimeoutException:
        return False
    except Exception as e:
        print(f"Failed to handle alert: {e}")
        return False

# =======================================================
# Extract Template Links from a Single Community Page (Enhanced)
# =======================================================
def extract_templates_from_community(driver, community_url, max_templates_per_page=200, cache_manager=None, rate_limiter=None):
    """Extract template links from a single community page."""
    # First, try to get from cache
    if cache_manager:
        cached_templates = cache_manager.get_cached_templates(community_url)
        if cached_templates:
            return cached_templates
    
    community_file_urls = set()
    
    try:
        print(f"Accessing: {community_url}")
        
        # Rate limit wait
        if rate_limiter:
            rate_limiter.wait_if_needed()
        
        driver.get(community_url)
        
        # Wait for the page to load completely
        if not wait_for_page_load(driver):
            raise Exception("Page failed to load")
        
        # Wait for template elements to appear
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href*="/community/file/"]'))
        )
        print("Template list page loaded")

        # Get template links by clicking "Load more"
        print(f"Loading template links (target: max {max_templates_per_page})...")
        load_more_attempts = 0
        max_load_more_attempts = 20  # Max 20 attempts to load more
        
        while len(community_file_urls) < max_templates_per_page and load_more_attempts < max_load_more_attempts:
            # Extract all template links on the current page
            links = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/community/file/"]')
            for link in links:
                href = link.get_attribute("href")
                if href and "/community/file/" in href:
                    community_file_urls.add(href.split("?")[0])  # Deduplicate
            
            # Show current progress
            print(f"Got {len(community_file_urls)}/{max_templates_per_page} template links")
            
            # Stop if the target is reached
            if len(community_file_urls) >= max_templates_per_page:
                print(f"Reached target of {max_templates_per_page} links")
                break
            
            # Try to click the "Load more" button
            try:
                # Smart scroll to the bottom of the page
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                smart_wait(driver, 2, 4)
                
                load_more_btn = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Load more') or contains(., 'Load more')]"))
                )
                
                # Scroll to the button's position
                driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", load_more_btn)
                smart_wait(driver, 1, 2)
                
                # Check button status
                if not load_more_btn.is_displayed() or not load_more_btn.is_enabled():
                    print("Load more button is not available, stopping.")
                    break
                
                # Use JavaScript to click
                driver.execute_script("arguments[0].click();", load_more_btn)
                print(f"Clicked Load more ({load_more_attempts + 1} time(s)), loading more templates...")
                load_more_attempts += 1
                
                # Wait for new content to load
                smart_wait(driver, 3, 6)
                
                # Rate limit wait
                if rate_limiter:
                    rate_limiter.wait_if_needed()
                
            except TimeoutException:
                print("No more Load more buttons, all templates loaded.")
                break
            except Exception as e:
                print(f"Failed to click Load more: {e}, stopping.")
                break

        # Limit the final number of links
        community_file_urls = list(community_file_urls)[:max_templates_per_page]
        print(f"Got {len(community_file_urls)} template links from {community_url}")
        
        # Cache the result
        if cache_manager:
            cache_manager.cache_templates(community_url, community_file_urls)
        
        if rate_limiter:
            rate_limiter.record_success()
        
    except Exception as e:
        print(f"Failed to extract template links from {community_url}: {e}")
        if rate_limiter:
            rate_limiter.record_failure()
        if cache_manager:
            cache_manager.cache_failed_url(community_url, str(e))
        return []
    
    return community_file_urls

# =======================================================
# Core Logic: Get Template Links and Extract File Keys (Enhanced)
# =======================================================
def get_figma_templates_from_multiple_urls(community_urls, email, password, max_templates_per_page=200, max_total_templates=1000):
    """Extract templates from multiple community links."""
    # Initialize managers
    cache_manager = CacheManager()
    rate_limiter = RateLimitManager()
    
    all_file_keys = set()
    all_community_file_urls = set()
    log_file = "log.txt"
    failed_urls_file = "failed_urls.txt"
    
    # Initialize log file
    with open(log_file, "w", encoding='utf-8') as f:
        f.write("Figma File Keys Log\n")
        f.write("====================\n")
        f.write(f"Processing time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of target community links: {len(community_urls)}\n")
        f.write(f"Max templates per page: {max_templates_per_page}\n")
        f.write(f"Max total templates: {max_total_templates}\n\n")

    # Initialize failed URL file
    with open(failed_urls_file, "w", encoding='utf-8') as f:
        f.write("Failed URLs Log\n")
        f.write("===============\n\n")

    driver = None
    try:
        driver = get_chrome_driver_with_options()
        if not login_to_figma(driver, email, password, rate_limiter):
            return []

        # Iterate through each community link
        for url_index, community_url in enumerate(community_urls, 1):
            print(f"\n{'='*60}")
            print(f"Processing community link {url_index}/{len(community_urls)}")
            print(f"Link: {community_url}")
            print(f"{'='*60}")
            
            # Check if the total template limit has been reached
            if len(all_community_file_urls) >= max_total_templates:
                print(f"Reached total template limit of {max_total_templates}, stopping.")
                break
            
            # Extract template links from the current community page (with cache and rate limiting)
            page_urls = extract_templates_from_community(
                driver, community_url, max_templates_per_page, cache_manager, rate_limiter
            )
            
            # Add to the total set of links
            for url in page_urls:
                if len(all_community_file_urls) < max_total_templates:
                    all_community_file_urls.add(url)
                else:
                    break
            
            print(f"Current cumulative template links: {len(all_community_file_urls)}/{max_total_templates}")
            
            # Add a separator to the log
            with open(log_file, "a", encoding='utf-8') as f:
                f.write(f"\n--- Community Link {url_index}: {community_url} ---\n")
                f.write(f"Got {len(page_urls)} template links\n")

        # Convert to list and limit the number
        all_community_file_urls = list(all_community_file_urls)[:max_total_templates]
        print(f"\nGot a total of {len(all_community_file_urls)} template links")

        # =======================================================
        # Extract file key for each link and save to log.txt in real-time (with cache and rate limiting)
        # =======================================================
        total = len(all_community_file_urls)
        cached_count = 0
        extracted_count = 0
        skipped_count = 0
        
        for i, file_url in enumerate(all_community_file_urls, 1):
            print(f"\nProcessing template {i}/{total}: {file_url}")
            
            # Check if it failed recently
            if cache_manager.is_recently_failed(file_url, hours=24):
                print(f"Skipping recently failed URL: {file_url}")
                skipped_count += 1
                continue
            
            # First, try to get file key from cache
            cached_filekey = cache_manager.get_cached_filekey(file_url)
            if cached_filekey:
                all_file_keys.add(cached_filekey)
                cached_count += 1
                print(f"Using cached file key: {cached_filekey}")
                
                # Save to log
                with open(log_file, "a", encoding='utf-8') as f:
                    f.write(f"{cached_filekey}\n")
                continue
            
            # Not in cache, need to extract it
            try:
                # Rate limit wait
                rate_limiter.wait_if_needed()
                
                driver.get(file_url)
                handle_webgl_alert(driver)
                
                # Wait for page to load
                if not wait_for_page_load(driver):
                    raise Exception("Page load timed out")
                
                # Random wait to simulate human behavior
                smart_wait(driver, 2, 4)

                # Click "Open in Figma" button (opens a new window)
                open_btn = WebDriverWait(driver, 15).until(
                    EC.element_to_be_clickable((By.XPATH, 
                        "//button[contains(@class, 'cta_button') and contains(., 'Open in Figma')]"
                    ))
                )
                original_window = driver.current_window_handle
                
                # Simulate human click
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", open_btn)
                smart_wait(driver, 1, 2)
                open_btn.click()

                # Switch to the new window
                WebDriverWait(driver, 15).until(EC.number_of_windows_to_be(2))
                for handle in driver.window_handles:
                    if handle != original_window:
                        driver.switch_to.window(handle)
                        break

                # Wait for the new window to load
                smart_wait(driver, 3, 5)

                # Extract file key
                WebDriverWait(driver, 20).until(
                    lambda d: "/file/" in d.current_url or "/design/" in d.current_url
                )
                new_url = driver.current_url
                match = re.search(r'/(file|design)/([^/]+)/', new_url)
                if match:
                    file_key = match.group(2)
                    all_file_keys.add(file_key)
                    extracted_count += 1
                    print(f"✅ Extracted file key: {file_key}")
                    
                    # Cache file key
                    cache_manager.cache_filekey(file_url, file_key)
                    
                    # Save to log.txt in real-time
                    with open(log_file, "a", encoding='utf-8') as f:
                        f.write(f"{file_key}\n")
                    print(f"Saved to {log_file}")
                    
                    # Record success
                    rate_limiter.record_success()
                else:
                    raise Exception("Could not extract file key from URL")

                # Close the new window and return to the original window
                driver.close()
                driver.switch_to.window(original_window)
                
                # Smart wait
                smart_wait(driver, 1, 3)

            except Exception as e:
                print(f"❌ Processing failed: {e}")
                
                # Record failure
                rate_limiter.record_failure()
                cache_manager.cache_failed_url(file_url, str(e))
                
                # Save failed URL
                with open(failed_urls_file, "a", encoding='utf-8') as f:
                    f.write(f"{file_url} - {str(e)}\n")
                
                # If it's a rate limit related error, add extra wait
                if any(keyword in str(e).lower() for keyword in ['rate limit', 'too many', '429', 'blocked']):
                    print("⚠️ Possible rate limit detected, adding extra wait time...")
                    time.sleep(random.uniform(30, 60))
                
                # Try to close any new windows that may exist
                try:
                    if len(driver.window_handles) > 1:
                        driver.close()
                        driver.switch_to.window(original_window)
                except:
                    pass
                
                continue
            
            # Show progress statistics every 10 templates
            if i % 10 == 0:
                success_rate = (cached_count + extracted_count) / i * 100
                print(f"\n📊 Progress ({i}/{total}):")
                print(f"   Success: {cached_count + extracted_count} (Cached: {cached_count}, Newly extracted: {extracted_count})")
                print(f"   Skipped: {skipped_count}")
                print(f"   Success rate: {success_rate:.1f}%")
                print(f"   Remaining: {total - i}")

    except KeyboardInterrupt:
        print("\n⚠️ User interrupted the program")
    except Exception as e:
        print(f"❌ Program error: {e}")
    finally:
        if driver:
            driver.quit()
            print("Browser closed")

    # Get cache statistics
    cache_stats = cache_manager.get_cache_stats()
    
    # Write final statistics
    with open(log_file, "a", encoding='utf-8') as f:
        f.write(f"\n--- Final Statistics ---\n")
        f.write(f"Total links processed: {len(all_community_file_urls)}\n")
        f.write(f"Successfully extracted file keys: {len(all_file_keys)}\n")
        f.write(f"Cache hits: {cached_count}\n")
        f.write(f"Newly extracted: {extracted_count}\n")
        f.write(f"Skipped: {skipped_count}\n")
        f.write(f"Success rate: {(cached_count + extracted_count) / len(all_community_file_urls) * 100:.1f}%\n")
        f.write(f"Completion time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n--- Cache Statistics ---\n")
        f.write(f"Template link cache count: {cache_stats['templates_cache_count']}\n")
        f.write(f"File key cache count: {cache_stats['filekeys_cache_count']}\n")
        f.write(f"Failed URL cache count: {cache_stats['failed_urls_count']}\n")

    print(f"\n🎉 All processing complete!")
    print(f"📈 Statistics:")
    print(f"   Total links: {len(all_community_file_urls)}")
    print(f"   Successfully extracted: {len(all_file_keys)} file keys")
    print(f"   Cache hits: {cached_count}")
    print(f"   Newly extracted: {extracted_count}")
    print(f"   Skipped failures: {skipped_count}")
    print(f"   Success rate: {(cached_count + extracted_count) / len(all_community_file_urls) * 100:.1f}%")
    print(f"📁 File output:")
    print(f"   Detailed log: {log_file}")
    print(f"   Failed URLs: {failed_urls_file}")
    print(f"   Cache stats: Templates {cache_stats['templates_cache_count']}, File keys {cache_stats['filekeys_cache_count']}")
    
    return list(all_file_keys)

# =======================================================
# Main Program Entry Point
# =======================================================
if __name__ == "__main__":
    try:
        from ...configs.paths import DATA_DIR
        file_keys_file = DATA_DIR / "file_keys.txt"
    except Exception:
        file_keys_file = "file_keys.txt"

    print("🚀 Figma Template Scraper Started")
    print("=" * 50)
    
    # Read community URL list
    community_urls = DEFAULT_URLS
    
    print(f"📋 Will process the following {len(community_urls)} community links:")
    for i, url in enumerate(community_urls, 1):
        print(f"   {i}. {url}")
    
    print(f"\n⚙️ Configuration:")
    print(f"   Max templates per page: 300")
    print(f"   Max total templates: 2500")
    print(f"   Rate limit policy: Max 30 req/min, Max 500 req/hr")
    print(f"   Delay range: 2-8 seconds (increases on failure)")
    
    # Confirm start
    try:
        input("\nPress Enter to start, or Ctrl+C to cancel...")
    except KeyboardInterrupt:
        print("\n❌ User cancelled operation")
        exit()
    
    # Record start time
    start_time = time.time()
    
    # Extract templates from multiple community links
    extracted_keys = get_figma_templates_from_multiple_urls(
        community_urls, 
        FIGMA_EMAIL, 
        FIGMA_PASSWORD, 
        max_templates_per_page=300,  # Max 300 templates per page
        max_total_templates=2500     # Max 2500 templates in total
    )

    # Calculate total time taken
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\n⏱️ Total time taken: {hours}h {minutes}m {seconds}s")
    
    if extracted_keys:
        print(f"✅ Finally extracted {len(extracted_keys)} file keys (saved to log.txt)")
        
        # Save deduplicated file keys to a separate file
        with open(file_keys_file, "w", encoding='utf-8') as f:
            for key in sorted(extracted_keys):
                f.write(f"{key}\n")
        print(f"📄 Deduplicated file keys saved to {file_keys_file}")
    else:
        print("❌ No file keys were extracted")
    
    print("\n🏁 Program finished")
