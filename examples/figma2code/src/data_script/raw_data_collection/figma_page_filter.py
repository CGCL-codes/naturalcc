"""
Used to download page information for manual filtering and annotation, generating a whitelist CSV.

Usage:
python figma_page_filter.py --mode download --filekeys data/filekeys.txt
python figma_page_filter.py --mode download --filekeys data/filekeyTest.txt
python figma_page_filter.py --mode filter
python figma_page_filter.py --mode trash

mode:
    download: Download page information and render images.
    filter: Generate a whitelist CSV.
    trash:    Generate a blacklist CSV.

Output directory structure:
```
output/page_filter/
├── candidate/
│   ├── page_stats.json
│   ├── filekey1/
│   │   ├── filekey1.json
│   │   ├── page_id1.png
│   │   ├── page_id2.png
│   │   └── ...
│   ├── filekey2/
│   │   ├── filekey2.json
│   │   ├── page_id1.png
│   │   ├── page_id2.png
│   │   └── ...
│   └── ...
├── selected/
│   ├── filekey2/
│   │   ├── page_id1.png
│   │   ├── page_id2.png
│   │   └── ...
│   └── ...
├── trash/
│   └── filekey3/...
├── whitelist.csv
└── blacklist.csv
```
"""



import os
import shutil
from typing import Dict, Optional
from ...utils.figma_utils import FigmaSession
from ...utils.files import save_json, load_json
from ...utils.console_logger import console, setup_logging, logger, create_progress
from argparse import ArgumentParser

class FigmaPageFilter:
    """Figma page filter, used to download page information and render images for manual filtering."""
    
    def __init__(self, output_dir: str = "output/page_filter", token: str = None):
        """
        Initializes the FigmaPageFilter.
        
        Args:
            output_dir (str): The output directory path.
        """
        self.output_dir = output_dir
        self.candidate_dir = os.path.join(output_dir, "candidate")
        self.selected_dir = os.path.join(output_dir, "selected")
        self.trash_dir = os.path.join(output_dir, "trash")
        self.stats_file = os.path.join(self.candidate_dir, "page_stats.json")
        
        # Create necessary directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.candidate_dir, exist_ok=True)
        os.makedirs(self.selected_dir, exist_ok=True)
        os.makedirs(self.trash_dir, exist_ok=True)
        
        # Initialize Figma session
        self.figma_session = FigmaSession(token)
        
        # Set up logging
        setup_logging(logger, "figma_page_filter", output_dir)
        self.logger = logger
        
        # Statistics
        self.page_stats = {}
        if os.path.exists(self.stats_file):
            self.page_stats = load_json(self.stats_file)
        self.whitelist = []
        self.blacklist = []
        
        # Define CANVAS name patterns to exclude (case-insensitive)
        # self.excluded_canvas_patterns = [
        #     'cover','thumbnail', 'preview', 'icon', 'logo',
        #     'background', 'bg', 'template', 'mockup',
        #     'wireframe', 'sketch', 'draft', 'concept', 'old', 
        #     'archive', 'backup', 'test', 'demo', 'example',
        #     'temp', 'placeholder', 'empty', 'blank', 
        #     'unused', 'deprecated'
        # ]
        self.excluded_canvas_patterns = [] # Conservatively don't delete any for now, filter by rules after all are downloaded to see the effect.
    
    # ============ Original candidate page download logic ============
    def process_filekeys(self, filekeys_file: str) -> None:
        """
        Processes the filekeys.txt file, downloading page information for all Figma files.
        
        Args:
            filekeys_file (str): The path to the filekeys.txt file.
        """
        if not os.path.exists(filekeys_file):
            raise FileNotFoundError(f"File not found: {filekeys_file}")
        
        # Read filekeys
        with open(filekeys_file, 'r', encoding='utf-8') as f:
            filekeys = [line.strip() for line in f if line.strip()]
        
        console.print(f"[bold blue]Starting to process {len(filekeys)} Figma files...[/bold blue]")

        page_find = 0
        file_failed = 0
        
        with create_progress() as progress:
            task_id = progress.add_task("Processing filekeys", total=len(filekeys), status="Preparing...")
            for filekey in filekeys:
                progress.update(task_id, status=f"Processing {filekey}... {page_find} pages downloaded.")
                try:
                    page_find += self.process_single_filekey(filekey)
                    progress.update(task_id, advance=1, status=f"{filekey} success")
                except Exception as e:
                    file_failed += 1
                    progress.update(task_id, advance=1, status=f"{filekey} failed")
        
        # Save statistics
        console.print(f"[bold green]Found {page_find} pages in total, {file_failed} files failed.[/bold green]")
        self.logger.info(f"Found {page_find} pages in total, {file_failed} files failed.")
        self.save_page_stats()
    
    def process_single_filekey(self, filekey: str) -> int:
        """
        Processes a single filekey.
        
        Args:
            filekey (str): The Figma file key.
        """
        
        # Create filekey directory
        filekey_dir = os.path.join(self.candidate_dir, filekey)
        os.makedirs(filekey_dir, exist_ok=True)
        
        try:
            # Download JSON data
            if os.path.exists(os.path.join(filekey_dir, f"{filekey}.json")):
                json_data = load_json(os.path.join(filekey_dir, f"{filekey}.json"))
            else:
                json_data = self.figma_session.get_raw_json(filekey)
                json_path = os.path.join(filekey_dir, f"{filekey}.json")
                save_json(json_data, json_path)
            
            # Get page information
            pages_info = self.extract_pages_info(json_data)
            
            # Render page images
            pages_info = self.render_pages(filekey, pages_info, filekey_dir)

            # Save page information to statistics
            self.page_stats[filekey] = pages_info
            return len(pages_info)
            
        except Exception as e:
            self.logger.error(f"Failed to process file {filekey}: {e}")
            raise
    
    def extract_pages_info(self, json_data: dict) -> Dict[str, str]:
        """
        Extracts page information from JSON data.
        
        Args:
            json_data (dict): The Figma file JSON data.
            
        Returns:
            Dict[str, str]: A mapping from page ID to page name.
        """
        pages_info = {}
        
        try:
            document = json_data.get('document', {})
            children = document.get('children', [])
            
            # Iterate through all CANVAS nodes
            for canvas in children:
                if canvas.get('type') == 'CANVAS':
                    canvas_name = canvas.get('name', '').lower().strip()
                    
                    # Check if the CANVAS name is in the exclusion list
                    should_exclude = False
                    for pattern in self.excluded_canvas_patterns:
                        if pattern in canvas_name:
                            should_exclude = True
                            self.logger.debug(f"Excluding CANVAS: {canvas.get('name', '')} (matched pattern: {pattern})")
                            break
                    
                    if should_exclude:
                        continue
                    
                    canvas_children = canvas.get('children', [])
                    
                    # Iterate through the FRAME nodes under CANVAS
                    for frame in canvas_children:
                        if frame.get('type') == 'FRAME':
                            node_id = frame.get('id', '')
                            name = frame.get('name', '')
                            
                            if node_id and name:
                                pages_info[node_id] = name
        
        except Exception as e:
            self.logger.error(f"Failed to extract page information: {e}")
            raise
        
        return pages_info
    
    def render_pages(self, filekey: str, pages_info: Dict[str, str], filekey_dir: str) -> Dict[str, str]:
        """
        Render page images
        
        Args:
            filekey (str): Figma file key
            pages_info (Dict[str, str]): Page information
            filekey_dir (str): File directory
        """
        if not pages_info:
            self.logger.warning(f"No pages found for file {filekey}")
            return pages_info
        
        # Get the list of page IDs
        page_ids = list(pages_info.keys())
        # Check the current download status in the directory
        files_in_dir = os.listdir(filekey_dir)
        already_downloaded = [f[:-4] for f in files_in_dir if f.endswith('.png')]
        page_ids = [page_id for page_id in page_ids if page_id not in already_downloaded]
        if len(page_ids) == 0:
            self.logger.debug(f"All {len(already_downloaded)} pages in file {filekey} have been downloaded, skipping")
            return pages_info
        else:
            self.logger.debug(f"{len(already_downloaded)} pages have been downloaded in file {filekey}, {len(page_ids)} pages need to be downloaded now")
        try:
            # Get rendered image URLs
            download_urls, failed_ids = self.figma_session.get_render_image_urls(
                filekey, page_ids, format='png', scale=2
            )
        except Exception as e:
            self.logger.error(f"Failed to get rendered image URL: {e}")
            pages_info = {page_id: pages_info[page_id] for page_id in already_downloaded}
            return pages_info

        if failed_ids:
            self.logger.warning(f"Failed to get rendered URL for {len(failed_ids)} pages in file {filekey}: {failed_ids}")
            
        # Download images
        new_downloaded = []
        for page_id, download_url in download_urls.items():
            try:
                file_ext = self.figma_session.download_image_from_url(
                    download_url, filekey_dir, page_id, ".png"
                )
                new_downloaded.append(page_id)
            except Exception as e:
                self.logger.error(f"Failed to download image for page {page_id}: {e}")
        already_downloaded.extend(new_downloaded)
        pages_info = {page_id: pages_info[page_id] for page_id in already_downloaded}
        self.logger.debug(f"Newly downloaded {len(new_downloaded)} pages, total downloaded {len(already_downloaded)} pages")
        return pages_info

    
    def save_page_stats(self) -> None:
        """Saves page statistics."""
        try:
            save_json(self.page_stats, self.stats_file)
            console.print(f"[green]Statistics saved to: {self.stats_file}[/green]")
        except Exception as e:
            self.logger.error(f"Failed to save statistics: {e}")
            raise
    
    # ============ Whitelist Logic ============
    def copy_selected_pages(self) -> None:
        """
        Copies the remaining images from the candidate directory to the selected directory.
        Reads the statistics data to build a whitelist.
        """
        if not os.path.exists(self.stats_file):
            raise FileNotFoundError(f"Statistics file not found: {self.stats_file}")
        
        # Load statistics data
        self.page_stats = load_json(self.stats_file)
        if not self.page_stats:
            raise ValueError("Statistics data is empty")
        
        console.print(f"[bold blue]Starting to copy selected page images...[/bold blue]")
        
        total_copied = 0
        
        for filekey, pages_info in self.page_stats.items():
            filekey_candidate_dir = os.path.join(self.candidate_dir, filekey)
            filekey_selected_dir = os.path.join(self.selected_dir, filekey)
            
            if not os.path.exists(filekey_candidate_dir):
                continue
            
            # Create selected directory
            os.makedirs(filekey_selected_dir, exist_ok=True)
            
            # Iterate through page information
            for page_id, page_name in pages_info.items():
                # Check if the image file exists
                image_path = os.path.join(filekey_candidate_dir, f"{page_id}.png")
                if os.path.exists(image_path):
                    # Copy to the selected directory
                    dest_path = os.path.join(filekey_selected_dir, f"{page_id}.png")
                    shutil.copy2(image_path, dest_path)
                    total_copied += 1
        
        console.print(f"[green]Done! Copied {total_copied} page images to the selected directory.[/green]")
    
    def build_whitelist_from_selected(self) -> None:
        """
        Builds a whitelist from the selected_dir directory structure.
        Directly views the page images present in the directory and gets related information from page_stats.
        """
        if not os.path.exists(self.stats_file):
            raise FileNotFoundError(f"Statistics file not found: {self.stats_file}")
        
        # Load statistics data
        self.page_stats = load_json(self.stats_file)
        if not self.page_stats:
            raise ValueError("Statistics data is empty")
        
        console.print(f"[bold blue]Building whitelist from the selected directory...[/bold blue]")
        
        self.whitelist = []
        
        # Iterate through the selected directory
        for filekey in os.listdir(self.selected_dir):
            filekey_selected_dir = os.path.join(self.selected_dir, filekey)
            
            # Check if it is a directory
            if not os.path.isdir(filekey_selected_dir):
                continue
            
            # Check if the filekey is in the statistics data
            if filekey not in self.page_stats:
                self.logger.warning(f"Found unknown filekey in the selected directory: {filekey}")
                continue
            
            # Get all page images under this filekey
            for filename in os.listdir(filekey_selected_dir):
                if filename.endswith('.png'):
                    # Extract page_id (remove .png suffix)
                    page_id = filename[:-4]
                    
                    # Get page name from page_stats
                    if page_id in self.page_stats[filekey]:
                        page_name = self.page_stats[filekey][page_id]
                        self.whitelist.append((filekey, page_id, page_name))
                    else:
                        self.logger.warning(f"Found unknown page_id in the selected directory: {filekey}/{page_id}")
        
        console.print(f"[green]Whitelist built, total {len(self.whitelist)} pages[/green]")
    
    def generate_whitelist_csv(self, output_file: Optional[str] = None) -> None:
        """
        Generates a whitelist CSV file.
        
        Args:
            output_file (str): The output CSV file path, defaults to output/page_filter/whitelist.csv.
        """
        if output_file is None:
            output_file = os.path.join(self.output_dir, "whitelist.csv")
        if not self.whitelist:
            raise ValueError("No whitelist available.")
        
        # Generate CSV content
        csv_lines = ["file_key,node_id,page_name,page_type"]
        
        for filekey, page_id, page_name in self.whitelist:
            # page_type is temporarily empty
            csv_lines.append(f"{filekey},{page_id},{page_name},")
        
        # Save CSV file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(csv_lines))
            console.print(f"[green]Whitelist CSV generated: {output_file}[/green]")
        except Exception as e:
            self.logger.error(f"Failed to generate whitelist CSV: {e}")
            raise
    # ============ Blacklist Logic ============
    def build_blacklist_from_trash(self) -> None:
        if not os.path.exists(self.stats_file):
            raise FileNotFoundError(f"Statistics file not found: {self.stats_file}")
        self.page_stats = load_json(self.stats_file)
        self.blacklist = []

        for filekey in os.listdir(self.trash_dir):
            filekey_trash_dir = os.path.join(self.trash_dir, filekey)
            if not os.path.isdir(filekey_trash_dir):
                continue
            if filekey not in self.page_stats:
                continue
            for filename in os.listdir(filekey_trash_dir):
                if filename.endswith('.png'):
                    page_id = filename[:-4]
                    if page_id in self.page_stats[filekey]:
                        page_name = self.page_stats[filekey][page_id]
                        self.blacklist.append((filekey, page_id, page_name))

    def generate_blacklist_csv(self, output_file: Optional[str] = None) -> None:
        if output_file is None:
            output_file = os.path.join(self.output_dir, "blacklist.csv")
        if not self.blacklist:
            raise ValueError("No blacklist available.")

        csv_lines = ["file_key,node_id,page_name"]
        for filekey, page_id, page_name in self.blacklist:
            csv_lines.append(f"{filekey},{page_id},{page_name}")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(csv_lines))
        console.print(f"[green]Blacklist CSV generated: {output_file}[/green]")
        
    # ============ Verification Logic ============
    def check_selection_consistency(self) -> None:
        """Verify if the total number of candidates equals the sum of selected and trash."""
        # Count the number of png files in candidate
        candidate_count = 0
        for filekey in os.listdir(self.candidate_dir):
            filekey_dir = os.path.join(self.candidate_dir, filekey)
            if os.path.isdir(filekey_dir):
                candidate_count += len([f for f in os.listdir(filekey_dir) if f.endswith(".png")])

        # Count the number of png files in selected
        selected_count = 0
        for filekey in os.listdir(self.selected_dir):
            filekey_dir = os.path.join(self.selected_dir, filekey)
            if os.path.isdir(filekey_dir):
                selected_count += len([f for f in os.listdir(filekey_dir) if f.endswith(".png")])

        # Count the number of png files in trash
        trash_count = 0
        for filekey in os.listdir(self.trash_dir):
            filekey_dir = os.path.join(self.trash_dir, filekey)
            if os.path.isdir(filekey_dir):
                trash_count += len([f for f in os.listdir(filekey_dir) if f.endswith(".png")])

        total_selected_trash = selected_count + trash_count

        console.print(f"[bold blue]Candidate: {candidate_count}, Selected: {selected_count}, Trash: {trash_count}, Total: {total_selected_trash}[/bold blue]")

        if candidate_count == total_selected_trash:
            console.print("[green]✅ Check passed: Filtering is complete, no omissions.[/green]")
        else:
            console.print("[red]❌ Check failed: Inconsistent quantities, please check for omissions.[/red]")
    
    def find_missing_filekeys(self, filekeys_file: str, output_file: Optional[str] = None) -> None:
        """
        Count the list of filekeys that have download problems.
        
        Args:
            filekeys_file (str): The original filekeys file path.
            output_file (str): The output file path for the list of missing filekeys, defaults to output/page_filter/filekey_miss.txt.
        """
        if output_file is None:
            output_file = os.path.join(self.output_dir, "filekey_miss.txt")
        
        if not os.path.exists(filekeys_file):
            raise FileNotFoundError(f"File not found: {filekeys_file}")
        
        # Read original filekeys
        with open(filekeys_file, 'r', encoding='utf-8') as f:
            original_filekeys = [line.strip() for line in f if line.strip()]
        
        console.print(f"[bold blue]Starting to check for missing filekeys...[/bold blue]")
        
        missing_filekeys = []
        incomplete_filekeys = []
        
        with create_progress() as progress:
            task_id = progress.add_task("Checking filekeys", total=len(original_filekeys), status="Preparing...")
            
            for filekey in original_filekeys:
                progress.update(task_id, status=f"Checking {filekey}...")
                
                # Check if the filekey directory exists
                filekey_dir = os.path.join(self.candidate_dir, filekey)
                if not os.path.exists(filekey_dir) or not os.path.isdir(filekey_dir):
                    missing_filekeys.append(filekey)
                    progress.update(task_id, advance=1, status=f"{filekey} directory does not exist")
                    continue
                
                # Check if the JSON file exists
                json_path = os.path.join(filekey_dir, f"{filekey}.json")
                if not os.path.exists(json_path):
                    missing_filekeys.append(filekey)
                    progress.update(task_id, advance=1, status=f"{filekey} JSON file does not exist")
                    continue
                
                try:
                    # Read the JSON file and extract page IDs
                    json_data = load_json(json_path)
                    expected_pages_info = self.extract_pages_info(json_data)
                    expected_pages = set(expected_pages_info.keys())
                    
                    # Get the actually downloaded PNG files
                    actual_pages = set([f[:-4] for f in os.listdir(filekey_dir) if f.endswith('.png')])
                    
                    # Check if all expected pages have been downloaded
                    if not expected_pages.issubset(actual_pages):
                        missing_pages = expected_pages - actual_pages
                        incomplete_filekeys.append((filekey, len(missing_pages), list(missing_pages)[:5]))
                        progress.update(task_id, advance=1, status=f"{filekey} is missing {len(missing_pages)} pages")
                    else:
                        progress.update(task_id, advance=1, status=f"{filekey} complete")
                except Exception as e:
                    self.logger.error(f"Failed to process file {filekey}: {e}")
                    missing_filekeys.append(filekey)
                    progress.update(task_id, advance=1, status=f"{filekey} processing failed: {e}")
        
        # Merge missing and incomplete filekeys
        all_problem_filekeys = missing_filekeys + [f[0] for f in incomplete_filekeys]
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_problem_filekeys))
        
        console.print(f"[green]Done! Found {len(missing_filekeys)} missing filekeys and {len(incomplete_filekeys)} incomplete filekeys.[/green]")
        console.print(f"[green]List of problematic filekeys saved to: {output_file}[/green]")
        
        # Output detailed information
        if incomplete_filekeys:
            console.print("[yellow]Incomplete filekeys and number of missing pages:[/yellow]")
            for filekey, missing_count, sample_pages in incomplete_filekeys:
                sample_str = ", ".join(sample_pages[:5])
                if missing_count > 5:
                    sample_str += f" and {missing_count - 5} more"
                console.print(f"  - {filekey}: Missing {missing_count} pages, e.g.: {sample_str}")



def main():
    # python figma_page_filter.py --filekeys data/filekeys_split<split_num>.txt --token your_Figma_Token
    parser = ArgumentParser()
    parser.add_argument("--filekeys", type=str, default="data/filekeys.txt")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output/page_filter")
    parser.add_argument("--mode", type=str, default="download",
                    choices=["download", "filter", "trash", "check", "missing"],
                    help="download - download candidate pages; filter - generate whitelist; trash - generate blacklist; check - verify filtering integrity; missing - count problematic filekeys")
    parser.add_argument("--output", type=str, default=None,
                    help="Output file path, only valid when mode=missing")

    args = parser.parse_args()

    from ...configs.paths import enter_project_root
    enter_project_root()
    
    filter_instance = FigmaPageFilter(args.output_dir, args.token)
    if args.mode == "download":
        filter_instance.process_filekeys(args.filekeys)
    elif args.mode == "filter":
        filter_instance.copy_selected_pages()
        filter_instance.build_whitelist_from_selected()
        filter_instance.generate_whitelist_csv()
    elif args.mode == "trash":
        filter_instance.build_blacklist_from_trash()
        filter_instance.generate_blacklist_csv()
    elif args.mode == "check":
        filter_instance.check_selection_consistency()
    elif args.mode == "missing":
        filter_instance.find_missing_filekeys(args.filekeys, args.output)



if __name__ == "__main__":
    main()
