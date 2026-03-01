import os
import time
import logging
from abc import abstractmethod
from typing import Any, Optional
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich.logging import RichHandler
from rich.text import Text
from ..configs.paths import LOGS_DIR
from pathlib import Path

# ==================== Rich Console Initialization ====================
console = Console()

# Create custom progress bar style
def create_progress() -> Progress:
    """Create a unified progress bar style"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="bright_green"),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("[dim]{task.fields[status]}[/dim]"),
        console=console,
        refresh_per_second=2
    )

# ==================== Logging System ====================
# Only needs to be called once in main
def setup_logging(logger_obj: logging.Logger, output_dir: str | Path= LOGS_DIR, log_name: Optional[str]=None, add_timestamp: bool=True, level: int=logging.DEBUG, file_level: int=logging.DEBUG, console_level: int=logging.INFO):
    """
    Set up an optimized logging system - focusing on errors, warnings, and statistics.
    
    Args:
        logger_obj (logging.Logger): Logger object
        output_dir (str | Path): Output directory path, default is output/logs
        log_name (Optional[str]): Logger name, default is None, if None, use the logger_obj.name (the default name is Figma2Code)
        add_timestamp (bool): Whether to add timestamp to the log filename, default is True
        level (int): Logger level, default is DEBUG
        file_level (int): File logger level, default is DEBUG
        console_level (int): Console logger level, default is INFO
    """
    output_dir = Path(output_dir)
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    if log_name:
        logger_obj.name = log_name
    logger_obj.setLevel(level)

    # Create a log filename with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if add_timestamp:
        log_filename = f"{logger_obj.name}_{timestamp}.log"
    else:
        log_filename = f"{logger_obj.name}.log"
    log_path = output_dir / log_filename
    
    # Clear existing handlers (to avoid duplicate logs)
    for handler in logger_obj.handlers[:]:
        logger_obj.removeHandler(handler)
    
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(file_level)
    
    # Create console handler - use Rich format, only show INFO level and above
    console_handler = RichHandler(console=console, show_path=False, show_time=False)
    console_handler.setLevel(console_level)
    
    # Create a simplified formatter
    file_formatter = logging.Formatter(
        '[%(asctime)s - %(funcName)s - %(filename)s, line:%(lineno)d] - %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to the logger
    logger_obj.addHandler(file_handler)
    logger_obj.addHandler(console_handler)
    
    # Log the initialization of the logging system
    console.print(f"[dim]Log file: {log_path}[/dim]")
    logger_obj.info(f"=== Logging System Initialized - {timestamp} ===")

# Global logger
logger = logging.getLogger('Figma2Code')

# ==================== Base Class for Statistics ====================
class StatsBase:
    """Stage statistics"""
    def __init__(self, stats_name: str):
        self.stats_name = stats_name
        self.start_time = time.time()
        self.errors = []
    
    def add_error(self, error_msg: str):
        self.errors.append(error_msg)
    
    def finish(self):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
    
    @abstractmethod
    def get_summary_table(self):
        pass

if __name__ == "__main__":
    # Test the logging system
    setup_logging(logger, output_dir="./output/logs", log_name="test")
    logger.info("This is a test log")
    logger.warning("This is a test warning")
    logger.error("This is a test error")
    logger.critical("This is a test critical")
    logger.debug("This is a test debug")

    # Test Progress Bar
    task_num = 20
    progress = create_progress()
    task_id = progress.add_task("Processing Test", total=task_num, status="Preparing...")
    with progress:
        for i in range(task_num):
            progress.update(task_id, advance=1, status=f"Processing {i+1}/{task_num}...")
            time.sleep(0.1)
    console.print("[bold green]Test completed![/bold green]")