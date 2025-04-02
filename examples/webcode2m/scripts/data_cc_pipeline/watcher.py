import os
import sys
import time
import argparse
import psutil

class FileCounter:
    def __init__(self, directory, interval, log_file, alpha=0.1):
        self.directory = directory
        self.file_count = 0
        self.interval = interval
        self.log_file = log_file
        self.alpha = alpha
        self.smoothed_speed = 0

        # Check if log file exists and clear it if it does
        if os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write('')

    def get_cpu_usage(self, interval=1):
        cpu_usage = psutil.cpu_percent(interval=interval)
        return cpu_usage

    def count_files(self):
        return len(os.listdir(self.directory))

    def estimate_speed(self, prev_count, current_count, elapsed_time):
        files_processed = current_count - prev_count
        speed = files_processed / elapsed_time if elapsed_time > 0 else 0
        self.smoothed_speed = self.alpha * speed + (1 - self.alpha) * self.smoothed_speed
        return self.smoothed_speed

    def display_progress(self):
        while True:
            start_time = time.time()
            prev_count = self.file_count
            self.file_count = self.count_files()
            speed = self.estimate_speed(prev_count, self.file_count, self.interval)
            cpu_usage = self.get_cpu_usage()
            log_message = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Cpu usage: {cpu_usage}, Files count: {self.file_count}, Speed: {speed:.2f} files/s\n"
            with open(self.log_file, 'a') as f:
                f.write(log_message)
            elapsed_time = time.time() - start_time
            time.sleep( max(1, self.interval - elapsed_time))

def main():
    parser = argparse.ArgumentParser(description='Count files in a directory and log the progress.')
    parser.add_argument('directory', nargs='?', default='/data02/users/lz/code/UICoder/datasets/cc-wash/H128-2560_C128-4096_R2/c4-format', help='Directory path')
    parser.add_argument('-i', '--interval', type=int, default=2, help='Interval in seconds for counting')
    parser.add_argument('-l', '--log-file', default='/data02/users/lz/code/UICoder/outputs/watcher.log', help='Path to the log file')
    args = parser.parse_args()

    directory_path = args.directory
    interval = args.interval
    log_file = args.log_file

    if directory_path and not os.path.exists(directory_path):
        print("Directory does not exist.")
        sys.exit(1)

    counter = FileCounter(directory_path, interval, log_file, 0.1)
    counter.display_progress()

if __name__ == "__main__":
    main()
