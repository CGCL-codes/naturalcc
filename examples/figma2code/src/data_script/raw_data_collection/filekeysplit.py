"""
split file_keys.txt into multiple files
"""
from pathlib import Path

def filekeysplit(file_keys_file: Path, split_count: int):
    with open(file_keys_file, 'r') as f:
        data = f.read()

    data = data.split('\n')
    print(f"Total data size: {len(data)}")
    for i in range(split_count):
        split_file_keys_file = file_keys_file.parent / f'filekeys_split{i}.txt'
        split_data = data[i * (len(data) // split_count):(i + 1) * (len(data) // split_count)]
        with open(split_file_keys_file, 'w') as f:
            f.write('\n'.join(split_data))
        print(f"Split {i} size: {len(split_data)}")
    
if __name__ == "__main__":
    from ...configs.paths import DATA_DIR
    file_keys_file = DATA_DIR / 'file_keys.txt'
    filekeysplit(file_keys_file, 4)


