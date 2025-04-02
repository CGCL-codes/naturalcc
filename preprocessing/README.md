# The NaturalCC Dataset Package

## Usage
To prepare a dataset for a specific task, use the `prepare_dataset` method. 
```python
DATASET_NAME = "codesearchnet"
import ncc_dataset
ncc_dataset.prepare_dataset(DATASET_NAME)
```

Currently, this method only supports `codesearchnet`, `python-wan` and `typilus`, to manipulate other datasets, 
please visit the corresponding dataset folder.

