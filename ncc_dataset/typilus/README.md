# Dataset: Typilus

<hr>

# Step 1
Use the official typilus graph generator to convert code to typilus graph, please refer to 
[typilus/src/data_preparation](https://github.com/typilus/typilus/tree/master/src/data_preparation)

A brief command listing is provided below.
```shell
git clone https://github.com/typilus/typilus.git
cd typilus/src/data_preparation/
docker build -t typilus-env
docker run --rm -it -v <<typilus_data_store_path>>:/usr/data typilus-env:latest bash

(in docker)
bash scripts/prepare_data.sh metadata/typedRepos.txt
```

# Step 2
Preprocessing the dataset for naturalcc.
```python
import ncc_dataset
ncc_dataset.prepare_dataset("typilus", typilus_path=<<typilus_data_store_path>>)
```
