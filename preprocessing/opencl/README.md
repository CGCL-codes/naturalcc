# OpenCL dataset for heterogeneous mapping

## Step 1. download raw dataset

```shell
bash dataset/opencl/download.sh
```

## Step 2. cast attributes of data (optional)

This procedure requires to install ``clgen``, therefore, we recommend use our processed data.

If you are insterested in our data processing, please refer to [main.ipynb](dataset/opencl/explore/main.ipynb), Or run

```shell
python -m dataset.opencl.attributes_cast -l amd
python -m dataset.opencl.attributes_cast -l nvidia
```
