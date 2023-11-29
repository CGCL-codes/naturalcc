```shell
# amd
python -m dataset.opencl.mapping.preprocess -f config/amd
python -m dataset.opencl.inst2vec_mapping.preprocess -f config/amd
# nvidia
python -m dataset.opencl.mapping.preprocess -f config/nvidia
python -m dataset.opencl.inst2vec_mapping.preprocess -f config/nvidia
```