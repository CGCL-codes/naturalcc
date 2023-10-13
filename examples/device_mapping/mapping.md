# Heterogeneous Mapping

This README contains instructions for code retrieval (search) models.



## Datasets & Pre-Processing

### heterogeneous mapping
| Dataset | Description | Download & Pre-Processing | Processing for Modalities |
|:-------:|:-----------:|:-------------------------:|:------------:|
| OpenCL  |  Nvidia/AMD <br> ([Grewe et al., 2013](https://ece.northeastern.edu/groups/nucar/NUCARTALKS/cgo2013-grewe.pdf))   |  [README.md](dataset/opencl/README.md)     |  [Vanilla](dataset/opencl/mapping/README.md) <br> [Inst2Vec](dataset/opencl/inst2vec_mapping/README.md) <br>  |

## Training & Inference

### [heterogeneous mapping](run/mapping)
Example usage (deeptune)
Follow the instruction in [README.md](run/mapping/deeptune/README.md)
```shell
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.mapping.deeptune.train
# inference
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.mapping.deeptune.10fold
```

## Models
- [static mapping](run/mapping/static_mapping)
- [decision tree](run/mapping/decision_tree)
- [deeptune](run/mapping/deeptune)
- [inst2vec](run/mapping/inst2vec)