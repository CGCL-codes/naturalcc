# hybrid retrieval task
generate retrieval dataset for CodeSearchNet

```shell
# for all CodeSearchNet dataset
python -m dataset.codesearchnet.retrieval.preprocess -f config/all

# only for ruby dataset
# python -m dataset.codesearchnet.retrieval.preprocess -f config/ruby
```