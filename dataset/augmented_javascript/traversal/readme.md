## Step 0: copy raw data as ```train.code```
If we want to obtain the AST and its traverse features (i.e., for `traverse_roberta`), continue.

Copy `javascript/train.code` to `~/.ncc/augmented_javascript/codebert/traverse_roberta/raw/javascript`
- For Yao & Yang
```
mkdir -p ~/.ncc/augmented_javascript/codebert/traverse_roberta/raw/javascript
cp ~/.ncc/augmented_javascript/raw/javascript_augmented.json ~/.ncc/augmented_javascript/codebert/traverse_roberta/raw/javascript/train.code
```
- For Jian-Guo
```
mkdir -p /export/share/jianguo/scodebert/augmented_javascript/codebert/traverse_roberta/raw/javascript
cp /export/share/jianguo/scodebert/augmented_javascript/raw/javascript_augmented.json /export/share/jianguo/scodebert/augmented_javascript/codebert/traverse_roberta/raw/javascript/train.code
```

## Step 1: build Tree-Sitter file
Build `javascript.so`
```
python -m dataset.augmented_javascript.build
```

## Step 2: generate ```ast``` of codes 
Extract features
- For Yao & Yang
```
python -m dataset.csn.feature_extract -l javascript -f ~/.ncc/augmented_javascript/codebert/traverse_roberta/raw -r ~/.ncc/augmented_javascript/codebert/traverse_roberta/refine -s ~/.ncc/augmented_javascript/libs -a code raw_ast ast -c 56
```
- For Jian-Guo
```
python -m dataset.csn.feature_extract -l javascript -f /export/share/jianguo/scodebert/augmented_javascript/codebert/traverse_roberta/raw -r /export/share/jianguo/scodebert/augmented_javascript/codebert/traverse_roberta/refine -s /export/share/jianguo/scodebert/augmented_javascript/libs -a code raw_ast ast -c 56
```

## Step 3: filter lines of None at ```code/ast``` files 
Filter out those cannot generate AST
- For Yao & Yang
```
python -m dataset.csn.filter -l javascript -r ~/.ncc/augmented_javascript/codebert/traverse_roberta/refine -f ~/.ncc/augmented_javascript/codebert/traverse_roberta/filter -a code ast
```
- For Jian-Guo
```
python -m dataset.csn.filter -l javascript -r /export/share/jianguo/scodebert/augmented_javascript/codebert/traverse_roberta/refine -f /export/share/jianguo/scodebert/augmented_javascript/codebert/traverse_roberta/filter -a code ast
```

# Step 4: get AST non-leaf node types and save them at ```.ast.node_types```
```
python -m dataset.augmented_javascript.traversal.get_type_node
```

# Step 5: get AST leaf node tokens and save them at ```*.ast.leaf_token```
```
python -m dataset.augmented_javascript.traversal.get_ast_leaf_token
```

# Step 6: run sentencepiece on ```*.ast.leaf_token``` with AST non-leaf node types```.ast.node_types```
- For Yao & Yang
```
python -m dataset.augmented_javascript.traversal.run_sentencepiece

cd ~/.ncc/augmented_javascript/codebert/traverse_roberta/filter/javascript/data-mmap
cut -f1 traversal.vocab | tail -n +10 | sed "s/$/ 100/g" > traversal.dict.txt
```
- For Jian-Guo

# Step 7: binarize traversal dataset
```
python -m dataset.augmented_javascript.traversal.preprocess -f preprocess.traversal
```