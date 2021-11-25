# StackOverflow Dataset Generation
*SQL parser is built and only functions in Python2 env.
However, since NaturalCC is designed on Python3, we have processed SQL/C#/Python data in a Python2 based environment. 
and saved them in [stack_overflow.zip](dataset/stack_overflow/stack_overflow.zip).
If interested in the data processing, you can follow original [stack_overflow](https://github.com/sriniiyer/stack_overflow).*

### Step 1. Download StackOverflow C#/SQL/Python datasets
```shell script
bash dataset/stack_overflow/download.sh
```

### Step 2. ```SQL``` Generation
1) flatten SQL code/docstring at ```~/stack_overflow/flatten/sql```
```shell script
python -m dataset.stack_overflow.flatten -l sql
```
2) move those decompressed files to ```~/stack_overflow/flatten/sql```
```shell script
unzip dataset/stack_overflow/sql_tokens.zip -d ~/stack_overflow/flatten/sql
``` 
3) binarize SQL dataset
```shell script
python -m dataset.stack_overflow.summarization.preprocess -f config/sql
```

### Step 3. ```C#``` Generation
1) install antlr4-python3-runtime
```shell script
pip install antlr4-python3-runtime==4.5.2
```
2) flatten C# code/docstring at ~/stack_overflow/flatten/csharp
```shell script
python -m dataset.stack_overflow.flatten -l csharp
```
3) tokenize ```code/docstring``` into ```code_token/docstring_token```
```shell script
python -m dataset.stack_overflow.tokenization -l csharp
```
*Since generating code_token/docstring_token is slow, you can move those decompressed files to ~/stack_overflow/flatten/csharp*
```shell script
unzip dataset/stack_overflow/csharp_tokens.zip -d ~/stack_overflow/flatten/csharp
``` 
4) binarize C# dataset
```shell script
python -m dataset.stack_overflow.summarization.preprocess -f config/csharp
```


### Step 4. ```Python``` Generation
1) flatten Python code/docstring at ~/stack_overflow/flatten/python
```shell script
python -m dataset.stack_overflow.flatten -l python
```
2) tokenize ```code/docstring``` into ```code_token/docstring_token```
```shell script
python -m dataset.stack_overflow.tokenization -l python
```
3) binarize Python dataset
```shell script
python -m dataset.stackoverflow.summarization.preprocess -f config/python
```