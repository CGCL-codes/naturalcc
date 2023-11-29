# CompletionTask
## Step1: Download the checkpoint
Recommend Model: [Codellama-7B](https://huggingface.co/camenduru/CodeLlama-7b)

## Step2: Prepare your testing dataset
For example:
```json
[
    {
        "input": "this is a"
    },
    {
        "input": "from tqdm import"
    },
    {
        "input": "def calculate("
    },
    {
        "input": "a = b**2"
    },
    {
        "input": "torch.randint"
    },
    {
        "input": "x = [1,2"
    }
]
```

## Step3: Running scripts
- Instantiate the task class and set device
```python
print('Initializing CompletionTask')
    task = SummarizationTask(task_name="llm",device="cuda:2")
```
- Load the checkpoint
```python
print('Loading model weights [{}]'.format(ckpt_path))
    task.from_pretrained(ckpt_path)
```
- Load the dataset from given path
```python
print('Processing dataset [{}]'.format(dataset_path))
    task.load_dataset(dataset_path)
```
- Run the model and output to the specified path
```python
task.run(output_path=output_path,batch_size=1,max_length=50)
    print('Output file: {}'.format(output_path))
```
