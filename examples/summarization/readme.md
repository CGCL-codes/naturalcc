# SummarizationTask
## Step1: Download the checkpoint
Recommend Model: [Codellama-7B-Instruct](https://huggingface.co/camenduru/CodeLlama-7b-Instruct)

## Step2: Prepare your testing dataset
For example:
```json
[
    {
        "input": "current_directory = os.path.dirname(os.path.abspath(__file__))"v
    },
    {
        "input": "import os\nos.environ['http_proxy'] = 'http://127.0.0.1:7890'\nos.environ['https_proxy'] = 'http://127.0.0.1:7890'"
    },
    {
        "input": "img.save('pad_img.png')\n mask.save('pad_mask.png')"
    },
    {
        "input": "def __init__(self, device='cuda:0'):\n\tself.device = device"
    },
    {
        "input": "def start_loop(loop):\n\tasyncio.set_event_loop(loop)\n\tloop.run_forever()"
    },
    {
        "input": "self.lock.acquire()\nself.idx = (self.idx+1)%len(GPT_KEY_POOL)\nself.lock.release()"
    }
]
```

## Step3: Running scripts
- Instantiate the task class and set device
```python
print('Initializing SummarizationTask')
    task = SummarizationTask(task_name="llm",device="cuda:6")
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
task.run(output_path=output_path,batch_size=1,max_length=100)
    print('Output file: {}'.format(output_path))
```
