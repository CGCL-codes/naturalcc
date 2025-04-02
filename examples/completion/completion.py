from ncc2.tasks.completion import CompletionTask

if __name__ == '__main__':
    
    ckpt_path = '~/checkpoints/CodeLlama-7b'
    dataset_path = './dataset.json'
    output_path = './result.json'
    test_input = ['this is a test']
    
    # init task by task_name
    print('Initializing CompletionTask')
    task = CompletionTask(task_name="llm",device="cuda:6")

    # set tokenizer and load model weights
    print('Loading model weights [{}]'.format(ckpt_path))
    task.from_pretrained(ckpt_path)

    # load dataset and run task
    print('Processing dataset [{}]'.format(dataset_path))
    task.load_dataset(dataset_path)
    task.run(output_path=output_path,batch_size=1,max_length=50)
    print('Output file: {}'.format(output_path))