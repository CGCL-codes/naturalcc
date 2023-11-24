from ncc2.models.llama import LLaMATokenizer
from ncc2.tasks.generation import GenerationTask

if __name__ == '__main__':
    
    ckpt_path = '/data/lz/models/CodeLlama-7b'
    dataset_path = './dataset.json'
    output_path = './result.json'
    test_input = ['this is a test']
    
    # init task by task_name
    print('Initializing GenerationTask')
    task = GenerationTask(task_name="codellama_7b_code",device="cuda:2")

    # set tokenizer and load model weights
    print('Loading model weights [{}]'.format(ckpt_path))
    task.from_pretrained(ckpt_path)

    # load dataset and run task
    print('Processing dataset [{}]'.format(dataset_path))
    task.load_dataset(dataset_path)
    task.run(output_path=output_path,batch_size=1,max_length=50,temperature=0.5,top_p=0.99)
    print('Output file: {}'.format(output_path))
    
    # run test
    print('Running generation test')
    test_output = task.generate(test_input,max_length=30,top_k=20,top_p=0.99,penalty_weight=0.8)
    print('Test input: {}\nTest output: {}'.format(test_input,test_output))