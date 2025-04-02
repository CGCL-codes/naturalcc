from ncc2.tasks.generation import GenerationTask

if __name__ == '__main__':
    
    ckpt_path = '~/checkpoints/CodeLlama-7b'
    dataset_path = './dataset.json'
    output_path = './result.json'
    test_input = ['this is a test']
    
task_name = "codellama_7b_generation"
device = "cuda:0"
    
# init task by task_name
task = GenerationTask(task_name=task_name,
                      device=device)

# set tokenizer and load model weights
task.from_pretrained(ckpt_path)

# load dataset
task.load_dataset(dataset_path)

# set config and run
task.run(output_path=output_path,max_length=50)


