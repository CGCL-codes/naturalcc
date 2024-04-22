import argparse
import os
import random

from util import py_tokenize


def load_py150_by_files(num_training, num_validation, base_dir, prefix, output_dir):
    """load py150 dataset by files
    
    Args:
        num_training: num of files for train a code completion model
        num_validation: num of validation files, 
                        finally, the dataset is splitted like: training dataset: [a, a+num_training]; test(non-member) dataset: [a+num_training, a+2*num_training]; validation_dataset: [a+2*num_training, a+2*num_training+num_validation]
        base_dir: the source file of the dataset
        prefix: if generating dataset for target model, the value of `prefix` is  'target', if for shadow model, is 'shadow_id'  
        output_dir: the output directory , in fact, is `output_dir` + `prefix`
    """
    output_dir = os.path.join(output_dir, prefix)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_files = num_training * 2 + num_validation  # number of files needed to use in one model, include member, non-member, validation
    
    all_file_paths = open(os.path.join(base_dir, "python100k_train.txt")).readlines()
    
    samples_file_paths = []
    
    # generate dataset for target model
    if "target" in prefix:
        samples_file_paths = all_file_paths[:num_files]
    
    # generate dataset for shadow model
    if "shadow" in prefix:
        samples_file_paths = all_file_paths[num_files: num_files*2]
        
        random.seed(int(prefix.split("_")[-1])) # for reproducing the experiments
        random.shuffle(samples_file_paths)
    
    # split the dataset
    train_paths = samples_file_paths[: num_training]
    test_paths = samples_file_paths[num_training: num_training*2] # test also means non-member
    valid_paths = samples_file_paths[-num_validation: ]
    
    for paths, filename in zip((train_paths, test_paths, valid_paths),
                                ("python_train.txt", "python_test.txt", "python_dev.txt")):
        with open(os.path.join(output_dir, filename), "w") as wf:
            for path in paths:
                wf.write(path)
    
    # tokenization
    py_tokenize(base_dir=base_dir, file_name=os.path.join(output_dir, "python_train.txt"),
                output_dir=os.path.join(output_dir, "code-gpt"), file_type="train")
    py_tokenize(base_dir=base_dir, file_name=os.path.join(output_dir, "python_test.txt"),
                output_dir=os.path.join(output_dir, "code-gpt"), file_type="test")
    py_tokenize(base_dir=base_dir, file_name=os.path.join(output_dir, "python_dev.txt"),
                output_dir=os.path.join(output_dir, "code-gpt"), file_type="dev")
    
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="/pathto/data/py150_files", type=str, 
                        help="The downloaded data path")
    parser.add_argument("--output_dir", default="/pathto/data/membership_inference", type=str, 
                        help="The output directory")
    
    
    parser.add_argument("-nshdow", "--number_of_shadow_model", default=10, type=int,
                        help="The number of shadow model")
    parser.add_argument("-ntrain", "--number_of_train_files", default=24000, type=int,
                        help="number of files for training a completion model, the number of non-member files is the same")
    parser.add_argument("-neval", "--number_of_valid_files", default=2000, type=int,
                        help="number of validation files")
    args = parser.parse_args()
    
    print("load dataset for target model")
    load_py150_by_files(num_training=args.number_of_train_files, num_validation=args.number_of_valid_files, 
                        base_dir=args.base_dir, prefix="target", output_dir=args.output_dir)
    
    for i in range(args.number_of_shadow_model):
        print("load dataset for shadow {} model".format(i))
        load_py150_by_files(num_training=args.number_of_train_files, num_validation=args.number_of_valid_files, 
                            base_dir=args.base_dir, prefix="shadow_{}".format(i), output_dir=args.output_dir)
    
    
if __name__ == "__main__":
    main()