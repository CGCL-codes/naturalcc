import task_configuration
from evaluate import Experiment
import argparse
import methods
# Set up the methods to use.
# To define your own method, inherit methods.TransferabilityMethod. See the methods in methods.py for more details.

# experiment = Experiment(task_configuration.selection_methods, name='test', append=False,
# 						budget=task_configuration.budget,
# 						runs=task_configuration.n_runs) # Set up an experiment with those methods named "test".
#                                                                # Append=True skips evaluations that already happend. Setting it to False will overwrite.
# # experiment.download_models()
# experiment.run()                                               # Run the experiment and save results to ./results/{name}.csv




def parse_args():
    parser = argparse.ArgumentParser("Which model to use")
    parser.add_argument('--task', type=str, default='codexglue_defect_detection',
                        help='task')
    parser.add_argument('--models', type=str, default='microsoft/codebert-base',
                        help='models')
    parser.add_argument('--selection_method', type=str, default='Logistic')
    parser.add_argument('--measurements', type=str, default='Measurement Criterion')
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--budget', type=int, default=1000)
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    task_configuration.variables['Target Dataset'] = [args.task]
    task_configuration.variables['Model'] = args.models.split(',')
    task_configuration.n_runs = args.n_runs
    task_configuration.budget = args.budget
    task_configuration.selection_methods = {args.selection_method: task_configuration.method_map[args.selection_method]}
    experiment = Experiment(task_configuration.selection_methods, name='selection', append=False,
    						budget=task_configuration.budget,
    						runs=task_configuration.n_runs)
    experiment.run()                                               # Run the experiment and save results to ./results/{name}.csv




