cd ../src

python main.py --run_prediction --num_dev 230 --model gpt-3.5-turbo --split python --method Dense --iterative_query verification_report --max_iteration 3 --num_shot 2 --with_dense_feedback
