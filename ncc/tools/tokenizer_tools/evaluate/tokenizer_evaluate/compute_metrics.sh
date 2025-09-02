# model=codebert
# model=codet5-base
# model=CodeLlama-7b-hf
# model=CodeGPT-small-java-adaptedGPT2
# model=WizardCoder-1B-V1.0

dataset_type=train
# dataset_type=validation

# for model in codebert codet5-base WizardCoder-1B-V1.0 CodeLlama-7b-hf deepseek-coder-6.7b-instruct plbart
# for model in deepseek-coder-6.7b-instruct
# for model in WizardCoder-1B-V1.0
# for model in codebert codet5-base
# for model in plbart WizardCoder-1B-V1.0
# for model in CodeLlama-7b-hf deepseek-coder-6.7b-instruct
for model in deepseek-coder-6.7b-instruct
do
data_file1=/data/sub3/Doo/datasets/CodeSearchNet/atxtfile/${dataset_type}_code_string.txt
data_file2=/data/sub3/Doo/datasets/CodeSearchNet/atxtfile/${dataset_type}_documentation_string.txt
out_dir=/data/sub3/Doo/TokenizationRevisiting/Tokenizer-Evaluate/results/$model
tokenizer_path=/home/wanyao/wangchen/models/$model
# tokenizer_path=/data/sub3/Doo/TokenizationRevisiting/models/transferedmodels/codebert/mix_avg

python metrics.py \
    -d $data_file1 $data_file2 \
    --out_dir $out_dir \
    --tokenizer_path $tokenizer_path \
    --name ${dataset_type}_token_frequencies \
    2>&1 | tee $out_dir/metrics_${dataset_type}.log
done