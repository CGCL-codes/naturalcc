# model=codebert
model=codet5
# transfer_type=tokenizer
# transfer_type=morphology
transfer_type=frequency

python transfer.py --model $model --transfer_type $transfer_type

