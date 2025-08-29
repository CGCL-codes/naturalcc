from codebleu import calc_codebleu

# # RQ2
# path = "/data/sub3/Doo/TokenizationRevisiting/Task/Code-Generation/codebert/saved_models/batchsize16_epoch30/test_RQ2"
# train_types= ["test_nl_tocontaminate", "test_nl_select", "test_nl_random"]

# RQ3
# path = "/data/sub3/Doo/TokenizationRevisiting/Task/Code-Generation/codebert/saved_models_mapping_RQ3"
# train_types= ["exchange_batchsize48_epoch60", "overuse-allaffix_batchsize48_epoch60", "overuse-noaffix_batchsize48_epoch60"]

# RQ4
path = "/data/sub3/Doo/TokenizationRevisiting/Task/Code-Generation/codebert/saved_models_transferred_RQ4_failed"
train_types= ["tokenizer_avg_batchsize16_epoch60", "tokenizer_one_batchsize16_epoch50", "mix_avg_batchsize16_epoch60"]

for train_type in train_types:
    print(train_type)
    references_path = f"{path}/{train_type}/test_-1.gold"
    predictions_path = f"{path}/{train_type}/test_-1.output"
    references = [x.strip() for x in open(references_path, 'r', encoding='utf-8').readlines()]
    predictions = [x.strip() for x in open(predictions_path, 'r', encoding='utf-8').readlines()]
    result = calc_codebleu(references, predictions, lang="cpp", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
    print(result)
    print()