import argparse
import torch
from transformers import AutoConfig, AutoModel, RobertaConfig
import os
# from utils_for_plbart import build_mbart

def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_pretrained_model(model_path):
    return AutoModel.from_pretrained(model_path)

def get_random_embeds(config):
    random_model = AutoModel.from_config(config)
    random_embeds = random_model.get_input_embeddings().weight
    return random_embeds

def get_mapping_matrices(mapping_file, new_vocab_size, old_vocab_size, embedding_dim=768, use_one_to_one=False, use_random_unk=False):
    mapping_matrix = torch.zeros((new_vocab_size, old_vocab_size))
    mask_matrix = torch.zeros((new_vocab_size, embedding_dim))

    with open(mapping_file) as f:
        for line in f:
            fields = line.rstrip().split('\t')

            if len(fields) < 3:
                fields.append('0')
                fields.append('<unk>')

            # only transfer wholesome old token embeds if use_one_to_one
            if use_one_to_one and ',' in fields[2]:
                continue

            # use random embeddings for unk
            if use_random_unk and fields[-1]=='<unk>' and fields[1]!='<unk>':
                continue

            new_idx = int(fields[0])
            mask_matrix[new_idx, :] = 1

            old_ids = fields[2].split(',')
            old_ids = [int(idx) for idx in old_ids]

            denominator = len(old_ids)
            for idx in old_ids:
                mapping_matrix[new_idx, idx] += 1. / denominator

    return (mapping_matrix, mask_matrix)

def modify_model(args):
    # conf = RobertaConfig()
    # conf.vocab_size = new_vocab_size
    # conf.num_hidden_layers = 12
    # conf = AutoConfig(args.source_model)
    conf = AutoConfig.from_pretrained(args.model_path)
    conf.vocab_size = args.new_vocab_size
    random_embeds = get_random_embeds(conf)
    
    model = get_pretrained_model(args.model_path)
    mapping_matrix, mask_matrix = get_mapping_matrices(args.mapping_file,
                                                        args.new_vocab_size,
                                                        args.old_vocab_size,
                                                        use_one_to_one=args.use_one_to_one,
                                                        use_random_unk=args.use_random_unk)
    
    new_embeds = mapping_matrix.matmul(model.get_input_embeddings().weight)
    new_embed_matrix = (1. - mask_matrix) * random_embeds + new_embeds
    model.set_input_embeddings(torch.nn.Embedding.from_pretrained(new_embed_matrix, freeze=False))
    # model.save_pretrained(args.save_path + name)
    model.save_pretrained(args.save_path)

def modify_plbart_model(args):
    config = argparse.Namespace(activation_fn='gelu', adam_betas='(0.9, 0.98)', adam_eps=1e-08, adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0, add_prev_output_tokens=True, all_gather_list_size=16384, arch='mbart_base', attention_dropout=0.1, batch_size=4, batch_size_valid=4, best_checkpoint_metric='accuracy', bf16=False, bpe=None, broadcast_buffers=False, bucket_cap_mb=25, checkpoint_shard_count=1, checkpoint_suffix='', classification_head_name='sentence_classification_head', clip_norm=1.0, cpu=False, criterion='sentence_prediction', cross_self_attention=False, curriculum=0, data='/home/zzr/CodeStudy/Defect-detection/plbart/processed/data-bin', data_buffer_size=10, dataset_impl=None, ddp_backend='no_c10d', decoder_attention_heads=12, decoder_embed_dim=768, decoder_embed_path=None, decoder_ffn_embed_dim=3072, decoder_input_dim=768, decoder_layerdrop=0, decoder_layers=6, decoder_layers_to_keep=None, decoder_learned_pos=True, decoder_normalize_before=False, decoder_output_dim=768, device_id=0, disable_validation=False, distributed_backend='nccl', distributed_init_method=None, distributed_no_spawn=False, distributed_num_procs=0, distributed_port=-1, distributed_rank=0, distributed_world_size=1, distributed_wrapper='DDP', dropout=0.1, empty_cache_freq=0, encoder_attention_heads=12, encoder_embed_dim=768, encoder_embed_path=None, encoder_ffn_embed_dim=3072, encoder_layerdrop=0, encoder_layers=6, encoder_layers_to_keep=None, encoder_learned_pos=True, encoder_normalize_before=False, end_learning_rate=0.0, fast_stat_sync=False, find_unused_parameters=True, finetune_from_model=None, fix_batches_to_gpus=False, fixed_validation_seed=None, force_anneal=None, fp16=False, fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, gen_subset='test', init_token=0, keep_best_checkpoints=-1, keep_interval_updates=-1, keep_last_epochs=-1, langs='java,python,en_XX', layernorm_embedding=True, local_rank=0, localsgd_frequency=3, log_format='json', log_interval=10, lr=[5e-05], lr_scheduler='polynomial_decay', max_epoch=5, max_positions=512, max_source_positions=1024, max_target_positions=1024, max_tokens=2048, max_tokens_valid=2048, max_update=15000, maximize_best_checkpoint_metric=True, memory_efficient_bf16=False, memory_efficient_fp16=False, min_loss_scale=0.0001, min_lr=-1.0, model_parallel_size=1, no_cross_attention=False, no_epoch_checkpoints=True, no_last_checkpoints=False, no_progress_bar=False, no_save=False, no_save_optimizer_state=False, no_scale_embedding=False, no_seed_provided=False, no_shuffle=False, no_token_positional_embeddings=False, nprocs_per_node=1, num_classes=2, num_shards=1, num_workers=1, optimizer='adam', optimizer_overrides='{}', patience=-1, pipeline_balance=None, pipeline_checkpoint='never', pipeline_chunks=0, pipeline_decoder_balance=None, pipeline_decoder_devices=None, pipeline_devices=None, pipeline_encoder_balance=None, pipeline_encoder_devices=None, pipeline_model_parallel=False, pooler_activation_fn='tanh', pooler_dropout=0.0, power=1.0, profile=False, quant_noise_pq=0, quant_noise_pq_block_size=8, quant_noise_scalar=0, quantization_config_path=None, regression_target=False, relu_dropout=0.0, required_batch_size_multiple=1, required_seq_len_multiple=1, reset_dataloader=True, reset_lr_scheduler=False, reset_meters=True, reset_optimizer=True, restore_file='/data2/cg/CodeStudy/PLBART/pretrain/checkpoint_11_100000.pt', save_dir='/home/zzr/CodeStudy/Defect-detection/plbart/devign', save_interval=1, save_interval_updates=0, scoring='bleu', seed=1234, sentence_avg=False, separator_token=None, shard_id=0, share_all_embeddings=True, share_decoder_input_output_embed=True, shorten_data_split_list='', shorten_method='truncate', skip_invalid_size_inputs_valid_test=False, slowmo_algorithm='LocalSGD', slowmo_momentum=None, stop_time_hours=0, task='plbart_sentence_prediction', tensorboard_logdir=None, threshold_loss_scale=None, tokenizer=None, total_num_update=1000000, tpu=False, train_subset='train', update_freq=[4], use_bmuf=False, use_old_adam=False, user_dir='/home/zzr/CodeStudy/PLBART/source', valid_subset='valid', validate_after_updates=0, validate_interval=1, validate_interval_updates=0, warmup_updates=500, weight_decay=0.0, zero_sharding='none')
    config.num_labels = 1
    config.vocab_size = 50005
    config.pad_id = 1

    model = build_mbart(config)
    random_embeds = model.encoder.embed_tokens.weight

    sd = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(sd["model"])

    mapping_matrix, mask_matrix = get_mapping_matrices(args.mapping_file,
                                                        args.new_vocab_size,
                                                        args.old_vocab_size,
                                                        use_one_to_one=args.use_one_to_one,
                                                        use_random_unk=args.use_random_unk)
    
    new_embeds = mapping_matrix.matmul(model.encoder.embed_tokens.weight)
    new_embed_matrix = (1. - mask_matrix) * random_embeds + new_embeds
    # model.set_input_embeddings(torch.nn.Embedding.from_pretrained(new_embed_matrix, freeze=False))
    model.encoder.embed_tokens = torch.nn.Embedding.from_pretrained(new_embed_matrix, freeze=False)

    output_model_file = os.path.join(args.save_path, "pytorch_model.pt")
    torch.save(model.state_dict(), output_model_file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapping_file', type=str, default=None)
    parser.add_argument('--source_model', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--new_vocab_size', type=int)
    parser.add_argument('--old_vocab_size', type=int)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--use_one_to_one', type=bool, default=False)
    parser.add_argument('--use_random_unk', type=bool, default=False)
    parser.add_argument('--model', type=str, default=None, help='codebert or codet5')
    parser.add_argument('--transfer_type', type=str, default=None, help='tokenizer, morphology or frequency')
    return parser.parse_args()

def main():
    global args
    args = parse_args()

    # # codebert
    if args.model == 'codebert':
        args.mapping_file = f'/data/sub3/Doo/TokenizationRevisiting/VocabularyTransfer/matches/{args.model}/{args.transfer_type}_match.tsv'
        args.source_model = 'microsoft/codebert-base'
        args.model_path = f'/data/sub3/Doo/TokenizationRevisiting/models/originalmodels/{args.model}'
        args.config = f'/data/sub3/Doo/TokenizationRevisiting/models/originalmodels/{args.model}'
        args.new_vocab_size = 50265
        args.old_vocab_size = 50265
        # args.save_path = f'/data/sub3/Doo/TokenizationRevisiting/models/transferredmodels/{args.model}/{args.transfer_type}_based'
        args.save_path = f'/data/sub3/Doo/TokenizationRevisiting/models/transferredmodels/{args.model}/concode/{args.transfer_type}_based'
        check_dir(args.save_path)
        modify_model(args)

    # # codet5
    if args.model == 'codet5':
        args.mapping_file=f'/data/sub3/Doo/TokenizationRevisiting/VocabularyTransfer/matches/{args.model}/{args.transfer_type}_match.tsv'
        args.source_model="Salesforce/codet5-base"
        args.model_path=f"/data/sub3/Doo/TokenizationRevisiting/models/originalmodels/{args.model}-base"
        args.config=f"/data/sub3/Doo/TokenizationRevisiting/models/originalmodels/{args.model}-base"
        args.new_vocab_size=32100
        args.old_vocab_size=32100
        # args.save_path=f"/data/sub3/Doo/TokenizationRevisiting/models/transferredmodels/{args.model}/{args.transfer_type}_based"
        args.save_path=f"/data/sub3/Doo/TokenizationRevisiting/models/transferredmodels/{args.model}/concode/{args.transfer_type}_based"
        check_dir(args.save_path)
        modify_model(args)

    # ## plbart
    # args.mapping_file = '/data/sub3/Doo/TokenizationRevisiting/VocabularyTransfer/matches/plbart/tokenizer_match.tsv'
    # args.source_model = 'uclanlp/plbart-base'
    # args.model_path = '/data/sub3/Doo/TokenizationRevisiting/models/originalmodels/plbart/checkpoint_11_100000.pt'
    # args.config = '/data/sub3/Doo/TokenizationRevisiting/models/originalmodels/plbart/config.json'
    # args.new_vocab_size = 50005
    # args.old_vocab_size = 50005
    # args.save_path = '/data/sub3/Doo/TokenizationRevisiting/models/transferedmodels/plbart/tokenizer_avg'

    # check_dir(args.save_path)
    # modify_plbart_model(args)

if __name__ == '__main__':
    main()
