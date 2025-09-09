import argparse
import pytorch_lightning as pl
from model import *
from transformers import logging as hf_logging

# Set the transformers logging to error only to suppress warnings
hf_logging.set_verbosity_error()


if __name__ == '__main__':
    # Parsing Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_set", default="", type=str,
                        help="Path of validation set.")
    parser.add_argument("--model_name_or_path", default="codeparrot/codeparrot", type=str,
                        help="Model to train and evaluate, provide a repo name in Hugging Face hub or a local path.")
    parser.add_argument("--peft_model_name_or_path", default="", type=str,
                        help="PEFT model to train and evaluate, provide a repo name in Hugging Face hub or a local path.")
    parser.add_argument('--ngpu', type=int, default=4,
                        help="Number of GPUs to use.")
    parser.add_argument('--eval_batch_size', type=int, default=8,
                        help="Batch size for evaluation.")
    parser.add_argument("--check_validation_only", action='store_true',
                        help="Whether to only evaluate the model.")
    parser.add_argument('--el_n', type=int, default=10,
                        help="Measure the extraction likelihood of extracting n consecutive tokens.")
    parser.add_argument("--eval_secret", action='store_true',
                        help="Whether to evaluate the secret memorization.")

    parser.add_argument("--unlearning_mode", default="GA", type=str,
                        help="Mode to unlearn.")
    parser.add_argument("--train_set", default="", type=str,
                        help="Path of forgot set.")
    parser.add_argument('--train_batch_size', type=int, default=4,
                        help="Batch size for training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of steps for accumulating gradients.")
    parser.add_argument("--learning_rate", default=3e-06, type=float,
                        help="The initial learning rate for training.")
    parser.add_argument("--ma_threshold", default=0, type=float,
                        help="Forgetting threshold of MA.")
    parser.add_argument("--el10_threshold", default=0, type=float,
                        help="Forgetting threshold of MA.")
    parser.add_argument("--margin", default=0.1, type=float,
                        help="")

    parser.add_argument("--control_set", default="", type=str,
                        help="Path of control set.")
    parser.add_argument("--control_alpha", default=1.0, type=float,
                        help="The hyper-parameter alpha.")
    parser.add_argument("--control_lambda", default=0.1, type=float,
                        help="The hyper-parameter lambda.")
    
    parser.add_argument("--select_gamma", default=0.1, type=float,
                        help="The hyper-parameter gamma.")
    
    parser.add_argument('--lora_r', type=int, default=32,
                        help="The rank of the update matrices. Lower rank results in smaller update matrices with fewer trainable parameters.")
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help="LoRA scaling factor.")
    parser.add_argument("--lora_dropout", default=0.1, type=float,
                        help="Dropout rate for LoRA.")

    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for initialization.")
    parser.add_argument('--num_workers', type=int, default=0,
                        help="Number of workers")
    parser.add_argument("--strategy", default="deepspeed_stage_2_offload", type=str,
                        help="Optimization strategy of DeepSpeed.")
    parser.add_argument("--fp16", action='store_true',
                        help="Whether to use fp16 model precision.")
    parser.add_argument('--num_train_epochs', type=int, default=100,
                        help="Maximum number of epochs for training.")
    parser.add_argument('--target_length', type=int, default=128,
                        help="Maximum length of generated sequence (prompt+generation).")
    parser.add_argument("--valid_save_path", default="", type=str,
                        help="File path for saving the validation results.")
    args = parser.parse_args()
    print(args)

    pl.seed_everything(args.seed, workers=True)

    # Setting for pytorch lightning trainer
    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        accelerator='gpu',
        devices=args.ngpu,
        max_epochs=args.num_train_epochs,
        precision=16 if args.fp16 else 32,
        check_val_every_n_epoch=1,
        enable_checkpointing=False,
        strategy=args.strategy,
        num_sanity_val_steps=0,
        limit_val_batches=1.0,
        log_every_n_steps=1,
        logger=False,
        deterministic=True
    )
    
    trainer = pl.Trainer(**train_params)
    
    if args.unlearning_mode == "GA":
        model = GradientAscent(args)
    elif args.unlearning_mode == "CU":
        model = ControllableUnlearning(args)
    elif args.unlearning_mode == "SSGA":
        model = SensitivitySelectiveGA(args)
    elif args.unlearning_mode == "SSCU":
        model = SensitivitySelectiveCU(args)
    elif args.unlearning_mode == "PECU":
        model = PEFTControllableUnlearning(args)
    elif args.unlearning_mode == "Original":
        model = UnlearningFramework(args)
    
    if args.check_validation_only:
        trainer.validate(model)
    else:
        trainer.fit(model)
