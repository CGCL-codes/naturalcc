from ncc.trainer.base_trainer import BaseTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, \
    TrainingArguments, DataCollatorForSeq2Seq
from omegaconf import OmegaConf
from ncc.utils.common.utils import get_abs_path
from peft import get_peft_model, LoraConfig, TaskType, AdaLoraConfig
from datasets import load_dataset
class InstructionTrainer(BaseTrainer):

    DEFAULT_INSTRUCTION_HYPERPARAMETERS_PATH = "configs/training/instruction_finetune.yaml"
    
    def __init__(self, train_dataset, validation_dataset=None, 
                checkpoints_path="./checkpoints", pretrained_model_or_path=None, 
                evaluation_fn=None, training_args=None, evaluator=None, peft=None,
                with_CoT=False, with_CoT_normal=False, with_CoT_simple=False, with_CoT_complex=False):
        
        instruction_parameters_config = OmegaConf.load(get_abs_path(self.DEFAULT_INSTRUCTION_HYPERPARAMETERS_PATH)).hyperparameters
        # Load dataset
        train_dataset = load_dataset('json', data_files=train_dataset, split='train')

        # Load model
        model_path = instruction_parameters_config["model_path"]
        print(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Tokenization
        tokenizer.add_eos_token = True
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"

        def tokenize(prompt):
            result = tokenizer(
                prompt,
                truncation=True,
                max_length=instruction_parameters_config["max_length"],
                padding=False,
                return_tensors=None,
            )

            result["labels"] = result["input_ids"].copy()

            return result

        def generate_and_tokenize_prompt(data_point, with_CoT=with_CoT, with_CoT_normal = with_CoT_normal, with_CoT_simple = with_CoT_simple, with_CoT_complex = with_CoT_complex):
            if with_CoT:
                # full_prompt = f"""
                #     You are very good at writing code. As long as you are provided with a problem, you can tell us how to solve the problem and finish the code pretty well.

                #     ### Probelm:
                #     {data_point["instruction"]}. Let's think step by step and then solve the problem.

                #     ### Solve Idea:
                #     {data_point["CoT"]}

                #     ### Response:
                #     {data_point["output"]}
                #     """
                full_prompt = f"""
                    You are very good at writing code. As long as you are provided with a problem, you can tell us how to solve the problem and finish the code pretty well.

                    ### Probelm:
                    {data_point["instruction"]}. Let's think step by step.

                    ### Solve Idea:
                    {data_point["CoT"]}

                    ### Response:
                    {data_point["output"]}
                    """
            elif with_CoT_normal:
                full_prompt = f"""
                    You are very good at writing code. As long as you are provided with a problem, you can tell us how to solve the problem and finish the code pretty well.

                    ### Probelm:
                    {data_point["instruction"]}. Let's think step by step.

                    ### Solve Idea:
                    {data_point["CoT_normal"]}

                    ### Response:
                    {data_point["output"]}
                    """
            elif with_CoT_simple:
                full_prompt = f"""
                    You are very good at writing code. As long as you are provided with a problem, you can tell us how to solve the problem and finish the code pretty well.

                    ### Probelm:
                    {data_point["instruction"]}. Let's think step by step.

                    ### Solve Idea:
                    {data_point["CoT_simple"]}

                    ### Response:
                    {data_point["output"]}
                    """
            elif with_CoT_complex:
                full_prompt = f"""
                    You are very good at writing code. As long as you are provided with a problem, you can tell us how to solve the problem and finish the code pretty well.

                    ### Probelm:
                    {data_point["instruction"]}. Let's think step by step.

                    ### Solve Idea:
                    {data_point["CoT_complex"]}

                    ### Response:
                    {data_point["output"]}
                    """
            else:
                full_prompt = f"""
                    You are very good at writing code. As long as you are provided with a problem, you can finish the code pretty well.
        
                    ### Instruction:
                    {data_point["instruction"]}
        
                    ### Response:
                    {data_point["output"]}
                    """
            return tokenize(full_prompt)

        tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)

        data_collator=DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            )
        super().__init__(model, tokenizer, tokenized_train_dataset, validation_dataset,
                        checkpoints_path, model_path, evaluator, evaluation_fn,data_collator)
            
        if training_args is None:
            self.training_args = TrainingArguments(
                per_device_train_batch_size=instruction_parameters_config["per_device_train_batch_size"],
                gradient_accumulation_steps=instruction_parameters_config["gradient_accumulation_steps"],
                warmup_steps=instruction_parameters_config["warm_up"],
                # max_steps=300,
                learning_rate=instruction_parameters_config["learning_rate"],
                # bf16=True,
                # optim="adamw_torch",
                # evaluation_strategy="steps",
                save_strategy="steps",
                # eval_steps=20,
                save_steps=1500,
                output_dir=instruction_parameters_config["output_dir"],
                save_total_limit=30,
                # load_best_model_at_end=True,
                group_by_length=True,  # group sequences of roughly the same length together to speed up training
                num_train_epochs=instruction_parameters_config["train_epochs"],
                seed=42,
                resume_from_checkpoint=instruction_parameters_config["checkpoint"],
            )
        else:
            self.training_args = training_args

        
        self.trainer = self.init_trainer()
        
        if peft:
            self.peft = peft
            # self.model = prepare_model_for_int8_training(self.model)
            peft_config = self.get_default_peft_config(peft)
            self.model.enable_input_require_grads()
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()