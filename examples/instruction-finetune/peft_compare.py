from ncc.trainer.instruction_trainer import InstructionTrainer

trainer = InstructionTrainer(
    train_dataset="/home/wanyao/wucai/work/data/code_instructions_filtered_CoT.jsonl",
    peft="adalora"
)
trainer.train()

trainer = InstructionTrainer(
    train_dataset="/home/wanyao/wucai/work/data/code_instructions_filtered_CoT.jsonl",
    peft="lora"
)
trainer.train()

trainer = InstructionTrainer(
    train_dataset="/home/wanyao/wucai/work/data/code_instructions_filtered_CoT.jsonl",
    peft="prefix"
)
trainer.train()

trainer = InstructionTrainer(
    train_dataset="/home/wanyao/wucai/work/data/code_instructions_filtered_CoT.jsonl",
    peft="p_tuning"
)
trainer.train()

trainer = InstructionTrainer(
    train_dataset="/home/wanyao/wucai/work/data/code_instructions_filtered_CoT.jsonl",
    peft="prompt"
)
trainer.train()