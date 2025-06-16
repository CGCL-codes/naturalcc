from ncc.trainer.instruction_trainer import InstructionTrainer

trainer = InstructionTrainer(
    train_dataset="/home/wanyao/wucai/work/data/code_instructions_filtered_CoT.jsonl",
    peft="lora"
)
trainer.train()

trainer = InstructionTrainer(
    train_dataset="/home/wanyao/wucai/work/data/code_instructions_filtered_CoT_comment_remove.jsonl",
    peft="lora"
)
trainer.train()

trainer = InstructionTrainer(
    train_dataset="/home/wanyao/wucai/work/data/code_instructions_filtered_CoT.jsonl",
    peft="lora",
    with_CoT=True
)
trainer.train()

trainer = InstructionTrainer(
    train_dataset="/home/wanyao/wucai/work/data/code_instructions_filtered_CoT.jsonl",
    peft="lora",
    with_CoT_simple=True
)
trainer.train()

trainer = InstructionTrainer(
    train_dataset="/home/wanyao/wucai/work/data/code_instructions_filtered_CoT.jsonl",
    peft="lora",
    with_CoT_normal=True
)
trainer.train()

trainer = InstructionTrainer(
    train_dataset="/home/wanyao/wucai/work/data/code_instructions_filtered_CoT.jsonl",
    peft="lora",
    with_CoT_complex=True
)
trainer.train()
