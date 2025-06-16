from ncc.trainer.instruction_trainer import InstructionTrainer

#比较指令数据格式的影响，，包括有无注释信息，思维链信息，思维链复杂度的影响。
# 配置信息在configs/training/instruction_finetune.yaml、configs/training/peft.yaml中

#指令微调标准种子数据集
trainer = InstructionTrainer(
    train_dataset="/home/wanyao/wucai/work/data/code_instructions_filtered_CoT.jsonl",
    peft="lora"
)
trainer.train()

#指令微调去除注释信息的标准种子数据集
trainer = InstructionTrainer(
    train_dataset="/home/wanyao/wucai/work/data/code_instructions_filtered_CoT_comment_remove.jsonl",
    peft="lora"
)
trainer.train()

#指令微调带有思维链信息的标准种子数据集
trainer = InstructionTrainer(
    train_dataset="/home/wanyao/wucai/work/data/code_instructions_filtered_CoT.jsonl",
    peft="lora",
    with_CoT=True
)
trainer.train()

#指令微调简单复杂度的思维链数据集
trainer = InstructionTrainer(
    train_dataset="/home/wanyao/wucai/work/data/code_instructions_filtered_CoT_complex_final.jsonl",
    peft="lora",
    with_CoT_simple=True
)
trainer.train()

#指令微调正常复杂度的思维链数据集
trainer = InstructionTrainer(
    train_dataset="/home/wanyao/wucai/work/data/code_instructions_filtered_CoT_complex_final.jsonl",
    peft="lora",
    with_CoT_normal=True
)
trainer.train()

#指令微调复杂复杂度的思维链数据集
trainer = InstructionTrainer(
    train_dataset="/home/wanyao/wucai/work/data/code_instructions_filtered_CoT_complex_final.jsonl",
    peft="lora",
    with_CoT_complex=True
)
trainer.train()
