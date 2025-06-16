from ncc.trainer.instruction_trainer import InstructionTrainer
#比较高效微调方法的影响，配置信息在configs/training/peft.yaml中

#使用lora方法指令微调标准种子数据集
trainer = InstructionTrainer(
    train_dataset="/home/wanyao/wucai/work/data/code_instructions_filtered_CoT.jsonl",
    peft="lora"
)
trainer.train()

#使用adalora方法指令微调标准种子数据集
trainer = InstructionTrainer(
    train_dataset="/home/wanyao/wucai/work/data/code_instructions_filtered_CoT.jsonl",
    peft="adalora"
)
trainer.train()

#使用prefix方法指令微调标准种子数据集
trainer = InstructionTrainer(
    train_dataset="/home/wanyao/wucai/work/data/code_instructions_filtered_CoT.jsonl",
    peft="prefix"
)
trainer.train()

#使用p_tuning方法指令微调标准种子数据集
trainer = InstructionTrainer(
    train_dataset="/home/wanyao/wucai/work/data/code_instructions_filtered_CoT.jsonl",
    peft="p_tuning"
)
trainer.train()

#使用prompt方法指令微调标准种子数据集
trainer = InstructionTrainer(
    train_dataset="/home/wanyao/wucai/work/data/code_instructions_filtered_CoT.jsonl",
    peft="prompt"
)
trainer.train()