data_process 文件夹包含了数据处理的逻辑
instruction_data_compare.py和peft_compare.py是可以运行的指令微调脚本
在h800上运行的步骤：
1.export PYTHONPATH="${PYTHONPATH}:/home/wanyao/wucai/naturalcc/ncc" 
或者export PYTHONPATH="${PYTHONPATH}:{你下载的natrualcc的ncc路径}"

2.conda activate ncc2
python {指令微调脚本路径}
即可开始微调