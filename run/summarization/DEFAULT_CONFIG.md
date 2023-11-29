# Motivation
Due to various hyper-parameters in different models, 
directly comparing those models' performance value from papers may cause injustice.
To assure those models share the same hyper-parameters. we re-implemented baselines of *Code Summarization* task and
trained them with the same config as possible as we can.

# Hyper-parameters
1. Dictionary:
    - Single-lingual dictionary: 
        - big token size: 50k # for code_tokens, BPE code_tokens
        - small token size: 30k # for natural language tokens
        - token frequency: 2 # a token must appear at least twice in files, otherwise it will not be absorbed in dictionaries
    - Multi-/Cross-lingual dictionary: 
        - big token size: 80k # for code_tokens, BPE code_tokens
        - small token size: 50k # for natural language tokens
        - token frequency: 4 # a token must appear at least twice in files; otherwise it will not be absorbed in dictionaries
    **```token size``` is not a dictionary's final size. 
    If tokens overwhelm our defined limitation, only Top-K frequentest tokens will be absorbed in dictionaries;
    otherwise, dictionaries will incorporate all tokens whose size is even less than our defined size.**

2. Truncation of Model Input:
    - (BPE) code_tokens: 512
    - (BPE) docstring_tokens: 30

3. Tokenization Methods:
    - Vanilla Tokenization: 1) space spliter + 2) ```dpu``` spliter + 3) ```str.tolow()``` for each token
    - BPE: ```sentencepiece``` lib.

4. Search Strategies
    - Greedy Search(= Beam Search with 1 candidate step):
        - fast but low performance
        - Greedy Search functions in validation/inference
    - Beam Search(> 1 candidate step, usually ```beam_size = 5```):
        - slow and GPU memory sourcing but high performance (about *< 1.0* performance enhancement in ```BLEU/Rouge-L/Metoer```)
        - Beam Search functions in inference

**This file is only providing configures to which we prefer, and we RECOMMEND you to employ those configures for your implementation. 
What's more, we would like to train/implement all baselines of Code Summarization task with such configures. 
If there are something that we have not considered, please use what you prefer. 
More importantly, you can add new default hyper-parameters in this file which is beneficial for us all.**