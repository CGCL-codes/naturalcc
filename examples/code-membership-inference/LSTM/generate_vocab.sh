MODEL=shadow_9

python -u generate_vocab.py \
       --input_fp /pathto/data/membership_inference/${MODEL}/lstm/train.txt \
       --out_fp /pathto/data/membership_inference/${MODEL}/lstm/vocab.pkl \
       --input_type source_code
