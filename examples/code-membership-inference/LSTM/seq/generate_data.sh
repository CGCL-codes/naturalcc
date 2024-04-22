MODEL=target
TYPES=("train" "test" "dev")

for TYPE in ${TYPES[*]}
python -u generate_data.py \
       -f=/pathto/data/membership_inference/${MODEL}/python_${TYPE}.txt \
       --base_dir=/pathto/data/py150_files \
       --out_fp=/pathto/data/membership_inference/${MODEL}/lstm/${TYPE}.txt \
       --id_type=token
done