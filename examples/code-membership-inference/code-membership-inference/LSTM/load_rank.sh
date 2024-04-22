export CUDA_VISIBLE_DEVICES=0
MODEL=shadow_3
DATADIR=/pathto/data/membership_inference/${MODEL}/lstm
RANKDIR=/pathto/data/membership_inference/${MODEL}/lstm/ranks
LOGFILE=${MODEL}_model_rank.log

nohup python -u rank.py --base_dir=${DATADIR} --rank_dir=${RANKDIR} > logfile/${LOGFILE} 2>&1 &