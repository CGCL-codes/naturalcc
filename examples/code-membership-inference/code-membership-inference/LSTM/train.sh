export CUDA_VISIBLE_DEVICES=0
MODEL=shadow_9
DATADIR=/pathto/data/membership_inference/${MODEL}/lstm
LOGFILE=${MODEL}_model_train.log

nohup python -u seqrnn.py --base_dir=${DATADIR} > logfile/${LOGFILE} 2>&1 &