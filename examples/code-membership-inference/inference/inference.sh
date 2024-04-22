export CUDA_VISIBLE_DEVICES=1
SHADOW_MODEL_TYPE=code-gpt
TARGET_MODEL_TYPE=code-gpt
SHADOW_MODEL_NUMBER=10
OUTPUT_SIZE=50234

LOGFILE=${SHADOW_MODEL_TYPE}-${SHADOW_MODEL_NUMBER}.log
# LOGFILE=shadow_${SHADOW_MODEL_TYPE}-target_${TARGET_MODEL_TYPE}.log

nohup python -u inference.py --do_train
                             --do_eval
                             --number_of_shadow_model ${SHADOW_MODEL_NUMBER} \
                             --shadow_model_type ${SHADOW_MODEL_TYPE} \
                             --target_model_type ${TARGET_MODEL_TYPE} \
                             --output_size ${OUTPUT_SIZE} > log/${LOGFILE} 2>&1 &
