#!/bin/bash
python run.py --task='codexglue_defect_detection' \
--models='microsoft/codebert-base' \
--selection_method='Logistic' \
--budget=10

