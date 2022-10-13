# -*- coding: utf-8 -*-
import os
if ('NCC' not in os.environ):
    print("The dataset variable $NCC is not set, please first define the variable");
    exit(1)
else:
    print(f"NaturalCC dataset and cache path: '{os.getenv('NCC')}'");
