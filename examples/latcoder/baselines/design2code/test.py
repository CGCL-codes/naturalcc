# import os

# def count_subfolders(folder_path):
#     try:
#         # 
#         entries = os.listdir(folder_path)
        
#         # 
#         subfolder_count = 0
#         for entry in entries:
#             # 
#             full_path = os.path.join(folder_path, entry)
#             # 
#             if os.path.isdir(full_path):
#                 subfolder_count += 1
        
#         return subfolder_count
#     except FileNotFoundError:
#         print(f" '{folder_path}' ")
#         return 0

# # 
# folder_path = 'xx'  # 
# num_subfolders = count_subfolders(folder_path)
# print(f" '{folder_path}'  {num_subfolders} ")

# import os
# import shutil

# def copy_folders(source_folder, target_folder):
#     """
#      source_folder  target_folder 
#     """
#     if not os.path.exists(source_folder):
#         print(f"：{source_folder}")
#         return

#     # ，
#     if not os.path.exists(target_folder):
#         os.makedirs(target_folder)
#         print(f"：{target_folder}")

#     # 
#     for folder_name in os.listdir(source_folder):
#         folder_path = os.path.join(source_folder, folder_name)
#         if os.path.isdir(folder_path):  # 
#             target_path = os.path.join(target_folder, folder_name)
            
#             # ，
#             if os.path.exists(target_path):
#                 print(f"：{folder_name} ")
#                 continue
            
#             # 
#             shutil.copytree(folder_path, target_path)
#             print(f"：{folder_name} -> {target_path}")
#         else:
#             print(f"：{folder_name} ")

# # 
# source_folder = 'xx'  # 
# target_folder = 'xx'  # 
# copy_folders(source_folder, target_folder)


# import pandas as pd

# df = pd.read_csv("XX/evaluation/result/design2code_hard/baseline_4o_method0/metrics_result.csv")
# for c in df.columns:
#         if c not in ["origin","pred"]:
#             print(f"{c}:{df[c].mean():.4f}")

import os,sys
sys.path.append(os.path.abspath('.'))

os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'

from vendors.google__ import gemini
from vendors.openai__ import gpt4o

answer = gemini(",")

print(answer)