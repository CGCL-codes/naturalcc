import json
import random

# 读取原始JSON文件
with open('nl2vis.json', 'r') as f:
    data = json.load(f)

# Sort the data by database name
data = sorted(data, key=lambda x: x['db_id'])

# 计算划分比例
n = len(data)
train_ratio = 0.7
valid_ratio = 0.2
test_ratio = 0.1

# 按数据库划分数据

for i in range(int(n*train_ratio), n):
    if data[i]['db_id'] == data[i+1]['db_id']:
        continue
    else:
        break
train_data = data[:i+1]

for j in range(i + int(n*valid_ratio), n):
    if data[j]['db_id'] == data[j + 1]['db_id']:
        continue
    else:
        break
valid_data = data[i+1:j+1]
test_data = data[j+1:]
print("按数据库划分")
print(n, len(train_data), len(valid_data), len(test_data))
# 计算按数据库划分方式下的train, valid, test之间重复db的数量
train_db_ids = set(d['db_id'] for d in train_data)
valid_db_ids = set(d['db_id'] for d in valid_data)
test_db_ids = set(d['db_id'] for d in test_data)

print("按数据库划分方式下：")
print("train数据集中有{}个数据库".format(len(train_db_ids)))
print("valid数据集中有{}个数据库".format(len(valid_db_ids)))
print("test数据集中有{}个数据库".format(len(test_db_ids)))
train_valid_overlap = len(train_db_ids.intersection(valid_db_ids))
train_test_overlap = len(train_db_ids.intersection(test_db_ids))
valid_test_overlap = len(valid_db_ids.intersection(test_db_ids))

print("按数据库划分方式下：")
print("train和valid之间有{}个重复db".format(train_valid_overlap))
print("train和test之间有{}个重复db".format(train_test_overlap))
print("valid和test之间有{}个重复db".format(valid_test_overlap))
# 将数据写入文件
# with open('./data_database_split/train.json', 'w') as f:
#     json.dump(train_data, f)
#
# with open('./data_database_split/valid.json', 'w') as f:
#     json.dump(valid_data, f)
#
# with open('./data_database_split/test.json', 'w') as f:
#     json.dump(test_data, f)

# 随机打乱数据
random.seed(42)
random.shuffle(data)
# 划分数据
train_data_radn = data[:int(n*train_ratio)+1]
valid_data_radn = data[int(n*train_ratio)+1:int(n*(train_ratio+valid_ratio))+1]
test_data_radn = data[int(n*(train_ratio+valid_ratio))+1:]
print("随机划分")
print(n, len(train_data_radn), len(valid_data_radn), len(test_data_radn))

# 将数据写入文件
with open('./data_radn_split/train.json', 'w') as f:
    json.dump(train_data_radn, f)

with open('./data_radn_split/valid.json', 'w') as f:
    json.dump(valid_data_radn, f)

with open('./data_radn_split/test.json', 'w') as f:
    json.dump(test_data_radn, f)

# 计算随机划分方式下的train, valid, test之间重复db的数量
random_train_db_ids = set(d['db_id'] for d in train_data_radn)
random_valid_db_ids = set(d['db_id'] for d in valid_data_radn)
random_test_db_ids = set(d['db_id'] for d in test_data_radn)
print("随机划分方式下：")
print("train数据集中有{}个数据库".format(len(random_train_db_ids)))
print("valid数据集中有{}个数据库".format(len(random_valid_db_ids)))
print("test数据集中有{}个数据库".format(len(random_test_db_ids)))

random_train_valid_overlap = len(random_train_db_ids.intersection(random_valid_db_ids))
random_train_test_overlap = len(random_train_db_ids.intersection(random_test_db_ids))
random_valid_test_overlap = len(random_valid_db_ids.intersection(random_test_db_ids))

print("随机划分方式下：")
print("train和valid之间有{}个重复db".format(random_train_valid_overlap))
print("train和test之间有{}个重复db".format(random_train_test_overlap))
print("valid和test之间有{}个重复db".format(random_valid_test_overlap))

print("radn train和final valid之间有{}个重复db".format(len(random_train_db_ids.intersection(valid_db_ids))))
print("radn train和final test之间有{}个重复db".format(len(random_train_db_ids.intersection(test_db_ids))))
# print("valid和test之间有{}个重复db".format(len(random_train_db_ids.intersection(random_valid_db_ids))))