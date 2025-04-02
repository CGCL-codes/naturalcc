import json

# 读取原始JSON文件
with open('NVBench.json', 'r') as f:
    data = json.load(f)

# 清洗数据
new_data = []
for item in data:
    item = data[item]

    if 'nl_queries' in item:
        for nl in item['nl_queries']:
            if nl is not None:
                new_item = {}
                new_item['nl_queries'] = nl
                if 'db_id' in item:
                    new_item['db_id'] = item['db_id']
                    if 'vis_query' in item:
                        new_item['vis_query'] = item['vis_query']
                        if 'hardness' in item:
                            new_item['hardness'] = item['hardness']
                        new_data.append(new_item)
print(len(new_data))
# 将新的数据写入新JSON文件
with open('nl2vis.json', 'w') as f:
    json.dump(new_data, f)
