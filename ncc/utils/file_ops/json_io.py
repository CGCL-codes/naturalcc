# -*- coding: utf-8 -*-

try:
    import ujson as json
except:
    import json


def json_dump(*args, **kwargs):
    ensure_ascii = kwargs.pop('ensure_ascii', False)
    return json.dump(*args, ensure_ascii=ensure_ascii, **kwargs)


def json_dumps(*args, **kwargs):
    ensure_ascii = kwargs.pop('ensure_ascii', False)
    return json.dumps(*args, ensure_ascii=ensure_ascii, **kwargs)


json_loads = json.loads
json_load = json.load


################## jsonl ##################
def jsonlines_load(file):
    # load jsonlines from a UTF-8 file
    with open(file, mode='r', encoding='UTF-8') as reader:
        return [json_dumps(line) for line in reader]


def jsonlines_dump(objs, file):
    # save jsonlines into a UTF-8 file
    with open(file, mode='w', encoding='UTF-8') as writer:
        for obj in objs:
            print(json_dumps(obj), file=writer)


__all__ = [
    json_load, json_loads,
    json_dump, json_dump,

    jsonlines_load, jsonlines_dump,
]
