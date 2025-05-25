import json
with open("./local_repo_usagegraph.json") as f:
    local_graph = json.load(f)

zero = 0
result_dict = {}
arg_total = 0
arg_list = 0
return_total = 0
return_list = 0
for r, value in local_graph.items():
    r1 = r
    zero = zero + 1
    #if zero == 500:
        #break;
    split_info = r.rsplit(".py", 1)
    r = {
        "file": split_info[0] + ".py",
        "loc": split_info[1][2:].split("--")[0],
        "name": split_info[1][2:].split("--")[1],
        "scope": split_info[1][2:].split("--")[2]
    }
    key = '{}--{}--{}--{}'.format(r["file"], r["loc"], r["name"], r["scope"])
    name = r["name"]
    scope = r["scope"]
    loc = r["loc"]
    filename = r["file"]
    if scope == "arg":
        arg_total += 1
    if scope == "return":
        return_total += 1
    part_to_remove = "user=proj'test/" + name
    print(r1)
    print(part_to_remove)
    #if r1 == "repos/BNDKPNTR/EmployeeScheduler/DeepQScheduler/addOnlyEnvironment.py--_calculateReward@AddOnlyEnvironment--actionIndex--arg":
    #    exit(1)
    if part_to_remove not in value:
        result_dict[r1] = {}
    else:
        if scope == "arg":
            arg_list += 1
        if scope == "return":
            return_list += 1
        result_dict[r1] = value
new_json_file = 'updated_local_graph.json'
with open(new_json_file, 'w') as json_file:
    json.dump(result_dict, json_file, indent=2)
print(arg_total)
print(arg_list)
print(return_total)
print(return_list)
