import json
with open("./data/testset_staticsliced_hop3.json") as f:
    data = json.load(f)

with open("./data/testset_transformed.json") as f:
    simple_data = json.load(f)


def count_occurrences(main_string, sub_string):
    count = 0
    start_index = 0

    while True:
        # Find the next occurrence of sub_string starting from start_index
        index = main_string.find(sub_string, start_index)

        # If sub_string is found
        if index != -1:
            # Increment count
            count += 1
            # Update start_index to search for next occurrence
            start_index = index + 1
        else:
            # If sub_string is not found, break the loop
            break

    return count
total = 0
total_dayuer = 0
zero = 0
for r, value in data.items():
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
    #if simple_data[r1][2]!= 'user-defined' and simple_data[r1][2]!= 'simple':
    try:
        #occurrences = count_occurrences(value, name + ' =')
        occurrences = count_occurrences(value, " if ")
        if occurrences >=1:
            #print(r1)
            total_dayuer = total_dayuer + 1
        total = total + 1
    except:
        pass
print(total_dayuer/total)
