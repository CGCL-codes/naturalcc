from collections import Counter
import matplotlib.pyplot as plt

# c = Counter({0: 4925, 1: 1136, 2: 550, 3: 216,
#              4: 74, 5: 33, 6: 12, 8: 9, 9: 7, 7: 3})

c = Counter(
    {
        0: 66903,
        1: 10827,
        2: 4922,
        3: 1844,
        4: 724,
        5: 271,
        6: 137,
        7: 52,
        8: 48,
        9: 16,
        11: 6,
        10: 6,
        12: 4,
        13: 2,
        14: 2,
        17: 1,
    }
)

# fig, ax = plt.subplots(1, 1)
# n_bins = max(list(c.keys()))
# # del c[0]
# bins = [x for x in range(0, n_bins+2)]

# x = [y for y in list(c.elements())]

# ax.hist(x,
#         bins=bins, align='left', ec='black', cumulative=False)
# ax.set_xticks(bins[:-1])
# ax.set_yscale('log')
# # plt.xlim(10, -1)
# plt.title("Histogram of type annotations' arity")
# plt.xlabel("Parameter-arity")
# plt.ylabel("Number of occurences")
# plt.show()

total = 0
for k in range(18):
    total += c[k]
s = 0
for k in range(18):
    s += c[k]
    if s / total >= 0.95:
        print("0.95%: ", k)

    if s / total >= 0.99:
        print("0.99%: ", k)
