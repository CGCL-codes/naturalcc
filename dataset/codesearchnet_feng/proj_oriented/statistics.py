import numpy as np

proj_lens = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62,
    63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91,
    92,
    93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
    117,
    118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
    141,
    142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
    165,
    166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188,
    189,
    190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212,
    213,
    214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 234, 235, 236, 237,
    238,
    239, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 254, 255, 256, 257, 258, 259, 261, 265, 266, 267,
    268,
    269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 281, 282, 283, 284, 287, 288, 289, 290, 292, 293, 294, 296,
    297,
    298, 299, 300, 301, 302, 303, 305, 306, 307, 308, 309, 310, 311, 312, 315, 316, 317, 318, 320, 321, 323, 324, 325,
    326,
    327, 328, 329, 330, 331, 334, 335, 336, 338, 339, 340, 342, 343, 344, 347, 348, 350, 351, 353, 355, 356, 358, 360,
    361,
    362, 365, 366, 367, 368, 369, 372, 374, 375, 377, 378, 379, 380, 381, 382, 383, 384, 387, 388, 389, 390, 392, 393,
    394,
    396, 400, 401, 402, 404, 408, 410, 412, 413, 416, 419, 420, 421, 423, 425, 428, 429, 433, 434, 435, 436, 437, 438,
    439,
    445, 446, 447, 448, 451, 458, 461, 462, 463, 464, 466, 468, 469, 472, 474, 476, 479, 480, 481, 483, 484, 485, 486,
    487,
    488, 490, 494, 495, 499, 502, 506, 507, 508, 511, 515, 521, 522, 523, 526, 530, 531, 537, 539, 540, 542, 543, 546,
    547,
    549, 550, 556, 557, 560, 568, 572, 573, 579, 584, 585, 587, 588, 589, 590, 594, 598, 603, 607, 610, 611, 612, 617,
    619,
    620, 622, 628, 631, 632, 637, 638, 646, 648, 650, 651, 660, 662, 664, 670, 677, 679, 685, 687, 692, 701, 703, 708,
    709,
    715, 720, 731, 740, 742, 774, 786, 792, 793, 794, 802, 803, 804, 813, 814, 816, 821, 822, 823, 829, 830, 834, 852,
    856,
    858, 884, 887, 889, 904, 909, 911, 913, 915, 920, 931, 939, 953, 955, 976, 990, 993, 994, 1007, 1013, 1014, 1032,
    1033,
    1035, 1044, 1055, 1070, 1076, 1080, 1086, 1087, 1093, 1097, 1112, 1133, 1134, 1141, 1183, 1205, 1212, 1217, 1236,
    1252,
    1272, 1290, 1308, 1312, 1330, 1335, 1369, 1370, 1404, 1412, 1417, 1427, 1436, 1442, 1458, 1533, 1629, 1675, 1728,
    1803,
    1850, 1882, 1885, 1983, 2131, 2239, 2270, 2302, 2696, 2711, 2828, 2975, 3180, 6329, 6637, 7016, 7778, 8759, 8842)
proj_nums = (
    11706, 6812, 4590, 3416, 2754, 2203, 1845, 1525, 1381, 1138, 991, 869, 784, 728, 616, 619, 502, 477, 432, 436, 360,
    307,
    320, 284, 276, 256, 265, 219, 217, 201, 197, 184, 142, 165, 150, 151, 139, 126, 130, 131, 98, 120, 100, 91, 112, 90,
    83,
    91, 90, 72, 66, 74, 53, 60, 66, 70, 69, 66, 62, 58, 56, 39, 63, 43, 41, 40, 38, 53, 50, 40, 31, 29, 37, 29, 36, 38,
    25,
    38, 30, 35, 27, 32, 33, 17, 24, 25, 28, 19, 23, 24, 27, 19, 24, 28, 20, 17, 14, 19, 19, 18, 21, 22, 25, 12, 17, 14,
    13,
    15, 20, 11, 16, 19, 13, 14, 14, 10, 14, 13, 15, 14, 18, 10, 18, 12, 18, 16, 9, 8, 9, 17, 9, 8, 16, 6, 5, 10, 7, 10,
    6,
    6, 8, 13, 6, 15, 8, 2, 9, 14, 6, 8, 13, 9, 8, 6, 12, 4, 7, 8, 7, 10, 8, 11, 7, 9, 6, 6, 4, 6, 10, 8, 8, 11, 8, 6, 3,
    7,
    3, 2, 4, 5, 11, 4, 3, 6, 3, 5, 5, 1, 5, 5, 5, 2, 6, 1, 5, 2, 4, 2, 6, 8, 5, 6, 1, 4, 4, 3, 4, 3, 4, 4, 4, 5, 8, 4,
    6, 5,
    3, 4, 2, 7, 1, 6, 2, 4, 3, 7, 2, 3, 2, 2, 3, 5, 2, 2, 4, 4, 6, 1, 2, 4, 2, 2, 3, 4, 1, 2, 2, 6, 2, 3, 4, 3, 1, 1, 3,
    5,
    5, 5, 2, 4, 4, 3, 4, 3, 1, 3, 1, 2, 1, 2, 2, 4, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 3, 4, 3, 2, 5, 1, 3, 2, 1,
    5,
    3, 2, 3, 1, 1, 1, 1, 3, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 3, 1, 1, 1, 3, 2, 2, 3, 1, 2, 1,
    1,
    1, 1, 2, 1, 2, 1, 2, 2, 2, 3, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 3, 3, 1, 1, 1, 2, 2,
    1,
    1, 1, 2, 3, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 2, 2, 1, 1,
    2,
    1, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 2,
    1,
    2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1,
    1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1,
    1, 1, 1, 1, 1, 1, 1)

proj_lens = np.asarray(proj_lens)
proj_nums = np.asarray(proj_nums)
codes = proj_lens * proj_nums
print(np.sum(50 < proj_nums < 500))