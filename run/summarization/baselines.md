# Baseline
**Search Strategy in inference: greedy search**

## Seq2Seq
##### 512*4
| Datasets       | Bleu-4 | Rouge-L | Meteor |
|----------------|--------|---------|--------|
| csn_ruby       | 10.86  | 14.50   | 3.11   |
| csn_go         | 20.21  | 43.02   | 18.90  |
| csn_python     | 15.99  | 31.03   | 11.40  |
| csn_php        | 22.01  | 37.54   | 14.76  |
| csn_java       | 16.58  | 33.68   | 12.82  |
| csn_javascript | 13.35  | 21.74   | 8.39   |
| python_wan     | 21.26  | 37.00   | 12.16  |
| java_hu        | 24.53  | 39.52   | 16.15  |
| so_c#          | 12.79  | 17.49   | 6.57   |
| so_sql         | 13.51  | 13.05   | 4.33   |
| so_python      | 8.94   | 12.01   | 4.89   |

## Seq2Seq
##### 128*4
| Datasets       | Bleu-4 | Rouge-L | Meteor |
|----------------|--------|---------|--------|
| csn_ruby       | 12.49  | 20.30   | 7.57   |
| csn_go         | 20.48  | 43.38   | 19.19  |
| csn_python     | 16.26  | 32.01   | 11.78  |
| csn_php        | 22.31  | 38.12   | 15.04  |
| csn_java       | 16.35  | 33.53   | 12.76  |
| csn_javascript | 13.43  | 21.73   | 8.43   |
| python_wan     | 25.24  | 40.11   | 14.47  |
| java_hu        | 31.44  | 44.47   | 20.19  |
| so_c#          | 13.33  | 20.74   | 9.63   |
| so_sql         |        |         |        |
| so_python      | 13.28  | 21.14   | 8.90   |

## Nary Tree2seq
##### 128*4
| Datasets       | Bleu-4 | Rouge-L | Meteor |
|----------------|--------|---------|--------|
| csn_ruby       | 12.05  | 19.06   | 7.19   |
| csn_go         | 17.27  | 39.92   | 17.28  |
| csn_python     | 15.81  | 30.58   | 10.97  |
| csn_php        | 21.25  | 36.52   | 14.28  |
| csn_java       | 15.53  | 31.57   | 11.70  |
| csn_javascript | 13.12  | 20.20   | 7.53   |
| python_wan     |        |         |        |
| java_hu        |        |         |        |
| so_c#          |        |         |        |
| so_sql         |        |         |        |
| so_python      |        |         |        |

## CodeNN
##### 100
| Datasets       | Bleu-4 | Rouge-L | Meteor |
|----------------|--------|---------|--------|
| csn_ruby       | 12.20  | 19.58   | 5.94   |
| csn_go         | 13.53  | 32.51   | 12.68  |
| csn_python     | 14.20  | 25.60   | 8.27   |
| csn_php        | 18.28  | 26.17   | 10.86  |
| csn_java       | 14.41  | 28.65   | 10.21  |
| csn_javascript | 12.71  | 20.83   | 7.18   |
| python_wan     | 20.33  | 34.84   | 11.26  |
| java_hu        | 19.96  | 33.21   | 12.80  |
| so_c#          | 13.18  | 18.32   | 8.51   |
| so_sql         | 13.31  | 13.98   | 6.03   |
| so_python      | 14.03  | 21.35   | 9.44   |

## DeepCom
##### 100
| Datasets       | Bleu-4 | Rouge-L | Meteor |
|----------------|--------|---------|--------|
| csn_ruby       | 11.88  | 15.84   | 7.63   |
| csn_go         | 14.05  | 28.70   | 15.22  |
| csn_python     | 14.61  | 25.66   | 8.46   |
| csn_php        | 18.64  | 29.84   | 10.46  |
| csn_java       | 14.48  | 26.87   | 9.32   |
| csn_javascript | 12.61  | 17.60   | 6.46   |
| python_wan     |        |         |        |
| java_hu        |        |         |        |
| so_c#          |        |         |        |
| so_sql         |        |         |        |
| so_python      |        |         |        |

## DeepCom
##### 100
| Datasets       | Bleu-4 | Rouge-L | Meteor |
|----------------|--------|---------|--------|
| csn_ruby       | 11.90  | 17.38   | 6.62   |
| csn_go         | 14.05  | 28.70   | 15.22  |
| csn_python     | 14.61  | 25.66   | 8.46   |
| csn_php        | 18.64  | 29.84   | 10.46  |
| csn_java       | 14.48  | 26.87   | 9.32   |
| csn_javascript | 12.61  | 17.60   | 6.46   |
| python_wan     |        |         |        |
| java_hu        |        |         |        |
| so_c#          |        |         |        |
| so_sql         |        |         |        |
| so_python      |        |         |        |

## Vanilla Transformer
##### 128*4
| Datasets       | Bleu-4 | Rouge-L | Meteor |
|----------------|--------|---------|--------|
| csn_ruby       | 11.62  | 15.53   | 2.24   |
| csn_go         | 19.80  | 42.69   | 18.36  |
| csn_python     | 15.70  | 29.95   | 10.62  |
| csn_php        | 22.09  | 36.70   | 14.47  |
| csn_java       | 15.86  | 32.22   | 11.46  |
| csn_javascript | 13.18  | 19.95   | 7.51   |
| python_wan     | 26.76  | 41.93   | 15.31  |
| java_hu        | 42.12  | 52.49   | 26.75  |
| so_c#          | 13.35  | 19.89   | 8.79   |
| so_sql         | 13.21  | 14.57   | 6.61   |
| so_python      | 14.38  | 19.86   | 8.84   |

