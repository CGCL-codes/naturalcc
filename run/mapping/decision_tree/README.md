# Staic Mapping

## 10-fold validation

no vanilla training because train dataset does not contain `coalesced` info

```shell
python -m run.mapping.static_mapping.10fold
```

|                 |                 | Accura | Speedup |
|-----------------|-----------------|--------|---------|
| Platform        | Benchmark Suite |        |         |
| AMD Tahiti 7970 | AMD SDK         | 68.75  | 1.08    |
|                 | NPB             | 71.92  | 2.14    |
|                 | NVIDIA SDK      | 41.67  | 2.50    |
|                 | Parboil         | 57.89  | 5.81    |
|                 | Polybench       | 48.15  | 12.84   |
|                 | Rodinia         | 64.52  | 3.69    |
|                 | SHOC            | 81.25  | 1.03    |
| NVIDIA GTX 970  | AMD SDK         | 75.00  | 0.86    |
|                 | NPB             | 79.89  | 1.29    |
|                 | NVIDIA SDK      | 33.33  | 0.89    |
|                 | Parboil         | 31.58  | 1.02    |
|                 | Polybench       | 55.56  | 1.03    |
|                 | Rodinia         | 48.39  | 1.06    |
|                 | SHOC            | 70.83  | 1.83    |

|                 | Accuracy | Speedup |
|-----------------|----------|---------|
| Platform        |          |         |
| AMD Tahiti 7970 | 70.29    | 2.64    |
| NVIDIA GTX 970  | 74.56    | 1.28    |


