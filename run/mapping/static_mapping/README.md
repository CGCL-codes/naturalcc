# Staic Mapping

## 10-fold validation/vanilla training

```shell
python -m run.mapping.static_mapping.10fold
python -m run.mapping.static_mapping.eval
```

|                 |                 | Accura | Speedup |
|-----------------|-----------------|--------|---------|
| Platform        | Benchmark Suite |        |         |
| AMD Tahiti 7970 | AMD SDK         | 62.50  | 1.0     |
|                 | NPB             | 61.86  | 1.0     |
|                 | NVIDIA SDK      | 8.33   | 1.0     |
|                 | Parboil         | 47.37  | 1.0     |
|                 | Polybench       | 7.41   | 1.0     |
|                 | Rodinia         | 54.84  | 1.0     |
|                 | SHOC            | 72.92  | 1.0     |
| NVIDIA GTX 970  | AMD SDK         | 93.75  | 1.0     |
|                 | NPB             | 62.24  | 1.0     |
|                 | NVIDIA SDK      | 58.33  | 1.0     |
|                 | Parboil         | 31.58  | 1.0     |
|                 | Polybench       | 55.56  | 1.0     |
|                 | Rodinia         | 38.71  | 1.0     |
|                 | SHOC            | 8.33   | 1.0     |

|                 | Accuracy | Speedup |
|-----------------|----------|---------|
| Platform        |          |         |
| AMD Tahiti 7970 | 58.82    | 1.0     |
| NVIDIA GTX 970  | 56.91    | 1.0     |

