# Bug report: merged attributes unexpectedly modify the input dataset

`merge([...], combine_attrs="override")` should use the first dataset's
attribute *values*, but the merged result must own a separate attribute mapping.

## Reproduction

```python
from mini_xarray import Dataset, merge

first = Dataset(attrs={"a": "b"})
second = Dataset(attrs={"a": "c"})
merged = merge([first, second], combine_attrs="override")

merged.attrs["a"] = "changed"
assert first.attrs == {"a": "b"}
```

The assertion currently fails because the result aliases the first input's
mutable `attrs` mapping.

## Expected behavior

Changing `merged.attrs` must not change either input dataset. Preserve the
`override` policy: the result should still initially contain the first
dataset's attribute values.

## Scope

- Make the smallest production-code change needed in `mini_xarray.py`.
- Do not edit `demo_repro.py` or files under `tests/`.
- Do not reformat unrelated code.

