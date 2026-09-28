"""Standalone reproduction for the attrs-aliasing bug.

Run ``python demo_repro.py``. It should fail before the repair and pass after
the repair without modifying this file.
"""

from mini_xarray import Dataset, merge


first = Dataset(attrs={"a": "b"})
second = Dataset(attrs={"a": "c"})
merged = merge([first, second], combine_attrs="override")

merged.attrs["a"] = "changed"

assert first.attrs == {"a": "b"}, (
    "BUG: updating the merged attrs unexpectedly modified the first input"
)
assert second.attrs == {"a": "c"}
print("attrs are independent")

