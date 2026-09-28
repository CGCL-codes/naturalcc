from mini_xarray import Dataset, merge


def test_override_attrs_are_independent_from_first_input() -> None:
    """The regression test corresponding to SWE-bench pydata__xarray-4629."""
    first = Dataset(attrs={"a": "b"})
    second = Dataset(attrs={"a": "c"})

    merged = merge([first, second], combine_attrs="override")
    merged.attrs["a"] = "changed"

    assert first.attrs == {"a": "b"}
    assert second.attrs == {"a": "c"}
    assert merged.attrs == {"a": "changed"}


def test_override_keeps_the_first_datasets_values() -> None:
    first = Dataset(attrs={"a": "first"})
    second = Dataset(attrs={"a": "second"})

    merged = merge([first, second], combine_attrs="override")

    assert merged.attrs == {"a": "first"}


def test_drop_discards_all_attributes() -> None:
    merged = merge(
        [Dataset(attrs={"a": "first"}), Dataset(attrs={"a": "second"})],
        combine_attrs="drop",
    )

    assert merged.attrs == {}

