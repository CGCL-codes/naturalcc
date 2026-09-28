"""A deliberately small model of xarray's Dataset attribute merge behavior.

This workspace begins in a buggy state for an Agent code-repair demonstration.
The intended behavior is described in ISSUE.md and enforced by tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


class Dataset:
    """A tiny dataset that owns a mutable mapping of metadata attributes."""

    def __init__(self, attrs: Mapping[str, Any] | None = None) -> None:
        self.attrs = {} if attrs is None else attrs

    @property
    def a(self) -> Any:
        """Convenience accessor used by the issue reproduction."""
        return self.attrs["a"]


def merge(datasets: Iterable[Dataset], *, combine_attrs: str = "override") -> Dataset:
    """Merge datasets and combine their metadata attributes.

    ``override`` means that values from the first dataset win. It must not
    mean that the result shares the first dataset's mutable ``attrs`` mapping.
    """
    datasets = list(datasets)
    if not datasets:
        raise ValueError("at least one Dataset is required")

    return Dataset(attrs=merge_attrs([dataset.attrs for dataset in datasets], combine_attrs))


def merge_attrs(variable_attrs: list[Mapping[str, Any]], combine_attrs: str) -> Mapping[str, Any]:
    """Return combined metadata according to the selected merge policy."""
    if combine_attrs == "drop":
        return {}
    if combine_attrs == "override":
        return variable_attrs[0]
    raise ValueError(f"unsupported combine_attrs policy: {combine_attrs!r}")

