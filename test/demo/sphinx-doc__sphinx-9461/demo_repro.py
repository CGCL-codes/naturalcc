"""Standalone reproduction for the class-property autodoc bug."""

from demo_targets import Example
from mini_sphinx.autodoc import document_member


rendered = document_member(Example, "class_property")

assert rendered is not None, (
    "BUG: @classmethod @property is skipped by the autodoc member selector"
)
assert rendered == (
    "class property class_property: "
    "A class-level property that must appear in generated docs."
)
print(rendered)

