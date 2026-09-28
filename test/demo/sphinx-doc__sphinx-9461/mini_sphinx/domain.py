"""The final rendering layer for Python property directives."""

from __future__ import annotations


def render_property(name: str, docstring: str | None, *, is_classmethod: bool) -> str:
    """Render one property directive.

    ``is_classmethod`` is supplied by autodoc, but the pre-fix renderer ignores
    it and consequently loses the fact that the property is class-level.
    """
    label = "property"
    description = docstring or "<missing docstring>"
    return f"{label} {name}: {description}"

