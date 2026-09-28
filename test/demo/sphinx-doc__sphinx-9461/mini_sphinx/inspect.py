"""Static helpers used by the mini autodoc implementation.

The functions intentionally model the pre-fix behavior for this demonstration.
"""

from __future__ import annotations

from typing import Any


def is_property(member: Any) -> bool:
    """Return whether *member* is a normal Python property descriptor.

    This is incomplete: Python permits ``@classmethod`` to wrap ``@property``.
    Such a descriptor is a ``classmethod`` whose ``__func__`` is a property.
    """
    return isinstance(member, property)


def unwrap_property(member: Any) -> property:
    """Return the underlying property descriptor.

    The current implementation only works for ordinary properties. A repair
    must preserve this contract for both ordinary and class properties.
    """
    if not isinstance(member, property):
        raise TypeError("member is not a property")
    return member


def is_class_property(member: Any) -> bool:
    """Return whether a property is exposed through ``@classmethod``.

    This placeholder is intentionally wrong so the rendering layer cannot
    distinguish a class property from an ordinary property.
    """
    return False

