"""A small analogue of Sphinx's autodoc member-selection path."""

from __future__ import annotations

from typing import Any

from . import domain
from . import inspect as mini_inspect


class PropertyDocumenter:
    """Document a property defined directly on a class."""

    @classmethod
    def can_document_member(cls, member: Any) -> bool:
        """Return whether this documenter can handle the raw descriptor."""
        return mini_inspect.is_property(member)

    @classmethod
    def document_member(cls, owner: type[Any], member_name: str) -> str | None:
        """Render a property defined on *owner*, or return None when skipped."""
        raw_member = owner.__dict__.get(member_name)
        if raw_member is None or not cls.can_document_member(raw_member):
            return None

        property_member = mini_inspect.unwrap_property(raw_member)
        return domain.render_property(
            member_name,
            property_member.__doc__,
            is_classmethod=False,
        )


def document_member(owner: type[Any], member_name: str) -> str | None:
    """Choose the property documenter for an attribute on *owner*."""
    return PropertyDocumenter.document_member(owner, member_name)

