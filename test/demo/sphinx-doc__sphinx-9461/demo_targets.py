"""Example API objects that the documentation system must inspect."""


class Example:
    @property
    def ordinary_property(self) -> str:
        """An ordinary instance property."""
        return "ordinary"

    @classmethod
    @property
    def class_property(cls) -> str:
        """A class-level property that must appear in generated docs."""
        return "class-level"

