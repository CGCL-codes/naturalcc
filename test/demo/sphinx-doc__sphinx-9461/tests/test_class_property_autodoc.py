from demo_targets import Example
from mini_sphinx import domain
from mini_sphinx.autodoc import document_member


def test_ordinary_property_is_still_documented() -> None:
    rendered = document_member(Example, "ordinary_property")

    assert rendered == "property ordinary_property: An ordinary instance property."


def test_classmethod_property_is_documented_as_a_class_property() -> None:
    rendered = document_member(Example, "class_property")

    assert rendered == (
        "class property class_property: "
        "A class-level property that must appear in generated docs."
    )


def test_domain_keeps_the_class_property_marker() -> None:
    rendered = domain.render_property("example", "A description.", is_classmethod=True)

    assert rendered == "class property example: A description."

