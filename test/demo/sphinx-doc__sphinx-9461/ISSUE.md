# Bug report: `@classmethod @property` members are omitted from generated docs

The documentation pipeline handles ordinary `@property` descriptors, but it
skips a property wrapped by `@classmethod`.

## Reproduction

```python
class Example:
    @classmethod
    @property
    def class_property(cls):
        """A class-level property that must appear in generated docs."""
        return "class-level"
```

`document_member(Example, "class_property")` currently returns `None`.

## Expected behavior

The member should be documented as:

```text
class property class_property: A class-level property that must appear in generated docs.
```

Ordinary properties must retain their existing rendering.

## Relevant multi-file path

```text
mini_sphinx/inspect.py
    -> recognises and unwraps Python descriptors
mini_sphinx/autodoc.py
    -> chooses PropertyDocumenter and forwards descriptor metadata
mini_sphinx/domain.py
    -> renders the final property/class-property directive
```

## Scope

- Make the smallest correct changes in `mini_sphinx/`.
- Do not modify `demo_targets.py`, `demo_repro.py`, or `tests/`.
- Preserve ordinary-property behavior.
- Do not reformat unrelated code.

