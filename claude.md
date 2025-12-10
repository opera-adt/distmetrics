# Python Coding & Editing Guidelines

## Table of Contents

1. Philosophy
1. Docstrings & Comments
1. Type Hints
1. Documentation

---

## Philosophy

- **Readability, reproducibility, performance – in that order.**
- Prefer explicit over implicit; avoid hidden state and global flags.
- Code should be *self-documenting* - we can assume readers understand the underlying libraries, do not write additional inline comments unless they provide crucial explanations.
- Code should be *re-useable* - often we copy/past code or want to use functions - this is what creates the greatest value.

## Docstrings & Comments

- Style: NumPyDoc.
- Do not create docstrings unless asked to do so.
- Start with a one‑sentence summary in the imperative mood.
- Sections: Parameters, Returns, Raises, Examples, References.
- Use backticks for code or referring to variables (e.g. `xarray.DataArray`).
- Do not use emojis, or non-unicode characters in comments/print statements.
- Cite peer‑reviewed papers with DOI links when relevant.
- Write code that explains itself rather than needs comments.
- Comments should be things which are not obvious to a reader with typical background knowledge.
- For pytorch einops, please describe dimension transfomrations

## Tools

- ruff is use for most code maintenance, black for formatting, pytest for testing

## Environment
- For python repositories, we use mamba to manage the environment. Please use the `name` in the `environment.yml and activate that before running tests.

## Code Style

- Annotate all public functions (PEP 484).
- Prefer `Protocol` over `ABC`s when only an interface is needed.
- Validate external inputs via Pydantic models (if existing); otherwise, use `dataclasses`
- Parse, don't validate, with your dataclasses. Checks should be at the serialization boundaries, not scattered everywhere in the code.
- If you need to add an ignore, ignore a specific check like # type: ignore[specific]
- Don't write error handing code or smooth over exceptions/errors unless they are expected as part of control flow.
- In general, write code that will raise an exception early if something isn't expected.
- Enforce important expectations with asserts, but raise errors for user-facing problems.

## Documentation

- mkdocs + Jupyter. Hosted on ReadTheDocs.
- Auto API from type hints.
- Provide tutorial notebooks covering common workflows.
- Include examples in docstrings.
- Add high-level guides for key functionality.