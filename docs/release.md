# Building a release

From the repository root in a fresh Python 3.12 environment, run

```bash
python -m pip install -e '.[test,docs]' build twine
python -m pytest -m 'not few_data'
python -m pytest -m few_data
python -m mkdocs build --strict
python -m build
python -m twine check --strict dist/*.whl dist/*.tar.gz
```

The first test command works offline after installation. The second exercises
the public FEW models and uses FEW's normal data cache. Allow about 5 GB for
the Kerr amplitude table. Open `site/index.html` to read the built manual.
For a local server with full-text search, run `python -m mkdocs serve`.
The wheel and source archive are written to `dist/`. To check the wheel
independently, install it in an empty environment outside the checkout, run
`python -m pip check`, then run the first render in [Getting started](index.md).

## GitHub release

The version is defined in `src/fewview/_version.py`. Push the reviewed commit,
then tag it with that version prefixed by `v` (for example, `v0.1.0`).
The release workflow checks the tag against the wheel metadata, runs the
tests, builds the manual and attaches the wheel, source archive, HTML manual
and logo bundle to a draft GitHub release. Review the draft and publish it on GitHub.
A PyPI upload is a separate maintainer action.
