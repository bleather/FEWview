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

## Read the Docs

Read the Docs builds the [online manual](https://fewview.readthedocs.io/en/latest/)
from `main` using `.readthedocs.yaml`. To reproduce the build in a fresh
Python 3.12 environment, run `python -m pip install -r docs/requirements.txt`
and `python -m mkdocs build --strict`. FEW and the waveform data are only
needed to run the examples. The [build history](https://app.readthedocs.org/projects/fewview/builds/)
shows which commit is online.

## PyPI

Publishing a final GitHub release starts **Publish to PyPI**. The workflow
downloads the release wheel and source archive, verifies their SHA-256 digests
and uploads those same files to PyPI. The HTML manual and artwork stay on GitHub.

For the first upload, the package owner must add a pending GitHub publisher
in [PyPI account settings](https://pypi.org/manage/account/publishing/).
Use project `fewview`, owner `bleather`, repository `FEWview`, workflow
`publish-pypi.yml` and environment `pypi`. GitHub obtains a short-lived
publishing credential for each run.

For a release already published on GitHub, open
[Publish to PyPI](https://github.com/bleather/FEWview/actions/workflows/publish-pypi.yml),
select **Run workflow** on `main` and enter the existing tag (for example,
`v0.2.0`). After the upload, check the public installation in a fresh Python
3.12 environment outside the checkout:

```bash
python -m pip install fewview
python -m pip check
fewview-render --help
```
