# Project generation tutorial

## Create distribution

On linux:

```bash
python3 setup.py sdist bdist_wheel
```

On Windows:

```bash
python setup.py sdist bdist_wheel
```

## Install it

To install FaceAnalyzer locally while pointing to the FaceAnalyzer path (useful to test and debug the library using another software). Assuming you are in the root path

```bash
python -m pip install --upgrade --force-reinstall -e .
```

To install QGraphViz locally without pushing it to pip do the following

```bash
python -m pip install --upgrade --force-reinstall dist/FaceAnalyzer-*.*.*-py3-none-any.whl
```

replace \* with the version you are using

## Publish it

You would need to install twine before pushing the file

pip install twine

python -m twine upload dist/\*

## Update README.md

Do all updates in misc/unprocessed_README.md, then preprocess it using [pp](https://github.com/CDSoft/pp) :

```bash
pp doc/unprocessed_README.md > README.md
```

pp will process the unprocessed_README.md and apply special macros to generate the final README.md file.
