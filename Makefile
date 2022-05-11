# Makefile
#
# Build scripts for testing, installation, and package maintenance.
#
# @author Rahul Dhodapkar

.PHONY: build upload-test install-local install-test

install-local:
	pip install --no-deps --force-reinstall dist/*.whl

build:
	pip install --upgrade build
	python -m build

upload-test:
	pip install --upgrade twine
	python -m twine upload --repository testpypi dist/* --verbose

install-test:
	pip install --index-url https://test.pypi.org/simple/ --no-deps ampnet

