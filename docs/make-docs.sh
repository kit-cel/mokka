#!/bin/sh

rm -rf docs/api
sphinx-apidoc -o docs/api -E -M -T -d 3 -t docs/_templates src/
sphinx-build -b html docs/ _build/html
