#!/usr/bin/env bash

# Trust bundled notebooks so pre-rendered outputs display without re-running cells.
# Must run after the signing key is generated (it is created on first use, so we
# trigger it by running 'jupyter trust' which initialises the keystore).
jupyter trust /notebooks/*.ipynb

jupyter notebook "$@"
