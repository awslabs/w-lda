#!/bin/bash

set -exu

PYTHONPATH=./ python run.py $(cat examples/args/wikitext-103/mmd.txt) -gpu 0 -mx_it 11 -dirich_alpha 0.1 -hybrid True -latent_noise 0.2
