#!venv/bin/python
# -*- coding: utf-8 -*-
import pickle

with open('snp_with_targets.pickle', 'r') as f:
    results = pickle.load(f)
