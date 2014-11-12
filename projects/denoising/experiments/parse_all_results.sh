#!/bin/bash
python parse_results.py results/snp_with_targets.pickle
python parse_results.py results/gaussian_with_targets.pickle
python parse_results.py results/snp_without_targets.pickle
python parse_results.py results/gaussian_without_targets.pickle