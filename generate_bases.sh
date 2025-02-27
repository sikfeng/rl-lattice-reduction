#!/bin/bash

# List of dimensions to generate bases for
dims=(4 6 8 12 16)

for dim in "${dims[@]}"; do
    echo "Generating random basis for n = ${dim}"
    python generate_basis.py -d "${dim}"
done