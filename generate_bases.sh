#!/bin/bash

# List of dimensions to generate bases for
dims=(40 48 56 64 96)
dists=(uniform qary ntrulike)

for dim in "${dims[@]}"; do
    echo "Generating random basis for n = ${dim}"
    for dist in "${dists[@]}"; do
        python generate_basis.py -d "${dim}" --distribution "${dist}" --samples 10000
    done
done
