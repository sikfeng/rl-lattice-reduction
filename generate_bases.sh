#!/bin/bash

# List of dimensions to generate bases for
dims=(16 32 64 96 128)
dists=(uniform qary ntrulike)

for dim in "${dims[@]}"; do
    for dist in "${dists[@]}"; do
        echo "Generating random basis for n = ${dim}"
        python generate_basis.py -d "${dim}" --distribution "${dist}"
    done
done
