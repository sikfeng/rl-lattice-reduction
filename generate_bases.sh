#!/bin/bash

# List of dimensions to generate bases for
dims=(16 32 64)
dists=(uniform qary ntrulike)

for dim in "${dims[@]}"; do
    echo "Generating random basis for n = ${dim}"
    for dist in "${dists[@]}"; do
        python generate_basis.py -d "${dim}" --distribution "${dist}"
    done
done

dims=(96 128)
dists=(qary ntrulike)

for dim in "${dims[@]}"; do
    echo "Generating random basis for n = ${dim}"
    python generate_basis.py -d "${dim}" --distribution uniform --train-samples 1000 --val-samples 400 --test-samples 400
    for dist in "${dists[@]}"; do
        python generate_basis.py -d "${dim}" --distribution "${dist}"
    done
done
