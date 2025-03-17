#!/bin/bash

# List of dimensions to generate bases for
dims=(16 32 48 64)
dists=(uniform qary ntrulike)

for dim in "${dims[@]}"; do
    echo "Generating random basis for n = ${dim}"
    for dist in "${dists[@]}"; do
        python generate_basis.py -d "${dim}" --distribution "${dist}" --train-samples 10000 --val-samples 1000 --test-samples 1000
    done
done

dims=(96 128)
dists=(uniform qary)

for dim in "${dims[@]}"; do
    echo "Generating random basis for n = ${dim}"
    for dist in "${dists[@]}"; do
        python generate_basis.py -d "${dim}" --distribution "${dist}" --train-samples 10000 --val-samples 1000 --test-samples 1000
    done
done
