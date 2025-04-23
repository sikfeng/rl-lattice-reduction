import argparse
from functools import partial
import math
import multiprocessing as mp
from pathlib import Path
import random

import fpylll
from fpylll import GSO, IntegerMatrix, LLL, ReductionError, SVP
import numpy as np
from tqdm import tqdm


def svp(basis):
    """
    Generate the shortest vector in the lattice defined by the given basis matrix using the SVP (Shortest Vector Problem) solver from fpylll library.

    Parameters:
        basis (numpy.ndarray): The basis matrix of the lattice, represented as a 2-dimensional numpy array.

    Returns:
        numpy.ndarray: The shortest vector in the lattice, represented as a 1-dimensional numpy array.
    """
    basis_fpylll = IntegerMatrix.from_matrix(basis)
    return np.array(SVP.shortest_vector(basis_fpylll, method='proved'))


def lll_reduction(basis):
    """
    Apply the LLL (Lenstra-Lenstra-Lovász) reduction algorithm to a basis matrix.

    Parameters:
        basis (numpy.ndarray): The input basis matrix as a 2-dimensional numpy array.

    Returns:
        numpy.ndarray: The LLL-reduced basis matrix as a 2-dimensional numpy array.
    """
    basis_fpylll = IntegerMatrix.from_matrix(basis)
    LLL.reduction(basis_fpylll)
    np_basis = np.zeros((len(basis[0]), len(basis[0])), dtype=int)
    basis_fpylll.to_matrix(np_basis)
    return np_basis


def log_defect(basis):
    """
    Compute the logarithm of the orthogonality defect of a given basis matrix.

    Parameters:
        basis (numpy.ndarray): The input basis matrix as a 2-dimensional numpy array.

    Returns:
        float: The logarithm of the orthogonality defect of the basis matrix.
        If the determinant of the basis matrix is zero, returns infinity.
    """
    # Compute the product of the Euclidean norms of the basis vectors.
    log_prod_norms = np.sum(np.log(np.linalg.norm(basis, axis=1)))
    # Compute the absolute value of the determinant (i.e. the volume).
    det = abs(np.linalg.det(basis))
    if det == 0:
        raise ValueError("Determinant is zero")
    return log_prod_norms - np.log(det)


def gaussian_heuristic(basis: np.ndarray):
    """
    Calculate the length of the shortest vector predicted by the Gaussian Heuristic
    Parameters:
        basis (np.ndarray): The input basis matrix as a 2-dimensional numpy array.
    Returns:
        float: The length of the shortest vector predicted by the Gaussian Heuristic.
    """
    B = IntegerMatrix.from_matrix(basis)
    M = GSO.Mat(B)
    M.update_gso()

    # Get the squared norms of the Gram-Schmidt vectors
    gs_norms_squared = [M.get_r(i, i) for i in range(M.d)]

    # Calculate the Gaussian Heuristic
    gh_squared = fpylll.util.gaussian_heuristic(gs_norms_squared)

    return math.sqrt(gh_squared)


def generate_uniform(n, b):
    basis = IntegerMatrix.random(n, "uniform", bits=b)
    np_basis = np.zeros((n, n), dtype=int)
    basis.to_matrix(np_basis)
    return np_basis


def generate_qary(n, q, k):
    assert n % 2 == 0

    basis = IntegerMatrix.random(n, "qary", q=q, k=k)
    np_basis = np.zeros((n, n), dtype=int)
    basis.to_matrix(np_basis)
    return np_basis


def generate_ntrulike(n, q):
    assert n % 2 == 0

    basis = IntegerMatrix.random(n // 2, "ntrulike", q=q)
    np_basis = np.zeros((n, n), dtype=int)
    basis.to_matrix(np_basis)
    return np_basis


def generate_knapsack(n, b):
    basis = IntegerMatrix.random(n - 1, "intrel", bits=b)
    np_basis = np.zeros((n, n), dtype=int)
    basis.to_matrix(np_basis)

    subset_size = random.randint(1, n - 2)
    elements = random.sample(range(n - 1), subset_size)

    for e in elements:
        np_basis[n - 1][0] += np_basis[e][0]

    return np_basis


def func(_, n, distribution):
    distributions = ["uniform", "qary", "ntrulike", "knapsack"]
    assert distribution in distributions, "Invalid distribution type."

    while True:
        try:
            if distribution == "uniform":
                basis = generate_uniform(n, b=6)
            elif distribution == "qary":
                basis = generate_qary(n, q=11887, k=3)
            elif distribution == "ntrulike":
                basis = generate_ntrulike(n, q=11887)
            elif distribution == "knapsack":
                basis = generate_knapsack(n, b=13)

            # Calculate the length of the shortest basis vector in the original basis
            original_basis_vector_lengths = np.linalg.norm(basis, axis=1)
            shortest_original_basis_vector_length = np.min(
                original_basis_vector_lengths)

            # Calculate log defect of the original basis
            original_log_defect = log_defect(basis)

            # Calculate LLL reduced basis
            lll_reduced_basis = lll_reduction(basis)

            # Calculate log defect of the LLL reduced basis
            defect = log_defect(lll_reduced_basis)

            # Calculate length of the shortest basis vector in the LLL reduced basis
            lll_basis_vector_lengths = np.linalg.norm(
                lll_reduced_basis, axis=1)
            shortest_lll_basis_vector_length = np.min(lll_basis_vector_lengths)

            # Calculate Gaussian heuristic
            gh = gaussian_heuristic(basis)

            # Store all the data
            return {
                "basis": basis,
                "shortest_original_basis_vector_length": shortest_original_basis_vector_length,
                "original_log_defect": original_log_defect,
                "lll_log_defect": defect,
                "shortest_lll_basis_vector_length": shortest_lll_basis_vector_length,
                "gaussian_heuristic": gh,
            }
        except (ReductionError, ValueError, RuntimeError) as e:
            continue


def main():
    distributions = ["uniform", "qary", "ntrulike", "knapsack"]
    parser = argparse.ArgumentParser(description="Generate random basis")
    parser.add_argument("-d", "--dim", type=int, default=4)
    parser.add_argument("--distribution", type=str, choices=distributions)
    parser.add_argument("--samples", type=int, default=1_000)
    parser.add_argument("--processes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    # Create the output directory using pathlib
    output_dir = Path("random_bases")
    output_dir.mkdir(parents=True, exist_ok=True)

    worker = partial(func, n=args.dim, distribution=args.distribution)
    np.random.seed(args.seed)
    with mp.Pool(args.processes) as pool:
        data = list(tqdm(
            pool.imap(worker,range(args.samples)),
            desc=f"Generating dim {args.dim}, {args.distribution} distribution",
            dynamic_ncols=True,
            total=args.samples))

    filename = f"random_bases/dim_{args.dim}_type_{args.distribution}.npy"
    np.save(filename, data)
    print(f"Saved {filename}")


if __name__ == "__main__":
    main()
