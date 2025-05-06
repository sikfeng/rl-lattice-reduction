import argparse
from functools import partial
import math
import multiprocessing as mp
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple

import fpylll
from fpylll import GSO, IntegerMatrix, LLL, ReductionError, SVP
import numpy as np
import torch
from tqdm import tqdm


def svp(basis: np.ndarray) -> np.ndarray:
    """
    Generate the shortest vector in the lattice defined by the given basis matrix using the SVP (Shortest Vector Problem) solver from fpylll library.

    Parameters:
        basis (numpy.ndarray): The basis matrix of the lattice, represented as a 2-dimensional numpy array.

    Returns:
        numpy.ndarray: The shortest vector in the lattice, represented as a 1-dimensional numpy array.
    """
    basis_fpylll = IntegerMatrix.from_matrix(basis)
    return np.array(SVP.shortest_vector(basis_fpylll, method="proved"))


def lll_reduction(basis: np.ndarray) -> np.ndarray:
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


def log_defect(basis: np.ndarray) -> float:
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


def gaussian_heuristic(basis: np.ndarray) -> float:
    """
    Calculate the length of the shortest vector predicted by the Gaussian Heuristic
    Parameters:
        basis (np.ndarray): The input basis matrix as a 2-dimensional numpy array.
    Returns:
        float: The length of the shortest vector predicted by the Gaussian Heuristic.
    """
    basis_ = np.array(basis, dtype=float)
    _, R = np.linalg.qr(basis_)
    diag = np.abs(np.diagonal(R, axis1=-2, axis2=-1))
    n = diag.shape[0]

    log_gh = np.sum(np.log(diag)) / n - np.log(np.pi) / 2 - math.lgamma(n / 2 + 1) / n
    return np.exp(log_gh)


def generate_uniform(n: int, b: int) -> Tuple[List[List[int]], int]:
    basis = IntegerMatrix.random(n, "uniform", bits=b)
    np_basis = np.zeros((n, n), dtype=int).tolist()
    basis.to_matrix(np_basis)
    return np_basis, -1


def generate_qary(n: int, q: int, k: int) -> Tuple[List[List[int]], int]:
    assert n % 2 == 0

    basis = IntegerMatrix.random(n, "qary", q=q, k=k)
    np_basis = np.zeros((n, n), dtype=int).tolist()
    basis.to_matrix(np_basis)
    return np_basis, -1


def generate_ntrulike(n: int, q: int) -> Tuple[List[List[int]], int]:
    assert n % 2 == 0

    basis = IntegerMatrix.random(n // 2, "ntrulike", q=q)
    np_basis = np.zeros((n, n), dtype=int).tolist()
    basis.to_matrix(np_basis)
    return np_basis, -1


def generate_knapsack_lo(n: int, b: int) -> Tuple[List[List[int]], int]:
    # return random knapsack basis and the shortest basis length
    # (assuming that it is the basis representing the subset sum for the last row)
    basis = IntegerMatrix.random(n - 1, "intrel", bits=b)
    basis_ = np.zeros((n, n), dtype=int).tolist()
    basis.to_matrix(basis_)

    subset_size = random.randint(1, n - 2)
    elements = random.sample(range(n - 1), subset_size)

    for e in elements:
        basis_[n - 1][0] += basis_[e][0]

    return basis_, np.sqrt(subset_size)


def generate_knapsack_clos(n: int, b: int) -> Tuple[List[List[int]], int]:
    # return random knapsack basis and the shortest basis length
    # (assuming that it is the basis representing the subset sum for the last row)
    basis = IntegerMatrix.random(n - 1, "intrel", bits=b)
    basis_ = np.zeros((n, n), dtype=int).tolist()
    basis.to_matrix(basis_)

    subset_size = random.randint(1, n - 2)
    elements = random.sample(range(n - 1), subset_size)

    for e in elements:
        basis_[n - 1][0] += basis_[e][0]
    for i in range(n - 2):
        basis_[i][i + 1] = 2
        basis_[n - 1][i + 1] = 1

    return basis_, np.sqrt(subset_size)


def func(_, n: int, distribution: str) -> Dict[str, Any]:
    while True:
        try:
            # tgt is -1 if distribution is not knapsack
            if distribution == "uniform":
                basis, tgt = generate_uniform(n, b=6)
            elif distribution == "qary":
                basis, tgt = generate_qary(n, q=11887, k=3)
            elif distribution == "ntrulike":
                basis, tgt = generate_ntrulike(n, q=11887)
            elif distribution == "knapsack_lo":
                basis, tgt = generate_knapsack_lo(n, b=n)
            elif distribution == "knapsack_clos":
                basis, tgt = generate_knapsack_clos(n, b=n)
            else:
                raise ValueError(f"Unknown distribution '{distribution}'")

            # Calculate the length of the shortest basis vector in the original basis
            original_basis_vector_lengths = np.linalg.norm(
                np.array(basis, dtype=float), axis=1
            )
            shortest_original_basis_vector_length = np.min(
                original_basis_vector_lengths
            )

            # Calculate Gaussian heuristic
            gh = gaussian_heuristic(basis)

            if gh == 0:
                raise ValueError("Gaussian heuristic is zero.")

            # Store all the data
            return {
                "basis": basis,
                "shortest_original_basis_vector_length": shortest_original_basis_vector_length,
                "gaussian_heuristic": gh,
                "target_length": tgt / gh if tgt != -1 else -1,
            }
        except (ReductionError, RuntimeError) as e:
            continue


def main():
    parser = argparse.ArgumentParser(description="Generate random basis")
    parser.add_argument("-d", "--dim", type=int, default=4)
    parser.add_argument("--samples", type=int, default=1_000)
    parser.add_argument("--processes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)

    dist_args = parser.add_mutually_exclusive_group(required=True)
    dist_args.add_argument(
        "--uniform",
        action="store_const",
        const="uniform",
        dest="dist",
        help="Use a uniform distribution.",
    )
    dist_args.add_argument(
        "--qary",
        action="store_const",
        const="qary",
        dest="dist",
        help="Use a q-ary distribution.",
    )
    dist_args.add_argument(
        "--ntrulike",
        action="store_const",
        const="ntrulike",
        dest="dist",
        help="Use an NTRU-like distribution.",
    )
    dist_args.add_argument(
        "--knapsack-lo",
        action="store_const",
        const="knapsack_lo",
        dest="dist",
        help="Use a knapsack distribution (LO85).",
    )
    dist_args.add_argument(
        "--knapsack-clos",
        action="store_const",
        const="knapsack_clos",
        dest="dist",
        help="Use a knapsack distribution (CLOS91).",
    )

    args = parser.parse_args()

    # Create the output directory using pathlib
    output_dir = Path("random_bases")
    output_dir.mkdir(parents=True, exist_ok=True)

    worker = partial(func, n=args.dim, distribution=args.dist)
    np.random.seed(args.seed)
    with mp.Pool(args.processes) as pool:
        data = list(
            tqdm(
                pool.imap(worker, range(args.samples)),
                desc=f"Generating dim {args.dim}, {args.dist} distribution",
                dynamic_ncols=True,
                total=args.samples,
            )
        )

    filename = f"random_bases/dim_{args.dim}_type_{args.dist}.npy"
    np.save(filename, data)
    print(f"Saved {filename}")


if __name__ == "__main__":
    main()
