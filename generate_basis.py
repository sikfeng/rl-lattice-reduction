import argparse
from functools import partial
import multiprocessing as mp
from pathlib import Path

from fpylll import IntegerMatrix, SVP, LLL
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
    return np.array(SVP.shortest_vector(basis_fpylll))

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
    np_basis = np.zeros((len(basis[0]), len(basis[0])))
    basis_fpylll.to_matrix(np_basis)
    return np_basis

def log_defect(basis):
    """
    Compute the logarithm of the orthogonality defect of a given basis matrix.

    Parameters:
        basis (numpy.ndarray): The input basis matrix as a 2-dimensional numpy array.

    Returns:
        float: The logarithm of the orthogonality defect of the basis matrix. If the determinant of the basis matrix is zero, returns infinity.
    """
    # Compute the product of the Euclidean norms of the basis vectors.
    log_prod_norms = np.sum(np.log(np.linalg.norm(basis, axis=1)))
    # Compute the absolute value of the determinant (i.e. the volume).
    det = abs(np.linalg.det(basis))
    return log_prod_norms - np.log(det) if det != 0 else float('inf')

def generate_uniform(n, low=-50, high=50):
    """
    Generate a uniform random integer matrix.
    
    This function generates a matrix of random integers within the specified range.
    
    Parameters:
        n (int): The number of rows and columns in the matrix.
        low (int, optional): The lower bound of the range (default is 0).
        high (int, optional): The upper bound of the range (default is 100).
    
    Returns:
        numpy.ndarray: A matrix of random integers within the specified range.
    """
    return np.random.randint(low=low, high=high, size=(n, n))


def func(_, n, distribution):
    np.random.seed()
    if distribution == 'uniform':
        basis = generate_uniform(n)
    else:
        raise ValueError("Invalid distribution type")
    
    shortest_vector = svp(basis)
    lll_reduced_basis = lll_reduction(basis)
    defect = log_defect(lll_reduced_basis)

    # Store original basis, shortest vector, reduced basis and log orthogonality defect together.
    return {"basis": basis, "shortest_vector": shortest_vector, "lll_reduced_basis": lll_reduced_basis, "lll_log_defect": defect}

def main():
    parser = argparse.ArgumentParser(description="Generate random basis")
    parser.add_argument("-d", "--dim", type=int, default=4)
    parser.add_argument("--distribution", type=str, default="uniform")
    parser.add_argument("--train_samples", type=int, default=10_000)
    parser.add_argument("--val_samples", type=int, default=4_000)
    parser.add_argument("--test_samples", type=int, default=4_000)
    parser.add_argument("--processes", type=int, default=20)
    args = parser.parse_args()

    # Create the output directory using pathlib
    output_dir = Path("random_bases")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_files = ['train', 'val', 'test']
    num_samples = [args.train_samples, args.val_samples, args.test_samples]

    worker = partial(func, n=args.dim, distribution=args.distribution)
    for data_type, num_sample in zip(data_files, num_samples):
        with mp.Pool(args.processes) as pool:
            data = list(tqdm(pool.imap(worker, range(num_sample)), desc=f"Generating {data_type}, dim {args.dim}, {args.distribution} distribution", dynamic_ncols=True, total=num_sample))
        np.save(f"random_bases/dim_{args.dim}_type_{args.distribution}_{data_type}.npy", data)
        print(f"Saved random_bases/dim_{args.dim}_type_{args.distribution}_{data_type}.npy")


if __name__ == "__main__":
    main()
