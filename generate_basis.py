from functools import partial
import multiprocessing as mp
from pathlib import Path

from fpylll import IntegerMatrix, SVP, LLL
import numpy as np
from scipy.linalg import expm
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

def generate_uniform(n, low=0, high=100):
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

def generate_ajtai(n, sigma=10):
    """
    Inspired by Ajtai (1996):
    1) Sample B' with i.i.d. entries from {0, 1, ..., q-1}/q.
    2) Return the dual basis B = (B')^(-T) = (B'^-1)^T.

    Note: We assume B' is invertible. If it is singular, we resample.
    """
    while True:
        # Sample integer matrix in [0, q-1], then scale by 1/q
        B_prime_int = np.random.randint(0, q, size=(n, n))
        B_prime = B_prime_int.astype(float) / q
        
        # Check invertibility (condition number not too large)
        if np.linalg.cond(B_prime) < 1 / np.finfo(float).eps:
            break

    # Dual basis is inverse-transpose
    B = np.linalg.inv(B_prime).T
    return B

##################################################

num_train_samples = 10_000
num_val_samples = 4_000
num_test_samples = 4_000

def func(_, n, distribution):
    np.random.seed()
    if distribution == 'uniform':
        basis = generate_uniform(n)
    elif distribution == 'ajtai':
        basis = generate_ajtai(n)
    else:
        raise ValueError("Invalid distribution type")
    
    shortest_vector = svp(basis)
    lll_reduced_basis = lll_reduction(basis)
    defect = log_defect(lll_reduced_basis)

    # Store original basis, shortest vector, reduced basis and log orthogonality defect together.
    return {"basis": basis, "shortest_vector": shortest_vector, "lll_reduced_basis": lll_reduced_basis, "lll_log_defect": defect}


# Create the output directory using pathlib
output_dir = Path("random_bases")
output_dir.mkdir(parents=True, exist_ok=True)

processes = 20

for n in [4, 6, 8, 10, 12, 14, 16]:
    for distribution in ["uniform"]:
        worker = partial(func, n=n, distribution=distribution)

        with mp.Pool(processes) as pool:
            train_data = list(tqdm(pool.imap(worker, range(num_train_samples)), total=num_train_samples))
        np.save(f"random_bases/train_dim_{n}_type_{distribution}.npy", train_data)
        print(f"Saved random_bases/train_dim_{n}_type_{distribution}.npy")

        with mp.Pool(processes) as pool:
            val_data = list(tqdm(pool.imap(worker, range(num_val_samples)), total=num_val_samples))
        np.save(f"random_bases/val_dim_{n}_type_{distribution}.npy", val_data)
        print(f"Saved random_bases/val_dim_{n}_type_{distribution}.npy")

        with mp.Pool(processes) as pool:
            test_data = list(tqdm(pool.imap(worker, range(num_test_samples)), total=num_test_samples))
        np.save(f"random_bases/test_dim_{n}_type_{distribution}.npy", test_data)
        print(f"Saved random_bases/test_dim_{n}_type_{distribution}.npy")
