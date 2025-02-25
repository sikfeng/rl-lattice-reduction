from dataclasses import dataclass
from itertools import takewhile
import time
from typing import Any, Dict, Tuple, Optional

from fpylll import BKZ, FPLLL, GSO, IntegerMatrix, LLL, SVP
import numpy as np
import torch


FPLLL.set_precision(1000)


def compute_log_defect(basis: IntegerMatrix) -> float:
    """Compute the orthogonality defect of a given basis."""
    m = GSO.Mat(basis, flags=GSO.INT_GRAM | GSO.ROW_EXPO, float_type="mpfr")
    m.update_gso()
    log_det = m.get_log_det(0, basis.nrows) / 2  # log(determinant)
    log_prod_norms = sum(np.log(v.norm()) for v in basis)
    log_defect = log_prod_norms - log_det
    return log_defect

class BKZReduction: # alternative implementation
    def __init__(self, A: IntegerMatrix):
        if not isinstance(A, IntegerMatrix):
            raise TypeError(f"Matrix must be IntegerMatrix but got {type(A)}")

        self.A = A

    def __call__(self, kappa, block_size):
        """Perform LLL reduction on a block.

        :param kappa: row index
        :param block_size: an integer >= 2

        """
        block_matrix = self.A.submatrix(range(kappa, kappa + block_size), range(self.A.ncols))
        
        LLL.reduction(block_matrix)

        for i in range(block_size):
            for j in range(self.A.ncols):
                self.A[kappa + i, j] = block_matrix[i, j]

        return self.A


class BKZReduction2:
    def __init__(self, A):
        if not isinstance(A, IntegerMatrix):
            raise TypeError(f"Matrix must be IntegerMatrix but got {type(A)}")

        self.A = A
        self.m = GSO.Mat(A, flags=GSO.INT_GRAM | GSO.ROW_EXPO, float_type="mpfr") # need GSO.INT_GRAM and float_type="mpfr" or the precision will cry
        self.lll_obj = LLL.Reduction(self.m)
        self.m.update_gso() # Update Gram Schmidt coefficients
        self.auto_abort = BKZ.AutoAbort(self.m, self.A.nrows) # heuristic check if BKZ can be terminated. Checks if the slope of the basis hasn't decreased in a while. 

    def __call__(self, kappa, block_size):
        """Perform LLL reduction on a block.

        :param kappa: row index
        :param block_size: an integer >= 2

        """
        self.lll_obj(kappa, kappa, kappa + block_size)


@dataclass
class BKZEnvConfig:
    max_steps: Optional[int] = None  # If not provided, will be set to 2 * basis_dim
    min_block_size: int = None # inclusive
    max_block_size: int = None # inclusive
    time_limit: float = 300
    basis_dim: int = None
    action_history_size: int = 10
    
    time_penalty_weight: float = 1.0

    def __post_init__(self):
        if self.max_steps is None:
            self.max_steps = 2 * self.basis_dim

        self.actions_n = 0
        for i in range(self.basis_dim):
            for j in range(self.min_block_size, self.max_block_size + 1):
                if i + j > self.basis_dim:
                    continue
                self.actions_n += 1

        '''def compute_actions_n(basis_dim: int, min_block_size: int, max_block_size: int) -> int:
            # Ensure that max_block_size does not exceed basis_dim.
            U = min(max_block_size, basis_dim)
            L = min_block_size
            # Number of allowed block sizes
            n_sizes = U - L + 1
            # Closed form: actions_n = ((U-L+1) * (2*(basis_dim+1) - (U+L))) // 2
            actions_n = (n_sizes * (2 * (basis_dim + 1) - (U + L))) // 2
            return actions_n'''

class BKZEnvironment:
    def __init__(self, config: BKZEnvConfig):
        super().__init__()
        self.config = config

        # Environment states
        self.optimal_length = None
        self.initial_length = None
        self.best_achieved = None
        self.action_history = []

        # Initialize FPLLL objects
        self.reset()

    def _get_observation(self) -> Dict[str, torch.Tensor]:
        basis = np.zeros((self.config.basis_dim, self.config.basis_dim))
        basis = torch.tensor(self.M.to_matrix(basis), dtype=torch.float32)

        last_actions = torch.tensor(self.action_history[-self.config.action_history_size:], dtype=torch.float32)
        history = torch.cat([torch.full((self.config.action_history_size - last_actions.size(0),), -1.0), last_actions])

        return {
            "basis": basis,
            "action_history": history
        }

    def _get_info(self) -> Dict[str, Any]:
        return {"log_defect": compute_log_defect(self.M)}

    def reset(self, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        if options is not None and 'basis' in options and 'shortest_vector' in options:
            # Use provided basis and shortest vector
            self.M = IntegerMatrix.from_matrix(options['basis'].int().tolist())
            self.shortest_vector = options['shortest_vector']
        else:
            # Generate new random basis
            self.M, self.shortest_vector = self._generate_random_basis("uniform")

        if type(self.shortest_vector) is torch.Tensor:
            self.shortest_vector = self.shortest_vector.cpu()
        self.optimal_length = np.linalg.norm(self.shortest_vector)
        
        self.bkz = BKZReduction(self.M)
        self.initial_length = min(v.norm() for v in self.M)
        self.best_achieved = self.initial_length
        self.start_time = time.time()
        self.action_history = []

        return self._get_observation(), self._get_info()

    def action_to_block(self, action: int) -> Tuple[int, int]:
        """Convert single action index to block size and start position"""
        '''
        n_block_actions = self.config.max_block_size - self.config.min_block_size + 1

        # Decode action index
        block_action = action % n_block_actions
        row_index_action = action // n_block_actions

        block_size = block_action + self.config.min_block_size
        start_pos = row_index_action

        return start_pos, block_size
        '''
        assert action < self.config.actions_n, f"Action {action} provided, but only {self.config.actions_n} actions available!"

        for i in range(self.config.basis_dim):
            for j in range(self.config.min_block_size, self.config.max_block_size + 1):
                if i + j > self.config.basis_dim:
                    continue
                if action == 0:
                    return i, j
                action -= 1

    def step(self, action: int) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict[str, Any]]:
        # Decode action index
        start_pos, block_size = self.action_to_block(action)

        # Execute reduction step on selected block
        self.bkz(start_pos, min(block_size, self.config.basis_dim - start_pos)) #################

        # Update state
        self.action_history.append(action)

        # Check termination conditions
        self.terminated = self._check_termination()
        self.truncated = self._check_truncation()

        return self._get_observation(), self._compute_reward(), self._check_termination(), self._check_truncation(), self._get_info()

    def _compute_reward(self) -> float:
        reward = 0.0
        
        time_penalty = self.config.time_penalty_weight * np.log(1 + (time.time() - self.start_time))
        reward -= time_penalty

        if (len(self.action_history) > 1) and (self.action_history[-1] == self.action_history[-2]):
            repeats = sum(1 for _ in takewhile(lambda x: x == self.action_history[-1], reversed(self.action_history)))
            repeat_action_penalty = 10 ** repeats
            reward -= repeat_action_penalty

        log_orthogonality_defect = compute_log_defect(self.M)
        reward -= log_orthogonality_defect

        return reward

    def _check_termination(self):
        """Check if episode has terminated"""
        #return False
        return (
            (self.best_achieved - self.optimal_length < 1e-6)
            or (len(self.action_history) >= self.config.action_history_size and all(x == self.action_history[-1] for x in self.action_history[-self.config.action_history_size:]))
            #or self.bkz.auto_abort.test_abort()
        )

    def _check_truncation(self):
        """Check if episode has been truncated due to time limit"""
        return (len(self.action_history) >= self.config.max_steps) or (time.time() - self.start_time >= self.config.time_limit)

    ### random basis generation ###

    def _generate_random_basis(self, distribution="uniform"):
        if distribution == "uniform":
            return self._generate_random_basis_uniform()
        else:
            raise ValueError("Invalid distribution \"%s\" provided!", distribution)
    
    def _generate_random_basis_uniform(self):
        random_basis = IntegerMatrix.random(self.config.basis_dim, "uniform", bits=10)
        random_shortest = SVP.shortest_vector(random_basis)
        return random_basis, random_shortest
