from dataclasses import dataclass
import math
from time import process_time
from typing import Any, Dict, Optional, Tuple

import fpylll
from fpylll import BKZ, Enumeration, EnumerationError, FPLLL, GSO, IntegerMatrix, LLL
from fpylll.tools.bkz_stats import dummy_tracer, normalize_tracer, Tracer
from fpylll.util import adjust_radius_to_gh_bound
import numpy as np
import torch

from generate_basis import func as generate_random_basis

FPLLL.set_precision(1000)


# adapted from https://github.com/fplll/fpylll/blob/master/src/fpylll/algorithms/bkz.py
class BKZReduction(object):
    """
    An implementation of the BKZ algorithm in Python.

    This class has feature parity with the C++ implementation in fplll's core.  Additionally, this
    implementation collects some additional statistics.  Hence, it should provide a good basis for
    implementing variants of this algorithm.
    """

    def __init__(self, A):
        """Construct a new instance of the BKZ algorithm.

        :param A: an integer matrix, a GSO object or an LLL object

        """
        if isinstance(A, GSO.Mat):
            L = None
            M = A
            A = M.B
        elif isinstance(A, LLL.Reduction):
            L = A
            M = L.M
            A = M.B
        elif isinstance(A, IntegerMatrix):
            L = None
            M = None
            A = A
        else:
            raise TypeError(
                "Matrix must be IntegerMatrix but got type '%s'" % type(A))

        if M is None and L is None:
            # run LLL first, but only if a matrix was passed
            LLL.reduction(A)

        self.A = A
        if M is None:
            self.M = GSO.Mat(A, flags=GSO.ROW_EXPO)
        else:
            self.M = M
        if L is None:
            self.lll_obj = LLL.Reduction(self.M, flags=LLL.DEFAULT)
        else:
            self.lll_obj = L

    def __call__(self, params, min_row=0, max_row=-1, tracer=False):
        """Run the BKZ algorithm with parameters `param`.

        :param params: BKZ parameters
        :param min_row: start processing in this row
        :param max_row: stop processing in this row (exclusive)
        :param tracer: see ``normalize_tracer`` for accepted values


        TESTS::

            >>> from fpylll import *
            >>> A = IntegerMatrix.random(60, "qary", k=30, q=127)
            >>> from fpylll.algorithms.bkz import BKZReduction
            >>> bkz = BKZReduction(A)
            >>> _ = bkz(BKZ.EasyParam(10), tracer=True); bkz.trace is None
            False
            >>> _ = bkz(BKZ.EasyParam(10), tracer=False); bkz.trace is None
            True

        """

        tracer = normalize_tracer(tracer)

        try:
            label = params["name"]
        except KeyError:
            label = "bkz"

        if not isinstance(tracer, Tracer):
            tracer = tracer(
                self,
                root_label=label,
                verbosity=params.flags & BKZ.VERBOSE,
                start_clocks=True,
                max_depth=2,
            )

        if params.flags & BKZ.AUTO_ABORT:
            auto_abort = BKZ.AutoAbort(self.M, self.A.nrows)

        cputime_start = process_time()

        with tracer.context("lll"):
            self.lll_obj()

        i = 0
        while True:
            with tracer.context("tour", i, dump_gso=params.flags & BKZ.DUMP_GSO):
                clean = self.tour(params, min_row, max_row, tracer)
            i += 1
            if clean or params.block_size >= self.A.nrows:
                break
            if (params.flags & BKZ.AUTO_ABORT) and auto_abort.test_abort():
                break
            if (params.flags & BKZ.MAX_LOOPS) and i >= params.max_loops:
                break
            if (params.flags & BKZ.MAX_TIME) and process_time() - cputime_start >= params.max_time:
                break

        tracer.exit()
        try:
            self.trace = tracer.trace
        except AttributeError:
            self.trace = None
        return clean

    def tour(self, params, min_row=0, max_row=-1, tracer=dummy_tracer):
        """One BKZ loop over all indices.

        :param params: BKZ parameters
        :param min_row: start index ≥ 0
        :param max_row: last index ≤ n

        :returns: ``True`` if no change was made and ``False`` otherwise
        """
        if max_row == -1:
            max_row = self.A.nrows

        clean = True

        for kappa in range(min_row, max_row - 1):
            block_size = min(params.block_size, max_row - kappa)
            clean &= self.svp_reduction(kappa, block_size, params, tracer)

        self.lll_obj.size_reduction(
            max(0, max_row - 1), max_row, max(0, max_row - 2))
        return clean

    def svp_preprocessing(self, kappa, block_size, params, tracer):
        """Perform preprocessing for calling the SVP oracle

        :param kappa: current index
        :param params: BKZ parameters
        :param block_size: block size
        :param tracer: object for maintaining statistics

        :returns: ``True`` if no change was made and ``False`` otherwise

        .. note::

            ``block_size`` may be smaller than ``params.block_size`` for the last blocks.

        """
        clean = True

        lll_start = kappa if params.flags & BKZ.BOUNDED_LLL else 0
        with tracer.context("lll"):
            self.lll_obj(lll_start, lll_start, kappa + block_size)
            if self.lll_obj.nswaps > 0:
                clean = False

        return clean

    def svp_call(self, kappa, block_size, params, tracer=dummy_tracer):
        """Call SVP oracle

        :param kappa: current index
        :param params: BKZ parameters
        :param block_size: block size
        :param tracer: object for maintaining statistics

        :returns: Coordinates of SVP solution or ``None`` if none was found.

        ..  note::

            ``block_size`` may be smaller than ``params.block_size`` for the last blocks.
        """
        max_dist, expo = self.M.get_r_exp(kappa, kappa)
        delta_max_dist = self.lll_obj.delta * max_dist

        if params.flags & BKZ.GH_BND:
            root_det = self.M.get_root_det(kappa, kappa + block_size)
            max_dist, expo = adjust_radius_to_gh_bound(
                max_dist, expo, block_size, root_det, params.gh_factor
            )

        try:
            enum_obj = Enumeration(self.M)
            with tracer.context("enumeration", enum_obj=enum_obj, probability=1.0):
                max_dist, solution = enum_obj.enumerate(kappa, kappa + block_size, max_dist, expo)[
                    0
                ]

        except EnumerationError as msg:
            if params.flags & BKZ.GH_BND:
                return None
            else:
                raise EnumerationError(msg)

        if max_dist >= delta_max_dist * (1 << expo):
            return None
        else:
            return solution

    def svp_postprocessing(self, kappa, block_size, solution, tracer=dummy_tracer):
        """Insert SVP solution into basis. Note that this does not run LLL; instead,
           it resolves the linear dependencies internally.

        :param solution: coordinates of an SVP solution
        :param kappa: current index
        :param block_size: block size
        :param tracer: object for maintaining statistics

        :returns: ``True`` if no change was made and ``False`` otherwise

        ..  note :: postprocessing does not necessarily leave the GSO in a safe state.  You may
            need to call ``update_gso()`` afterwards.
        """
        if solution is None:
            return True

        j_nz = None

        for i in range(block_size)[::-1]:
            if abs(solution[i]) == 1:
                j_nz = i
                break

        if len([x for x in solution if x]) == 1:
            self.M.move_row(kappa + j_nz, kappa)

        elif j_nz is not None:
            with self.M.row_ops(kappa + j_nz, kappa + j_nz + 1):
                for i in range(block_size):
                    if solution[i] and i != j_nz:
                        self.M.row_addmul(
                            kappa + j_nz, kappa + i, solution[j_nz] * solution[i])

            self.M.move_row(kappa + j_nz, kappa)

        else:
            solution = list(solution)

            for i in range(block_size):
                if solution[i] < 0:
                    solution[i] = -solution[i]
                    self.M.negate_row(kappa + i)

            with self.M.row_ops(kappa, kappa + block_size):
                offset = 1
                while offset < block_size:
                    k = block_size - 1
                    while k - offset >= 0:
                        if solution[k] or solution[k - offset]:
                            if solution[k] < solution[k - offset]:
                                solution[k], solution[k - offset] = (
                                    solution[k - offset],
                                    solution[k],
                                )
                                self.M.swap_rows(kappa + k - offset, kappa + k)

                            while solution[k - offset]:
                                while solution[k - offset] <= solution[k]:
                                    solution[k] = solution[k] - \
                                        solution[k - offset]
                                    self.M.row_addmul(
                                        kappa + k - offset, kappa + k, 1)

                                solution[k], solution[k - offset] = (
                                    solution[k - offset],
                                    solution[k],
                                )
                                self.M.swap_rows(kappa + k - offset, kappa + k)
                        k -= 2 * offset
                    offset *= 2

            self.M.move_row(kappa + block_size - 1, kappa)

        return False

    def svp_reduction(self, kappa, block_size, params, tracer=dummy_tracer):
        """Find shortest vector in projected lattice of dimension ``block_size`` and insert into
        current basis.

        :param kappa: current index
        :param params: BKZ parameters
        :param block_size: block size
        :param tracer: object for maintaining statistics

        :returns: ``True`` if no change was made and ``False`` otherwise
        """
        clean = True
        with tracer.context("preprocessing"):
            clean_pre = self.svp_preprocessing(
                kappa, block_size, params, tracer)
        clean &= clean_pre

        solution = self.svp_call(kappa, block_size, params, tracer)

        with tracer.context("postprocessing"):
            clean_post = self.svp_postprocessing(
                kappa, block_size, solution, tracer)
        clean &= clean_post

        self.lll_obj.size_reduction(0, kappa + 1)
        return clean


@dataclass
class ReductionEnvConfig:
    # If not provided, will be set to 2 * basis_dim
    max_steps: Optional[int] = None
    max_block_size: int = None  # inclusive
    time_limit: float = 1.0
    max_basis_dim: int = None
    batch_size: int = 1

    time_penalty_weight: float = -1.0
    defect_reward_weight: float = 0.1
    length_reward_weight: float = 1.0

    distribution: str = None

    def __post_init__(self):
        if self.max_steps is None:
            self.max_steps = 2 * self.max_basis_dim

        self.actions_n = self.max_block_size


class ReductionEnvironment:
    def __init__(self, config: ReductionEnvConfig):
        super().__init__()
        self.config = config

    def _get_observation(self) -> Dict[str, torch.Tensor]:
        basis = np.zeros((self.config.max_basis_dim, self.config.max_basis_dim))
        basis = torch.tensor(self.basis.to_matrix(basis), dtype=torch.float32)

        last_action = torch.tensor(
            [self.action_history[-1]], dtype=torch.float32)

        return {
            "basis": basis,
            "last_action": last_action,
            "basis_dim": torch.tensor([self.basis.ncols])
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            "log_defect": self.log_defect_history[-1],
            "shortest_length": self.shortest_length_history[-1],
            "time": self.time_history[-1],
            "action_history": self.action_history
        }

    def reset(self, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        if options is None:
            basis_dim = self.config.max_basis_dim
            options = generate_random_basis(None, basis_dim, self.config.distribution)
            options["basis"] = torch.tensor(options["basis"])

        self.basis = IntegerMatrix.from_matrix(options["basis"].int().tolist())
        self.lll_log_defect = options["lll_log_defect"]
        self.gh = self.gaussian_heuristic(self.basis)

        self.tracer = normalize_tracer(True)
        if not isinstance(self.tracer, Tracer):
            self.tracer = self.tracer(
                self,
                root_label="bkz",
                verbosity=0,
                start_clocks=True,
                max_depth=2,
            )
        self.bkz = BKZReduction(self.basis)
        with self.tracer.context("lll"):
            self.bkz.lll_obj()

        self.action_history = [1]  # initial basis is always LLL reduced
        self.log_defect_history = []
        self.shortest_length_history = []
        self.time_history = []
        self._update_history()

        self.current_step = 0

        return self._get_observation(), self._get_info()

    def _action_to_block(self, action: int) -> int:
        """Convert single action index to block size"""
        assert action < self.config.actions_n, f"Action {action} provided, but only {self.config.actions_n} actions available!"

        return action + 1

    def _block_to_action(self, block_size: int) -> int:
        # _block_to_action should be the (both left and right) inverse of _action_to_block
        return block_size - 1

    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict[str, Any]]:
        if action == 0:
            self.terminated = True
            self.truncated = self._check_truncation()

            return self._get_observation(), torch.zeros(1), self.terminated, self.truncated, self._get_info()
        else:
            block_size = self._action_to_block(action)
            self.clean = self.bkz.tour(BKZ.EasyParam(
                block_size=block_size, max_loops=1, gh_factor=1.1, auto_abort=True), tracer=self.tracer)

            self.action_history.append(action)
            self._update_history()

            self.terminated = self._check_termination()
            self.truncated = self._check_truncation()
            self.current_step += 1

            return self._get_observation(), self._compute_reward(), self.terminated, self.truncated, self._get_info()

    def _update_history(self):
        self.time_history.append(process_time())
        self.log_defect_history.append(self.compute_log_defect(self.basis))
        self.shortest_length_history.append(
            min(v.norm() for v in self.basis) / self.gh)

    def _compute_reward(self) -> float:
        # Initialize reward components dictionary for better tracking
        rewards = {
            "time_penalty": 0.0,
            "defect_reward": 0.0,
            "length_reward": 0.0,
        }

        rewards["time_penalty"] = self.config.time_penalty_weight * \
            (self.time_history[-1] - self.time_history[-2])
        rewards["defect_reward"] = self.config.defect_reward_weight * \
            (self.log_defect_history[-2] - self.log_defect_history[-1])
        rewards["length_reward"] = self.config.length_reward_weight * \
            (self.shortest_length_history[-2] -
             self.shortest_length_history[-1])

        total_reward = sum(rewards.values())
        return total_reward

    def _check_termination(self):
        """Check if episode has terminated"""

        return False

    def _check_truncation(self):
        """Check if episode has been truncated due to time limit or exceeded max loops"""
        if len(self.action_history) >= self.config.max_steps:
            return True

        if process_time() - self.time_history[0] >= self.config.time_limit:
            return True

        return False

    @staticmethod
    def gaussian_heuristic(basis: IntegerMatrix) -> float:
        M = GSO.Mat(basis)
        M.update_gso()

        # Get the squared norms of the Gram-Schmidt vectors
        gs_norms_squared = [M.get_r(i, i) for i in range(M.d)]

        # Calculate the Gaussian Heuristic
        gh_squared = fpylll.util.gaussian_heuristic(gs_norms_squared)

        return math.sqrt(gh_squared)

    @staticmethod
    def compute_log_defect(basis: IntegerMatrix) -> float:
        """Compute the orthogonality defect of a given basis."""
        m = GSO.Mat(basis, flags=GSO.INT_GRAM |
                    GSO.ROW_EXPO, float_type="mpfr")
        m.update_gso()
        log_det = m.get_log_det(0, basis.nrows) / 2  # log(determinant)
        log_prod_norms = sum(np.log(v.norm()) for v in basis)
        log_defect = log_prod_norms - log_det
        return log_defect
