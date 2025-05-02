from collections import defaultdict
from dataclasses import dataclass
import math
import multiprocessing as mp
from time import process_time
from typing import Any, Dict, Optional, Tuple

import fpylll
from fpylll import BKZ, Enumeration, EnumerationError, FPLLL, GSO, IntegerMatrix, LLL
from fpylll.tools.bkz_stats import dummy_tracer, normalize_tracer, Tracer
from fpylll.util import adjust_radius_to_gh_bound
import numpy as np
from tensordict import TensorDict
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
            raise TypeError("Matrix must be IntegerMatrix but got type '%s'" % type(A))

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

        self.lll_obj.size_reduction(max(0, max_row - 1), max_row, max(0, max_row - 2))
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
                max_dist, solution = enum_obj.enumerate(
                    kappa, kappa + block_size, max_dist, expo
                )[0]

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
                            kappa + j_nz, kappa + i, solution[j_nz] * solution[i]
                        )

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
                                    solution[k] = solution[k] - solution[k - offset]
                                    self.M.row_addmul(kappa + k - offset, kappa + k, 1)

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
            clean_pre = self.svp_preprocessing(kappa, block_size, params, tracer)
        clean &= clean_pre

        solution = self.svp_call(kappa, block_size, params, tracer)

        with tracer.context("postprocessing"):
            clean_post = self.svp_postprocessing(kappa, block_size, solution, tracer)
        clean &= clean_post

        self.lll_obj.size_reduction(0, kappa + 1)
        return clean


@dataclass
class ReductionEnvConfig:
    # If not provided, will be set to 2 * basis_dim
    max_steps: Optional[int] = None
    max_block_size: int = None  # inclusive
    time_limit: float = 1.0
    train_max_dim: int = 16
    train_min_dim: int = None
    net_dim: int = None
    batch_size: int = 1

    time_penalty_weight: float = -1.0
    defect_reward_weight: float = 0.1
    length_reward_weight: float = 1.0

    distribution: str = None

    def __post_init__(self):
        if self.max_steps is None:
            self.max_steps = self.net_dim

    def __str__(self):
        self_dict = vars(self)
        return (
            f"ReductionEnvConfig({', '.join(f'{k}={v}' for k, v in self_dict.items())})"
        )


class ReductionEnvironment:
    def __init__(self, config: ReductionEnvConfig):
        super().__init__()
        self.config = config

    def _get_observation(self) -> Dict[str, torch.Tensor]:
        basis = np.zeros((self.config.net_dim, self.config.net_dim))
        self.basis.to_matrix(basis)
        basis = torch.tensor(basis, dtype=torch.float32)

        last_action = torch.tensor([self.action_history[-1]], dtype=torch.float32)

        return {
            "basis": basis,
            "last_action": last_action,
            "basis_dim": torch.tensor([self.basis.ncols]),
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            "log_defect": self.log_defect_history[-1],
            "shortest_length": self.shortest_length_history[-1],
            # "time": sum(self.time_history),
            "time": self.enum_history[-1],  # temp hack
            "gh": self.gh,
            "tgt_length": self.tgt_length,
        }

    def reset(
        self, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        if options is None:
            # basis_dim = np.random.randint(self.config.min_basis_dim, self.config.max_basis_dim + 1)
            basis_dim = (
                np.random.randint(
                    self.config.train_min_dim // 2,
                    self.config.train_max_dim // 2 + 1,
                )
                * 2
            )  # temp hack until I figure out how to properly represent allowable lattice dimensions
            options = generate_random_basis(None, basis_dim, self.config.distribution)
            options["basis"] = torch.tensor(options["basis"])

        self.basis = IntegerMatrix.from_matrix(options["basis"].int().tolist())
        self.gh = self.gaussian_heuristic(self.basis)
        self.tgt_length = options["target_length"]

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

        self.action_history = []
        self.log_defect_history = []
        self.shortest_length_history = []
        self.time_history = []
        self.enum_history = []  # store number of nodes visited by enumeration
        # initial basis is always LLL = BKZ-2 reduced
        self._update_history(action=1, time_taken=0)

        self.current_step = 0

        return self._get_observation(), self._get_info()

    def _action_to_block(self, action: int) -> int:
        """Convert single action index to block size"""
        assert np.all(
            np.array(action) < self.config.net_dim
        ), f"Action {action} provided, but only {self.config.net_dim} actions available!"

        return np.array(action) + 1

    def _block_to_action(self, block_size: int) -> int:
        # _block_to_action should be the (both left and right) inverse of _action_to_block
        return block_size - 1

    def step(
        self, action: int
    ) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict[str, Any]]:
        time_taken = 0
        if action != 0:
            start_time = process_time()
            block_size = self._action_to_block(action)
            self.clean = self.bkz.tour(
                BKZ.EasyParam(
                    block_size=block_size,
                    max_loops=1,
                    gh_factor=1.1,
                    auto_abort=True,
                    strategies=None,
                ),
                tracer=self.tracer,
            )
            time_taken = process_time() - start_time

        self._update_history(action=int(action), time_taken=time_taken)

        self.terminated = self._check_termination()
        self.truncated = self._check_truncation()
        self.current_step += 1

        obs = self._get_observation()
        rewards = self._compute_reward()
        info = self._get_info()
        done = self.terminated or self.truncated

        if done:
            obs, info = self.reset()

        return obs, rewards, self.terminated, self.truncated, info

    def _update_history(self, action: int, time_taken: float = 0.0):
        self.action_history.append(action)
        self.time_history.append(time_taken)
        self.log_defect_history.append(self.compute_log_defect(self.basis))
        self.shortest_length_history.append(min(v.norm() for v in self.basis) / self.gh)

        enum_nodes = None
        for child in self.tracer.trace.children:
            if child.label == "enumeration":
                enum_nodes = child.data["#enum"]
                break

        # 1e-6 because the number of enum calls is really big
        # TODO: set as configurable param
        self.enum_history.append(
            1e-6 * float(enum_nodes) - sum(self.enum_history)
            if enum_nodes is not None
            else 0
        )

    def _compute_reward(self) -> float:
        # Initialize reward components dictionary for better tracking
        rewards = {
            "time_penalty": 0.0,
            "defect_reward": 0.0,
            "length_reward": 0.0,
        }

        if self.action_history[-1] == 0:
            return rewards

        # rewards["time_penalty"] = self.config.time_penalty_weight * self.time_history[-1]
        rewards["time_penalty"] = (
            self.config.time_penalty_weight * self.enum_history[-1]
        )
        rewards["defect_reward"] = self.config.defect_reward_weight * (
            self.log_defect_history[-2] - self.log_defect_history[-1]
        )
        rewards["length_reward"] = self.config.length_reward_weight * (
            self.shortest_length_history[-2] - self.shortest_length_history[-1]
        )

        return rewards

    def _check_termination(self):
        """Check if episode has terminated"""
        if self.action_history[-1] == 0:
            return True

        return False

    def _check_truncation(self):
        """Check if episode has been truncated due to time limit or exceeded max loops"""
        if len(self.action_history) >= self.config.max_steps:
            return True

        if sum(self.time_history) >= self.config.time_limit:
            return True

        return False

    @staticmethod
    def gaussian_heuristic(basis: IntegerMatrix) -> float:
        basis_ = np.zeros((basis.ncols, basis.ncols), dtype=float)
        for i in range(basis.nrows):
            for j in range(basis.ncols):
                basis_[i, j] = basis[i, j]
        _, R = np.linalg.qr(basis_)
        diag = np.abs(np.diagonal(R, axis1=-2, axis2=-1))
        n = diag.shape[0]

        log_gh = np.sum(np.log(diag)) / n - np.log(np.pi) / 2 - math.lgamma(n / 2 + 1) / n
        return np.exp(log_gh)

    @staticmethod
    def compute_log_defect(basis: IntegerMatrix) -> float:
        """Compute the orthogonality defect of a given basis."""
        m = GSO.Mat(basis, flags=GSO.INT_GRAM | GSO.ROW_EXPO, float_type="mpfr")
        m.update_gso()
        log_det = m.get_log_det(0, basis.nrows) / 2  # log(determinant)
        log_prod_norms = sum(np.log(v.norm()) for v in basis)
        log_defect = log_prod_norms - log_det
        return log_defect


def _worker(work_remote, remote, config):
    """Worker function to run environment in subprocess."""
    remote.close()
    env = ReductionEnvironment(config)
    try:
        while True:
            cmd, data = work_remote.recv()
            if cmd == "reset":
                obs, info = env.reset(data)
                work_remote.send((obs, info))
            elif cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                work_remote.send((obs, reward, terminated, truncated, info))
            elif cmd == "close":
                work_remote.close()
                break
            else:
                raise NotImplementedError(f"Command {cmd} not recognized")
    except (EOFError, KeyboardInterrupt):
        return


class VectorizedReductionEnvironment:
    def __init__(self, config: ReductionEnvConfig):
        self.config = config
        self.batch_size = config.batch_size

        start_method = (
            "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        )
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(self.batch_size)]
        )
        self.processes = []
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            args = (work_remote, remote, config)
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.closed = False

    def reset(self, options: TensorDict = None) -> Tuple[TensorDict, TensorDict]:
        if options is None:
            options_list = [None] * self.batch_size
        else:
            options_list = options.unbind(dim=0)

        for remote, action in zip(self.remotes, options_list):
            remote.send(("reset", action))
        results = [remote.recv() for remote in self.remotes]
        states, infos = zip(*results)

        states_ = {
            key: torch.stack([state[key] for state in states]).squeeze(-1)
            for key in states[0]
        }
        infos_ = {}
        for key in infos[0]:
            if isinstance(infos[0][key], torch.Tensor):
                infos_[key] = torch.stack([info[key] for info in infos])
            else:
                infos_[key] = torch.Tensor([info[key] for info in infos])

        return TensorDict(states_), TensorDict(infos_)

    def step(self, actions: torch.Tensor):
        actions_list = actions.cpu().int().tolist()
        for remote, action in zip(self.remotes, actions_list):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        next_states, rewards, terminateds, truncateds, infos = zip(*results)

        next_states_ = TensorDict(
            {
                key: torch.stack([state[key] for state in next_states]).squeeze(-1)
                for key in next_states[0]
            }
        ).to(actions.device)

        rewards_ = defaultdict(list)
        for d in rewards:
            for key, value in d.items():
                rewards_[key].append(value)
        rewards_ = {
            key: torch.tensor(value, device=actions.device)
            for key, value in rewards_.items()
        }

        terminateds_ = torch.tensor(
            terminateds, dtype=torch.bool, device=actions.device
        )
        truncateds_ = torch.tensor(truncateds, dtype=torch.bool, device=actions.device)
        infos_ = {}
        for key in infos[0]:
            if isinstance(infos[0][key], torch.Tensor):
                infos_[key] = torch.stack([info[key] for info in infos]).to(
                    actions.device
                )
            else:
                infos_[key] = torch.tensor(
                    [info[key] for info in infos], device=actions.device
                )
        infos_ = TensorDict(infos_)

        return next_states_, rewards_, terminateds_, truncateds_, infos_

    def close(self):
        """Clean up resources."""
        if self.closed:
            return
        self.closed = True
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except BrokenPipeError:
                pass
        for process in self.processes:
            process.join()
        for remote in self.remotes:
            remote.close()
