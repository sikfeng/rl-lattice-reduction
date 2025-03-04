from dataclasses import dataclass
from itertools import takewhile
import multiprocessing as mp
import time
from typing import Any, Dict, List, Optional, Tuple

from fpylll import FPLLL, GSO, IntegerMatrix, LLL
import numpy as np
from tensordict import TensorDict
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


class BlockReduction:
    def __init__(self, A: IntegerMatrix):
        if not isinstance(A, IntegerMatrix):
            raise TypeError(f"Matrix must be IntegerMatrix but got {type(A)}")

        self.A = A
        LLL.reduction(self.A)

    def __call__(self, kappa, block_size):
        """Perform one step of Lattice reduction.

        :param kappa: row index
        :param block_size: an integer >= 2

        """
        block_matrix = self.A.submatrix(
            range(kappa, kappa + block_size), range(self.A.ncols))

        LLL.reduction(block_matrix)

        shortest_idx = np.argmin([row.norm() for row in block_matrix])
        row = block_matrix[shortest_idx]
        self.A.set_rows(self.A.nrows + 1)

        for i in range(self.A.ncols):
            self.A[-1, i] = row[i]
        LLL.reduction(self.A)

        self.A = self.A.submatrix(range(1, self.A.nrows), range(self.A.ncols))
        return self.A


@dataclass
class ReductionEnvConfig:
    # If not provided, will be set to 2 * basis_dim
    max_steps: Optional[int] = None
    min_block_size: int = None  # inclusive
    max_block_size: int = None  # inclusive
    time_limit: float = 300
    basis_dim: int = None
    action_history_size: int = 10
    batch_size: int = 1

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


class ReductionEnvironment:
    def __init__(self, config: ReductionEnvConfig):
        super().__init__()
        self.config = config

    def _get_observation(self) -> Dict[str, torch.Tensor]:
        basis = np.zeros((self.config.basis_dim, self.config.basis_dim))
        basis = torch.tensor(self.basis.to_matrix(basis), dtype=torch.float32)

        # For an ideal model, it should not have to use the action history,
        # but the model frequently repeats the immediate last action and gets stuck.
        # Hence, this is an attempt to teach the model not to do that.
        last_actions = torch.tensor(
            self.action_history[-self.config.action_history_size:], dtype=torch.float32)
        history = torch.cat([torch.full(
            (self.config.action_history_size - last_actions.size(0),), -1.0), last_actions])

        return {
            "basis": basis,
            "action_history": history
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            "log_defect": self.log_defect_history[-1],
            "shortest_length": self.shortest_length_history[-1]
        }

    def reset(self, options: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        self.basis = IntegerMatrix.from_matrix(options["basis"].int().tolist())
        self.lll_log_defect = options["lll_log_defect"]
        self.shortest_lll_basis_vector_length = options["shortest_lll_basis_vector_length"]
        self.shortest_vector_length = options["shortest_vector_length"]

        self.bkz = BlockReduction(self.basis)
        self.start_time = time.time()
        self.action_history = []
        self.log_defect_history = [compute_log_defect(self.basis)]
        self.shortest_length_history = [
            options["shortest_original_basis_vector_length"]]
        self.current_step = 0

        return self._get_observation(), self._get_info()

    def _action_to_block(self, action: int) -> Tuple[int, int]:
        """Convert single action index to block size and start position"""
        assert action < self.config.actions_n, f"Action {action} provided, but only {self.config.actions_n} actions available!"

        for start_pos in range(self.config.basis_dim):
            for block_size in range(self.config.min_block_size, self.config.max_block_size + 1):
                if start_pos + block_size > self.config.basis_dim:
                    continue
                if action == 0:
                    return start_pos, block_size
                action -= 1

    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict[str, Any]]:
        start_pos, block_size = self._action_to_block(action)
        self.basis = self.bkz(start_pos, block_size)

        self.action_history.append(action)

        self.terminated = self._check_termination()
        self.truncated = self._check_truncation()
        self.current_step += 1

        return self._get_observation(), self._compute_reward(), self._check_termination(), self._check_truncation(), self._get_info()

    def _compute_reward(self) -> float:
        # Initialize reward components dictionary for better tracking
        rewards = {
            "time_penalty": 0.0,
            "repeat_penalty": 0.0,
            "defect_reward": 0.0,
            "length_reward": 0.0,
            "proximity_reward": 0.0,
            "improvement_bonus": 0.0
        }

        # Penalty for time taken - using a more gradual time penalty
        elapsed_time = time.time() - self.start_time
        rewards["time_penalty"] = self.config.time_penalty_weight * \
            (elapsed_time / (10.0 + elapsed_time))

        # Penalty for repeated actions - smoothed with a cap
        if len(self.action_history) > 1:
            if self.action_history[-1] == self.action_history[-2]:
                repeats = sum(1 for _ in takewhile(lambda x: x == self.action_history[-1],
                                                   reversed(self.action_history)))
                # More linear penalty scaling with soft maximum
                rewards["repeat_penalty"] = min(1.0 * repeats, 5.0)

        # Compute current metrics
        current_log_defect = compute_log_defect(self.basis)
        self.log_defect_history.append(current_log_defect)
        current_shortest_length = min(v.norm() for v in self.basis)
        self.shortest_length_history.append(current_shortest_length)

        # Calculate improvements from previous steps
        if len(self.action_history) > 0:
            # Calculate improvement from last step for log defect
            defect_improvement = self.log_defect_history[-2] - \
                self.log_defect_history[-1]

            # Calculate improvement from last step for shortest vector length
            length_improvement = self.shortest_length_history[-2] - \
                self.shortest_length_history[-1]

            # Normalize length improvement relative to optimal length
            normalized_length_improvement = 0
            # Compute how much closer we got to the optimal length
            distance_to_optimal_before = abs(
                self.shortest_length_history[-2] - self.shortest_vector_length)
            distance_to_optimal_after = abs(
                self.shortest_length_history[-1] - self.shortest_vector_length)

            # Use relative improvement with a safety factor
            if distance_to_optimal_before > 0:
                normalized_length_improvement = (
                    distance_to_optimal_before - distance_to_optimal_after) / (distance_to_optimal_before + 1e-6)

            # Apply a bonus for reducing the shortest vector length - more linear
            if length_improvement > 0:
                rewards["length_reward"] = 7.0 * normalized_length_improvement

            # Apply a bonus for improving the defect - use a more linear reward function
            if defect_improvement > 0:
                # More linear scaling with normalization
                rewards["defect_reward"] = 5.0 * \
                    min(defect_improvement / (current_log_defect + 1e-6), 1.0)
            elif defect_improvement < 0:
                # Small penalty for regression with diminishing effect
                rewards["defect_reward"] = -0.5 * \
                    min(abs(defect_improvement) /
                        (current_log_defect + 1e-6), 0.5)

            # Calculate proximity reward - smoother function
            if self.shortest_vector_length > 0:
                proximity_ratio = current_shortest_length / self.shortest_vector_length

                # Sigmoid-like reward that approaches maximum as we get closer to optimal
                if proximity_ratio < 10.0:  # Increased valid range
                    rewards["proximity_reward"] = 3.0 / \
                        (1.0 + abs(proximity_ratio - 1.0))

            # Smaller improvement bonus to avoid overshadowing the main rewards
            if defect_improvement > 0 or length_improvement > 0:
                rewards["improvement_bonus"] = 0.5

        # Clip final reward to prevent extreme values
        total_reward = sum(rewards.values())
        return max(min(total_reward, 10.0), -5.0)  # Clip between -5 and 10

    def _check_termination(self):
        """Check if episode has terminated"""

        if len(self.log_defect_history) < self.config.action_history_size:
            return False

        if len(self.log_defect_history) > self.config.action_history_size:
            recent_defects = self.log_defect_history[-self.config.action_history_size:]
            if max(recent_defects) - min(recent_defects) < 1e-6:
                return True

        if (len(self.action_history) >= self.config.action_history_size and
                all(x == self.action_history[-1] for x in self.action_history[-self.config.action_history_size:])):
            return True

        return False

    def _check_truncation(self):
        """Check if episode has been truncated due to time limit"""
        if len(self.action_history) >= self.config.max_steps:
            return True

        if time.time() - self.start_time >= self.config.time_limit:
            return True

        return False


def _worker(work_remote, remote, config):
    """Worker function to run environment in subprocess."""
    remote.close()
    env = ReductionEnvironment(config)
    try:
        while True:
            cmd, data = work_remote.recv()
            if cmd == 'reset':
                obs, info = env.reset(data)
                work_remote.send((obs, info))
            elif cmd == 'step':
                obs, reward, terminated, truncated, info = env.step(data)
                work_remote.send((obs, reward, terminated, truncated, info))
            elif cmd == 'close':
                env.close()
                work_remote.close()
                break
            else:
                raise NotImplementedError(f"Command {cmd} not recognized")
    except (EOFError, KeyboardInterrupt):
        env.close()


class VectorizedReductionEnvironment:
    def __init__(self, config: ReductionEnvConfig):
        self.config = config
        self.batch_size = config.batch_size

        start_method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(self.batch_size)])
        self.processes = []
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            args = (work_remote, remote, config)
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

    def reset(self, options: TensorDict) -> List[Tuple[Dict[str, torch.Tensor], Dict[str, Any]]]:
        options_list = options.unbind(dim=0)
        for remote, action in zip(self.remotes, options_list):
            remote.send(('reset', action))
        results = [remote.recv() for remote in self.remotes]
        states, infos = zip(*results)

        states_ = {key: torch.stack([state[key]
                                    for state in states]) for key in states[0]}
        infos_ = {}
        for key in infos[0]:
            if isinstance(infos[0][key], torch.Tensor):
                infos_[key] = torch.stack([info[key] for info in infos])
            else:
                infos_[key] = torch.Tensor([info[key] for info in infos])

        return TensorDict(states_), TensorDict(infos_)

    def step(self, actions: torch.Tensor):
        actions_list = actions.cpu().tolist()
        for remote, action in zip(self.remotes, actions_list):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        next_states, rewards, terminateds, truncateds, infos = zip(*results)

        next_states_ = TensorDict({key: torch.stack(
            [state[key] for state in next_states]) for key in next_states[0]})
        rewards_ = torch.tensor(rewards, dtype=torch.float32)
        terminateds_ = torch.tensor(terminateds, dtype=torch.bool)
        truncateds_ = torch.tensor(truncateds, dtype=torch.bool)
        infos_ = {}
        for key in infos[0]:
            if isinstance(infos[0][key], torch.Tensor):
                infos_[key] = torch.stack([info[key] for info in infos])
            else:
                infos_[key] = torch.Tensor([info[key] for info in infos])

        return next_states_, rewards_, terminateds_, truncateds_, infos_

    def close(self):
        """Clean up resources."""
        if self.closed:
            return
        self.closed = True
        for remote in self.remotes:
            try:
                remote.send(('close', None))
            except BrokenPipeError:
                pass
        for process in self.processes:
            process.join()
        for remote in self.remotes:
            remote.close()
