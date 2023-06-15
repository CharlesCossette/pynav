"""Main interface for nonlinear least squares problems.

Allows for the construction of nonlinear least squares problem by adding 
arbitrary residual terms, and allows for solving these using either Gauss-Newton
or Levenberg Marquardt.

This code is inspired by utiasSTARTS `pySLAM` repo (https://github.com/utiasSTARS/pyslam),
as well as Ceres. 
"""

import collections.abc
import time
from typing import Dict, Hashable, List, Tuple

import numpy as np
from scipy import sparse

from navlie.batch.losses import L2Loss, LossFunction
from navlie.batch.residuals import Residual
from navlie.types import State


class OptimizationSummary:
    """Class to store a summary of the optimization problem."""

    def __init__(
        self,
        size_state: int,
        size_error: int,
        cost: List[float],
        time: float,
    ):
        self.size_state = size_state
        self.size_error = size_error
        self.cost = cost
        self.time = time

    def __repr__(self):
        string = (
            "Number of states optimized: {0}.\nNumber of error terms: {1}.\n"
            "Initial cost: {2}.\nFinal cost: {3}.\nTotal time: {4}"
        ).format(
            self.size_state,
            self.size_error,
            self.cost[0],
            self.cost[-1],
            self.time,
        )

        return string


class Problem:
    """Main class for building nonlinear least squares problems."""

    def __init__(
        self,
        solver: str = "GN",
        max_iters: int = 100,
        step_tol: float = 1e-7,
        tau: float = 1e-11,
        verbose: bool = True,
    ):
        # Set solver parameters
        self.solver = solver
        self.max_iters = max_iters
        self.step_tol = step_tol
        self.tau = tau
        self.verbose = verbose

        # Initial value of all the variables
        self.variables_init: Dict[str, State] = {}
        # Dict of all current values of the variables
        self.variables: Dict[str, State] = {}
        self.variable_slices: Dict[str, slice] = {}
        # List of variable keys to be held constant
        self.constant_variable_keys: List[Hashable] = []
        # List of all factors in problem
        self.residual_list: List[Residual] = []
        self.residual_slices: List[slice] = []
        self.loss_list: List[LossFunction] = []

        # Size of the optimization problem
        self._size_state: int = None
        self._size_errors: int = None

        # History of the cost
        self._cost_history = None
        # Information matrix upon convergence
        self._information_matrix: np.ndarray = None
        # Inverse of information matrix
        self._covariance_matrix: np.ndarray = None

        # Cost List per iteration
        self.residual_values_by_iter: List[List[float]] = []

    def add_residual(self, residual: Residual, loss: LossFunction = L2Loss()):
        """Adds a residual to the problem, along with a robust loss
        function to use. Default loss function is the standard L2Loss.

        Parameters
        ----------
        residual : Residual
            the error term to be added to the problem.
        loss : LossFunction, optional
            robust loss to be used for this residual, by default L2Loss().
        """
        # If the user specifies a list of residual, extend existing list
        if isinstance(residual, list):
            self.residual_list.extend(residual)

            if not isinstance(loss, list):
                self.loss_list.extend([loss for i in residual])
            else:
                self.loss_list.extend(loss)
        else:
            self.residual_list.append(residual)
            self.loss_list.append(loss)

    def add_variable(self, key: Hashable, variable: State):
        """Adds a variable to the problem with a given key."""
        self.variables_init[key] = variable

    def set_variables_constant(self, keys: List[Hashable]):
        """Sets a variable to be held constant during optimization.

        Parameters
        ----------
        keys : List[Hashable]
            List of keys to be held constant
        """
        if isinstance(keys, collections.abc.Hashable):
            keys = [keys]

        for key in keys:
            if key not in self.constant_variable_keys:
                self.constant_variable_keys.append(key)

    def solve(self):
        """Solve the problem using either Gauss-Newton or Levenberg-Marquardt."""

        # Timing
        start_t = time.time()

        # Make a copy of the initial variables. This is done so that the
        # initial values of the variables are not modified by the solver.
        self.variables = {k: v.copy() for k, v in self.variables_init.items()}

        # Compute the size of the problem and slices of variables
        # and residuals.
        self._compute_size_of_problem()

        # Solve using solver of choice
        # TODO: these two _solve functions still have much repeated code.
        # there is probably a way to refactor slightly to reduce this.
        if self.solver == "GN":
            self._solve_gauss_newton()
        elif self.solver == "LM":
            self._solve_LM()

        # Compute total time
        total_time = time.time() - start_t

        # Create optimization summary
        summary = OptimizationSummary(
            self._size_state, self._size_errors, self._cost_history, total_time
        )

        # Return result
        result = {
            "variables": self.variables,
            "info_matrix": self._information_matrix,
            "summary": summary,
        }

        return result

    def _solve_gauss_newton(self) -> Dict[Hashable, State]:
        """Solves the optimization problem using Gauss-Newton.

        Returns
        -------
        Dict[Hashable, State]
            New dictionary of optimized variables.
        """

        dx = 10
        iter_idx = 0
        cost_list = []

        e, H, cost = self.compute_error_jac_cost()
        cost_list.append(cost)

        # Print initial cost
        if self.verbose:
            header = "Initial cost: " + str(cost)
            print(header)

        while (iter_idx < self.max_iters) and (dx > self.step_tol):
            H_spr = sparse.csr_matrix(H)

            A = H_spr.T @ H_spr
            b = H_spr.T @ e

            delta_x = sparse.linalg.spsolve(A, -b).reshape((-1, 1))

            # Update the variables
            self._correct_states(delta_x)

            e, H, cost = self.compute_error_jac_cost()
            cost_list.append(cost)

            dx = np.linalg.norm(delta_x)
            if self.verbose:
                self._display_header(iter_idx, cost, dx)

            iter_idx += 1

        # After convergence, compute final value of the cost function
        e, H, cost = self.compute_error_jac_cost()
        cost_list.append(cost)

        self._cost_history = np.array(cost_list).reshape((-1))
        self._information_matrix = A

        return self.variables

    def _solve_LM(self) -> Dict[Hashable, State]:
        """Solves the optimization problem using Gauss-Newton.

        Returns
        -------
        Dict[Hashable, State]
            New dictionary of optimized variables.
        """

        e, H, cost = self.compute_error_jac_cost()
        cost_list = [cost]

        H_spr = sparse.csr_matrix(H)

        A = H_spr.T @ H_spr
        b = H_spr.T @ e

        iter_idx = 0
        dx = 10
        mu = self.tau * np.amax(A.diagonal())
        nu = 2
        prev_cost = cost

        if self.verbose:
            header = "Initial cost: " + str(cost)
            print(header)

        # Main LM loop
        while (iter_idx < self.max_iters) and (dx > self.step_tol):
            A_solve = A + mu * sparse.identity(A.shape[0])
            delta_x = sparse.linalg.spsolve(A_solve, -b).reshape((-1, 1))

            variables_test = {k: v.copy() for k, v in self.variables.items()}

            # Update the variables
            self._correct_states(delta_x, variables_test)

            # Compute the new value of the cost function after the update
            e, H, cost = self.compute_error_jac_cost(variables=variables_test)

            gain_ratio = (prev_cost - cost) / (0.5 * delta_x.T @ (mu * delta_x - b))
            gain_ratio = gain_ratio.item(0)

            # If the gain ratio is above zero, accept the step
            if gain_ratio > 0:
                self.variables = variables_test
                mu = mu * max(1.0 / 3.0, 1.0 - (2.0 * gain_ratio - 1) ** 3)
                nu = 2

                e, H, cost = self.compute_error_jac_cost()
                cost_list.append(cost)
                prev_cost = cost

                H_spr = sparse.csr_matrix(H)

                A = H_spr.T @ H_spr
                b = H_spr.T @ e
                status = "Accepted."
            else:
                mu = mu * nu
                nu = 2 * nu
                status = "Rejected."

            dx = np.linalg.norm(delta_x)

            if self.verbose:
                self._display_header(iter_idx + 1, cost, dx, status=status)

            iter_idx += 1

        # After convergence, compute final value of the cost function
        e, H, cost = self.compute_error_jac_cost()
        cost_list.append(cost)

        self._cost_history = np.array(cost_list).reshape((-1))
        self._information_matrix = A

        return self.variables

    def compute_error_jac_cost(
        self, variables: Dict[Hashable, State] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Computes the full error vector, Jacobian, and cost of the problem.

        Parameters
        ----------
        variables : Dict[Hashable, State], optional
            Variables, by default None. If None, uses the variables stored in
            the optimizer.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            Error vector, Jacobian, and cost.
        """

        if variables is None:
            variables = self.variables

        # Compute the size of the problem if needed
        if self._size_errors is None or self._size_state is None:
            self._compute_size_of_problem()

        # Initialize the error vector and Jacobian
        e = np.zeros((self._size_errors,))
        H = np.zeros((self._size_errors, self._size_state))
        cost_list = []
        raw_residual_list: List[float] = []
        # For each factor, evaluate error and Jacobian
        for i, (residual, loss) in enumerate(zip(self.residual_list, self.loss_list)):
            variables_list = [variables[key] for key in residual.keys]

            # Do not compute Jacobian for variables that are held fixed
            compute_jacobians = [
                False if key in self.constant_variable_keys else True
                for key in residual.keys
            ]

            # Evaluate current factor at states
            error, jacobians = residual.evaluate(variables_list, compute_jacobians)

            # Compute the robust loss weight and then weight the error
            u = np.linalg.norm(error)
            sqrt_loss_weight = np.sqrt(loss.weight(u))
            weighted_error = sqrt_loss_weight * error

            # Place errors
            e[self.residual_slices[i]] = weighted_error.ravel()

            # Compute cost
            cost = np.sum(loss.loss(u))
            cost_list.append(cost)
            raw_residual_list.append(u)

            # Place each Jacobian in the correct spot
            for j, key in enumerate(residual.keys):
                jacobian = jacobians[j]
                if jacobian is not None:
                    # Correctly weight the Jacobian
                    jacobian = sqrt_loss_weight * jacobian

                    H[self.residual_slices[i], self.variable_slices[key]] = jacobian

        # Save all the residual values
        self.residual_values_by_iter.append(raw_residual_list)
        # Sum up costs from each residual
        cost = np.sum(np.array(cost_list))

        return e.reshape((-1, 1)), H, cost

    def _compute_size_of_problem(self) -> None:
        """Computes the total size of the problem, i.e. the number of variables
        and number of residuals.

        Also computes the slices for both the the residual and the variables,
        to determine their place in the covariance matrix and Jacobian.
        """

        # Compute the state slices
        idx = 0
        for key, var in self.variables.items():
            if key not in self.constant_variable_keys:
                self.variable_slices[key] = slice(idx, idx + var.dof)
                idx += var.dof
        size_state = idx

        # For each factor, evaluate error and Jacobian
        idx = 0
        self.residual_slices = [None] * len(self.residual_list)
        for i, residual in enumerate(self.residual_list):
            # Get the current variables that this residual depends on
            cur_variables = [self.variables[key] for key in residual.keys]

            # Evaluate current factor at states
            error = residual.evaluate(cur_variables)
            self.residual_slices[i] = slice(idx, idx + error.size)
            idx += error.size

        size_errors = idx
        self._size_errors = size_errors
        self._size_state = size_state

    def _correct_states(
        self,
        delta_x: np.ndarray,
        variables: Dict[Hashable, State] = None,
    ):
        """Updates each of the variables using the individual "plus" operators
        of each variable.
        Parameters
        ----------
        delta_x : np.ndarray
            Increment to the entire state
        variables : Dict[Hashable, State], optional
            Variables to be updated, by default None
            if None, uses the variables stored in the optimizer.
        """
        # TODO: i feel like this function should return updated variables
        # instead of saving to self.

        if variables is None:
            variables = self.variables

        for key, var in variables.items():
            if not key in self.constant_variable_keys:
                slc = self.variable_slices[key]
                delta_xi_current = delta_x[slc, [0]]
                variables[key] = var.plus(delta_xi_current)

    def get_covariance_block(self, key_1: Hashable, key_2: Hashable) -> np.ndarray:
        """Retrieve the covariance block corresponding to two variables.

        Parameters
        ----------
        key_1 : Hashable
            Key of first variable.
        key_2 : Hashable
            Key of second variable.

        Returns
        -------
        np.ndarray
            Covariance block corresponding to the two variables.
        """

        # Compute the full covariance if it has not already been done
        if self._covariance_matrix is None:
            self.compute_covariance()

        # Extract relevant block
        try:
            var_1_slice = self.variable_slices[key_1]
            var_2_slice = self.variable_slices[key_2]

            return self._covariance_matrix[var_1_slice, var_2_slice]
        except KeyError as e:
            print(f"Cannot compute covariance block!")

    def compute_covariance(self):
        """Compute covariance matrix after convergence of problem."""
        try:
            self._covariance_matrix = sparse.linalg.inv(
                self._information_matrix
            ).toarray()
            return self._covariance_matrix
        except Exception as e:
            print("Covariance computation failed!\n{}".format(e))
            return None

    def _display_header(
        self, iter_idx: int, current_cost: float, dx: float, status: str = None
    ):
        """Displays the optimization progress.

        Parameters
        ----------
        iter_idx : int
            Iteration number.
        current_cost : float
            Current objective function cost.
        dx : float
            Norm of step size.
        status : str, optional
            Status for LM, by default None
        """
        header = ("Iter: {0} || Cost: {1:.4e} || Step size: {2:.4e}").format(
            iter_idx, current_cost, dx
        )

        if status is not None:
            header += " || Status: " + status

        print(header)
