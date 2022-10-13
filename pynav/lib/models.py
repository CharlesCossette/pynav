from pynav.types import (
    Measurement,
    ProcessModel,
    MeasurementModel,
    StampedValue,
    Input
)
from pynav.lib.states import (
    CompositeState,
    MatrixLieGroupState,
    VectorState,
)
from pylie import SO2, SO3
import numpy as np
from typing import List
from scipy.linalg import block_diag


class SingleIntegrator(ProcessModel):
    """
    The single-integrator process model is a process model of the form

        x_dot = u .
    """

    def __init__(self, Q: np.ndarray):

        if Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be an n x n matrix.")

        self._Q = Q
        self.dim = Q.shape[0]

    def evaluate(
        self, x: VectorState, u: StampedValue, dt: float
    ) -> np.ndarray:
        x.value = x.value + dt * u.value
        return x

    def jacobian(self, x, u, dt) -> np.ndarray:
        return np.identity(self.dim)

    def covariance(self, x, u, dt) -> np.ndarray:
        return dt**2 * self._Q


class DoubleIntegrator(ProcessModel):
    """
    A second-order kinematic process model with discretization as in
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=303738
    """

    def __init__(self, Q: np.ndarray):
        '''
        inputs:
            Q: Discrete time covariance on the input u. 
        '''
        if Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be an n x n matrix.")

        self._Q = Q
        self.dim = 2

    def evaluate(self, x: VectorState, u: StampedValue, dt: float) -> np.ndarray:
        '''
        Evaluate discrete-time process model
        '''
        Ad = np.array([[1, dt],
                      [0, 1]])
        Ld = np.array([0.5*dt**2, dt]).reshape((-1,1))
        x.value = (Ad @ x.value.reshape((-1,1)) + Ld * u.value).ravel()
        return x

    def jacobian(self, x, u, dt) -> np.ndarray:
        '''
        Discrete-time state Jacobian
        '''
        Ad = np.array([[1, dt],
                [0, 1]])
        return Ad

    def covariance(self, x, u, dt) -> np.ndarray:
        '''
        Discrete-time covariance on process model
        '''
        Ld = np.array([0.5*dt**2, dt]).reshape((-1,1))
        fudge_factor = np.array([[1e-8, 0],
                                [0, 0]])
        return  Ld @ self._Q @ Ld.T + fudge_factor

class OneDimensionalPositionVelocityRange(MeasurementModel):
    '''
    A 1D range measurement for a state consisting of position and velocity
    '''
    def __init__(self, R: float):
        self._R = np.array(R)

    def evaluate(self, x: VectorState) -> np.ndarray:
        return x.value[0]

    def jacobian(self, x: VectorState) -> np.ndarray:
        return np.array([1, 0]).reshape(1,-1)

    def covariance(self, x: VectorState) -> np.ndarray:
        return self._R
        
class BodyFrameVelocity(ProcessModel):
    """
    The body-frame velocity process model assumes that the input contains
    both translational and angular velocity measurements, both relative to
    a local reference frame, but resolved in the robot body frame.

    This is commonly the process model associated with SE(n).
    """

    def __init__(self, Q: np.ndarray):
        self._Q = Q

    def evaluate(
        self, x: MatrixLieGroupState, u: StampedValue, dt: float
    ) -> MatrixLieGroupState:
        x.value = x.value @ x.group.Exp(u.value * dt)
        return x

    def jacobian(
        self, x: MatrixLieGroupState, u: StampedValue, dt: float
    ) -> np.ndarray:
        if x.direction == "right":
            return x.group.adjoint(x.group.Exp(-u.value * dt))
        elif x.direction == "left":
            return np.identity(x.dof)

    def covariance(
        self, x: MatrixLieGroupState, u: StampedValue, dt: float
    ) -> np.ndarray:
        if x.direction == "right":
            L = dt * x.group.left_jacobian(-u.value * dt)
        elif x.direction == "left":
            Ad = x.group.adjoint(x.value @ x.group.Exp(u.value * dt))
            L = dt * Ad @ x.group.left_jacobian(-u.value * dt)

        return L @ self._Q @ L.T

class RelativeBodyFrameVelocity(ProcessModel):
    def __init__(self, Q1: np.ndarray, Q2: np.ndarray):
        self._Q1 = Q1
        self._Q2 = Q2

    def evaluate(
        self, x: MatrixLieGroupState, u: StampedValue, dt: float
    ) -> MatrixLieGroupState:
        u = u.value.reshape((2, round(u.value.size / 2)))
        x.value = x.group.Exp(-u[0] * dt) @ x.value @ x.group.Exp(u[1] * dt)
        return x

    def jacobian(
        self, x: MatrixLieGroupState, u: StampedValue, dt: float
    ) -> np.ndarray:
        u = u.value.reshape((2, round(u.value.size / 2)))
        if x.direction == "right":
            return x.group.adjoint(x.group.Exp(-u[1] * dt))
        else:
            raise NotImplementedError(
                "TODO: left jacobian not yet implemented."
            )

    def covariance(
        self, x: MatrixLieGroupState, u: StampedValue, dt: float
    ) -> np.ndarray:
        u = u.value.reshape((2, round(u.value.size / 2)))
        u1 = u[0]
        u2 = u[1]
        if x.direction == "right":
            L1 = (
                dt
                * x.group.adjoint(x.value @ x.group.Exp(u2 * dt))
                @ x.group.left_jacobian(dt * u1)
            )
            L2 = dt * x.group.left_jacobian(-dt * u2)
            return L1 @ self._Q1 @ L1.T + L2 @ self._Q2 @ L2.T
        else:
            raise NotImplementedError(
                "TODO: left covariance not yet implemented."
            )

class CompositeInput(Input):
    def __init__(self, input_list: List[Input]) -> None:
        self.input_list = input_list

    @property 
    def dof(self) -> int:
        return sum([input.dof for input in self.input_list])

    @property
    def stamp(self) -> float:
        return self.input_list[0].stamp

    def copy(self) -> "CompositeInput":
        return CompositeInput([input.copy() for input in self.input_list])

    def plus(self, w: np.ndarray):
        new = self.copy()
        for i, input in enumerate(self.input_list):
            new.input_list[i] = input.plus(w[:input.dof])
            w = w[input.dof:]
    
        return new

class CompositeProcessModel(ProcessModel):
    """
    Should this be called a StackedProcessModel?
    """

    def __init__(self, model_list: List[ProcessModel]):
        self._model_list = model_list

    def evaluate(
        self, x: CompositeState, u: CompositeInput, dt: float
    ) -> CompositeState:
        for i, x_sub in enumerate(x.value):
            u_sub = u.input_list[i]
            x.value[i] = self._model_list[i].evaluate(x_sub, u_sub, dt)

        return x

    def jacobian(
        self, x: CompositeState, u: CompositeInput, dt: float
    ) -> np.ndarray:
        jac = []
        for i, x_sub in enumerate(x.value):
            u_sub = u.input_list[i]
            jac.append(self._model_list[i].jacobian(x_sub, u_sub, dt))

        return block_diag(*jac)

    def covariance(
        self, x: CompositeState, u: CompositeInput, dt: float
    ) -> np.ndarray:
        cov = []
        for i, x_sub in enumerate(x.value):
            u_sub = u.input_list[i]
            cov.append(self._model_list[i].covariance(x_sub, u_sub, dt))

        return block_diag(*cov)


class CompositeMeasurementModel(MeasurementModel):
    """
    Wrapper for a standard measurement model that assigns the model to a specific
    substate (referenced by `state_id`) inside a CompositeState.
    """

    def __init__(self, model: MeasurementModel, state_id):
        self.model = model
        self.state_id = state_id

    def evaluate(self, x: CompositeState) -> np.ndarray:
        return self.model.evaluate(x.get_state_by_id(self.state_id))

    def jacobian(self, x: CompositeState) -> np.ndarray:
        x_sub = x.get_state_by_id(self.state_id)
        jac_sub = self.model.jacobian(x_sub)
        jac = np.zeros((jac_sub.shape[0], x.dof))
        slc = x.get_slice_by_id(self.state_id)
        jac[:, slc] = jac_sub
        return jac

    def covariance(self, x: CompositeState) -> np.ndarray:
        x_sub = x.get_state_by_id(self.state_id)
        return self.model.covariance(x_sub)


class RangePointToAnchor(MeasurementModel):
    """
    Range measurement from a point state to an anchor (which is also another
    point).
    """

    def __init__(self, anchor_position: List[float], R: float):
        self._r_cw_a = np.array(anchor_position).flatten()
        self._R = np.array(R)

    def evaluate(self, x: VectorState) -> np.ndarray:
        r_zw_a = x.value.flatten()
        y = np.linalg.norm(self._r_cw_a - r_zw_a)
        return y

    def jacobian(self, x: VectorState) -> np.ndarray:
        r_zw_a = x.value.flatten()
        r_zc_a: np.ndarray = r_zw_a - self._r_cw_a
        y = np.linalg.norm(r_zc_a)
        return r_zc_a.reshape((1, -1)) / y

    def covariance(self, x: VectorState) -> np.ndarray:
        return self._R


class PointRelativePosition(MeasurementModel):
    def __init__(
        self,
        landmark_position: np.ndarray,
        R: np.ndarray,
    ):
        self._landmark_position = np.array(landmark_position).ravel()
        self._R = R

    def evaluate(self, x: MatrixLieGroupState) -> np.ndarray:
        """Evaluates the measurement model of a landmark from a given pose."""
        r_zw_a = x.position.reshape((-1, 1))
        C_ab = x.attitude
        r_pw_a = self._landmark_position.reshape((-1, 1))
        return C_ab.T @ (r_pw_a - r_zw_a)

    def jacobian(self, x: MatrixLieGroupState) -> np.ndarray:
        r_zw_a = x.position.reshape((-1, 1))
        C_ab = x.attitude
        r_pw_a = self._landmark_position.reshape((-1, 1))
        y = C_ab.T @ (r_pw_a - r_zw_a)

        if x.direction == "right":
            return x.jacobian_from_blocks(
                attitude=-SO3.odot(y), position=-np.identity(r_zw_a.shape[0])
            )

        elif x.direction == "left":
            return x.jacobian_from_blocks(
                attitude= -C_ab.T @ SO3.odot(r_pw_a), position= -C_ab.T
            )

    def covariance(self, x: MatrixLieGroupState) -> np.ndarray:
        return self._R


class InvariantPointRelativePosition(MeasurementModel):
    def __init__(self, y: np.ndarray, model: PointRelativePosition):
        self.y = y.ravel()
        self.measurement_model = model

    def evaluate(self, x: MatrixLieGroupState) -> np.ndarray:
        """Computes the right-invariant innovation.


        Parameters
        ----------
        x : MatrixLieGroupState
            Evaluation point of the innovation.

        Returns
        -------
        np.ndarray
            Residual.
        """
        y_hat = self.measurement_model.evaluate(x)
        e: np.ndarray = y_hat.ravel() - self.y.ravel()
        z = x.attitude @ e

        return z

    def jacobian(self, x: MatrixLieGroupState) -> np.ndarray:
        """Compute the Jacobian of the innovation directly.

        Parameters
        ----------
        x : MatrixLieGroupState
            Matrix Lie group state containing attitude and position

        Returns
        -------
        np.ndarray
            Jacobian of the innovation w.r.t the state
        """

        if x.direction == "left":
            jac_attitude = SO3.cross(
                self.measurement_model._landmark_position
            )
            jac_position = -np.identity(3)
        else:
            raise NotImplementedError("Right jacobian not implemented.")

        jac = x.jacobian_from_blocks(
            attitude=jac_attitude,
            position=jac_position,
        )

        return jac

    def covariance(self, x: MatrixLieGroupState) -> np.ndarray:

        R = np.atleast_2d(self.measurement_model.covariance(x))
        M = x.attitude
        cov = M @ R @ M.T

        return cov


class RangePoseToAnchor(MeasurementModel):
    """
    Range measurement from a pose state to an anchor.
    """

    def __init__(
        self,
        anchor_position: List[float],
        tag_body_position: List[float],
        R: float,
    ):
        self._r_cw_a = np.array(anchor_position).flatten()
        self._R = R
        self._r_tz_b = np.array(tag_body_position).flatten()

    def evaluate(self, x: MatrixLieGroupState) -> np.ndarray:
        r_zw_a = x.position
        C_ab = x.attitude

        r_tw_a = C_ab @ self._r_tz_b.reshape((-1, 1)) + r_zw_a.reshape((-1, 1))
        r_tc_a: np.ndarray = r_tw_a - self._r_cw_a.reshape((-1, 1))
        return np.linalg.norm(r_tc_a)

    def jacobian(self, x: MatrixLieGroupState) -> np.ndarray:
        r_zw_a = x.position
        C_ab = x.attitude
        if C_ab.shape == (2, 2):
            att_group = SO2
        elif C_ab.shape == (3, 3):
            att_group = SO3

        r_tw_a = C_ab @ self._r_tz_b.reshape((-1, 1)) + r_zw_a.reshape(
            (-1, 1)
        )
        r_tc_a: np.ndarray = r_tw_a - self._r_cw_a.reshape((-1, 1))
        rho = r_tc_a / np.linalg.norm(r_tc_a)

        if x.direction == "right":
            jac_attitude = rho.T @ C_ab @ att_group.odot(self._r_tz_b)
            jac_position = rho.T @ C_ab
        elif x.direction == "left":
            jac_attitude = rho.T @ att_group.odot( C_ab @ self._r_tz_b + r_zw_a)
            jac_position = rho.T @ np.identity(r_zw_a.size)

        jac = x.jacobian_from_blocks(
            attitude=jac_attitude,
            position=jac_position,
        )
        return jac

    def covariance(self, x: MatrixLieGroupState) -> np.ndarray:
        return self._R


class RangePoseToPose(MeasurementModel):
    """
    Range model given two absolute poses of rigid bodies, each containing a tag.
    """

    def __init__(
        self, tag_body_position1, tag_body_position2, state_id1, state_id2, R
    ):
        self.tag_body_position1 = np.array(tag_body_position1).flatten()
        self.tag_body_position2 = np.array(tag_body_position2).flatten()
        self.state_id1 = state_id1
        self.state_id2 = state_id2
        self._R = R

    def evaluate(self, x: CompositeState) -> np.ndarray:
        x1: MatrixLieGroupState = x.get_state_by_id(self.state_id1)
        x2: MatrixLieGroupState = x.get_state_by_id(self.state_id2)
        r_1w_a = x1.position.reshape((-1, 1))
        C_a1 = x1.attitude
        r_2w_a = x2.position.reshape((-1, 1))
        C_a2 = x2.attitude
        r_t1_1 = self.tag_body_position1.reshape((-1, 1))
        r_t2_2 = self.tag_body_position2.reshape((-1, 1))
        r_t1t2_a: np.ndarray = (C_a1 @ r_t1_1 + r_1w_a) - (
            C_a2 @ r_t2_2 + r_2w_a
        )
        return np.array(np.linalg.norm(r_t1t2_a.flatten()))

    def jacobian(self, x: CompositeState) -> np.ndarray:
        x1: MatrixLieGroupState = x.get_state_by_id(self.state_id1)
        x2: MatrixLieGroupState = x.get_state_by_id(self.state_id2)
        r_1w_a = x1.position.reshape((-1, 1))
        C_a1 = x1.attitude
        r_2w_a = x2.position.reshape((-1, 1))
        C_a2 = x2.attitude
        r_t1_1 = self.tag_body_position1.reshape((-1, 1))
        r_t2_2 = self.tag_body_position2.reshape((-1, 1))
        r_t1t2_a: np.ndarray = (C_a1 @ r_t1_1 + r_1w_a) - (
            C_a2 @ r_t2_2 + r_2w_a
        )

        if C_a1.shape == (2, 2):
            att_group = SO2
        elif C_a1.shape == (3, 3):
            att_group = SO3

        rho: np.ndarray = (
            r_t1t2_a / np.linalg.norm(r_t1t2_a.flatten())
        ).reshape((-1, 1))

        if x1.direction == "right":
            jac1 = x1.jacobian_from_blocks(
                attitude=rho.T @ C_a1 @ att_group.odot(r_t1_1),
                position=rho.T @ C_a1,
            )
        elif x1.direction == "left":
            jac1 = x1.jacobian_from_blocks(
                attitude=rho.T @ att_group.odot(C_a1 @ r_t1_1 + r_1w_a),
                position=rho.T @ np.identity(r_t1_1.size),
            )

        if x2.direction == "right":
            jac2 = x2.jacobian_from_blocks(
                attitude=-rho.T @ C_a2 @ att_group.odot(r_t2_2),
                position=-rho.T @ C_a2,
            )
        elif x2.direction == "left":
            jac2 = x2.jacobian_from_blocks(
                attitude=-rho.T @ att_group.odot(C_a2 @ r_t2_2 + r_2w_a),
                position=-rho.T @ np.identity(r_t2_2.size),
            )

        return x.jacobian_from_blocks({self.state_id1: jac1, self.state_id2: jac2})

    def covariance(self, x: CompositeState) -> np.ndarray:
        return self._R


class RangeRelativePose(CompositeMeasurementModel):
    """
    Range model given a pose of another body relative to current pose.
    """

    def __init__(self, tag_body_position, nb_tag_body_position, nb_state_id, R):
        model = RangePoseToAnchor(tag_body_position, nb_tag_body_position, R)
        super(RangeRelativePose, self).__init__(model, nb_state_id)


class GlobalPosition(MeasurementModel):
    """
    Global, world-frame, or "absolute" position measurement.
    """

    def __init__(self, R: np.ndarray):
        self.R = R

    def evaluate(self, x: MatrixLieGroupState):
        return x.position

    def jacobian(self, x: MatrixLieGroupState):
        C_ab = x.attitude
        if C_ab.shape == (2, 2):
            att_group = SO2
        elif C_ab.shape == (3, 3):
            att_group = SO3

        if x.direction == "right":
            return x.jacobian_from_blocks(position=x.attitude)
        elif x.direction == "left":
            return x.jacobian_from_blocks(
                attitude=att_group.odot(x.position),
                position=np.identity(x.position.size),
            )

    def covariance(self, x: MatrixLieGroupState) -> np.ndarray:
        if np.isscalar(self.R):
            return self.R * np.identity(x.position.size)
        else:
            return self.R


class Altitude(MeasurementModel):
    def __init__(self, R: np.ndarray, minimum=0.0, bias=0.1):
        self.R = R
        self.minimum = minimum
        self.bias = bias

    def evaluate(self, x: MatrixLieGroupState):
        h = x.position[2] + self.bias
        return h if h > self.minimum else None

    def jacobian(self, x: MatrixLieGroupState):

        if x.direction == "right":
            return x.jacobian_from_blocks(
                position=x.attitude[2, :].reshape((1, -1))
            )
        elif x.direction == "left":
            return x.jacobian_from_blocks(
                attitude=SO3.odot(x.position)[2, :].reshape((1, -1)),
                position=np.array(([[0, 0, 1]])),
            )

    def covariance(self, x: MatrixLieGroupState) -> np.ndarray:
        return self.R


class Gravitometer(MeasurementModel):
    """
    Gravitometer model of the form

    .. math::

        \mathbf{y} = \mathbf{C}_{ab}^T \mathbf{g}_a + \mathbf{v}

    where :math:`\mathbf{g}_a` is the magnetic field vector in a world frame `a`.
    """

    def __init__(
        self, R: np.ndarray, gravity_vector: List[float] = [0, 0, -9.80665]
    ):
        """
        Parameters
        ----------
        R : np.ndarray
            Covariance associated with :math:`\mathbf{v}`
        gravity_vector : list[float] or numpy.ndarray, optional
            local magnetic field vector, by default [0, 0, -9.80665]
        """
        self.R = R
        self._g_a = np.array(gravity_vector).reshape((-1, 1))

    def evaluate(self, x: MatrixLieGroupState):
        return x.attitude.T @ self._g_a

    def jacobian(self, x: MatrixLieGroupState):
        if x.direction == "right":
            return x.jacobian_from_blocks(
                attitude=-SO3.odot(x.attitude.T @ self._g_a)
            )
        elif x.direction == "left":

            return x.jacobian_from_blocks(
                attitude=x.attitude.T @ -SO3.odot(self._g_a)
            )

    def covariance(self, x: MatrixLieGroupState) -> np.ndarray:
        return self.R


class Magnetometer(MeasurementModel):
    """
    Magnetometer model of the form

    .. math::

        \mathbf{y} = \mathbf{C}_{ab}^T \mathbf{m}_a + \mathbf{v}

    where :math:`\mathbf{m}_a` is the magnetic field vector in a world frame `a`.
    """

    def __init__(self, R: np.ndarray, magnetic_vector: List[float] = [1, 0, 0]):
        """

        Parameters
        ----------
        R : np.ndarray
            Covariance associated with :math:`\mathbf{v}`
        magnetic_vector : list[float] or numpy.ndarray, optional
            local magnetic field vector, by default [1, 0, 0]
        """
        self.R = R
        self._m_a = np.array(magnetic_vector).reshape((-1, 1))

    def evaluate(self, x: MatrixLieGroupState):
        return x.attitude.T @ self._m_a

    def jacobian(self, x: MatrixLieGroupState):
        if x.direction == "right":
            return x.jacobian_from_blocks(
                attitude=-SO3.odot(x.attitude.T @ self._m_a)
            )
        elif x.direction == "left":

            return x.jacobian_from_blocks(
                attitude=-x.attitude.T @ SO3.odot(self._m_a)
            )

    def covariance(self, x: MatrixLieGroupState) -> np.ndarray:
        return self.R


class _InvariantInnovation(MeasurementModel):
    def __init__(
        self, y: np.ndarray, model: MeasurementModel, direction="right"
    ):
        self.measurement_model = model
        self.y = y.ravel()
        self.direction = direction

    def evaluate(self, x: MatrixLieGroupState) -> np.ndarray:
        y_hat = self.measurement_model.evaluate(x)
        e: np.ndarray = y_hat.ravel() - self.y.ravel()

        if self.direction == "left":
            z = x.attitude.T @ e
        elif self.direction == "right":
            z = x.attitude @ e

        return z

    def jacobian(self, x: MatrixLieGroupState) -> np.ndarray:
        G = self.measurement_model.jacobian(x)

        if self.direction == "left":
            jac = x.attitude.T @ G
        elif self.direction == "right":
            jac = x.attitude @ G
        return jac

    def covariance(self, x: MatrixLieGroupState) -> np.ndarray:

        R = np.atleast_2d(self.measurement_model.covariance(x))

        if self.direction == "left":
            M = x.attitude.T
            cov = M @ R @ M.T
        elif self.direction == "right":
            M = x.attitude
            cov = M @ R @ M.T
        return cov


class InvariantMeasurement(Measurement):
    """
    Given a Measurement object, the class will construct a
    left- or right-invariant innovation ready to be fused into a state estimator.

    If a right-invariant innovation is chosen then the following will be formed.

    .. math::
        \mathbf{z} &= \\bar{\mathbf{X}}(\mathbf{y} - \\bar{\mathbf{y}})

        &= \\bar{\mathbf{X}}(\mathbf{g}(\mathbf{X}) +
        \mathbf{v} - \mathbf{g}(\\bar{\mathbf{X}}))

        &\\approx \\bar{\mathbf{X}}( \mathbf{g}(\\bar{\mathbf{X}})
        + \mathbf{G}\delta \mathbf{\\xi} + \mathbf{v}
        - \mathbf{g}(\\bar{\mathbf{X}}))

        &= \\bar{\mathbf{X}}\mathbf{G}\delta \mathbf{\\xi}
        + \\bar{\mathbf{X}}\mathbf{v}

    and hence :math:`\\bar{\mathbf{X}}\mathbf{G}` is the Jacobian of
    :math:`\mathbf{z}`, where :math:`\mathbf{G}` is the Jacobian of
    :math:`\mathbf{g}(\mathbf{X})`.  Similarly, if a left-invariant innovation is chosen,

     .. math::
        \mathbf{z} &= \\bar{\mathbf{X}}^{-1}(\mathbf{y} - \\bar{\mathbf{y}})

        &\\approx \\bar{\mathbf{X}}^{-1}\mathbf{G}\delta \mathbf{\\xi}
        + \\bar{\mathbf{X}}^{-1}\mathbf{v}

    and hence :math:`\\bar{\mathbf{X}}^{-1}\mathbf{G}` is the Jacobian of
    :math:`\mathbf{z}`.
    """

    def __init__(self, meas: Measurement, direction, model=None):
        """
        Parameters
        ----------
        meas : Measurement
            Measurement value
        direction : "left" or "right", optional
            whether to form a left- or right-invariant innovation, by default "right"
        model : MeasurementModel, optional
            a measurement model that directly returns the innovation and
            Jacobian and covariance of the innovation. If none is supplied,
            the default InvariantInnovation will be used, which computes the
            Jacobian of the innovation indirectly via chain rule.
        """

        if model is None:
            model = _InvariantInnovation(meas.value, meas.model, direction)

        super(InvariantMeasurement, self).__init__(
            value=np.zeros((meas.value.size,)),
            stamp=meas.stamp,
            model=model,
        )

