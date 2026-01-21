import numpy as np


def ticks2vel(leftEnc, rightEnc, encoderResolution, timeStep, wheelRadius, wheelBase):
    """
    Convert accumulated encoder ticks into robot forward velocity and yaw rate.

    Inputs
    ------
    leftEnc, rightEnc : np.ndarray, shape (N,)
        Accumulated encoder ticks for left/right wheel.
    encoderResolution : int
        Ticks per wheel revolution (e.g., 4096).
    timeStep : float
        Sampling period in seconds.
    wheelRadius : float
        Wheel radius in meters (e.g., 0.1).
    wheelBase : float
        Distance between wheel centers in meters (e.g., 0.2).

    Returns
    -------
    vf : np.ndarray, shape (N-1,)
        Forward velocity of robot (m/s) over each interval k -> k+1.
    omega : np.ndarray, shape (N-1,)
        Yaw rate of robot (rad/s) over each interval k -> k+1.
    """
    # TODO: Implement
    # Ensure numpy arrays
    leftEnc = np.asarray(leftEnc, dtype=float)
    rightEnc = np.asarray(rightEnc, dtype=float)

    # 1. Encoder tick increments (ticks per timestep)
    leftTicks = np.diff(leftEnc)
    rightTicks = np.diff(rightEnc)

    # 2. Convert ticks -> wheel angular displacement (rad)
    #    2*pi rad per revolution, encoderResolution ticks per revolution
    dphi_l = (2.0 * np.pi / encoderResolution) * leftTicks
    dphi_r = (2.0 * np.pi / encoderResolution) * rightTicks

    # 3. Wheel angular velocity (rad/s)
    omega_l = dphi_l / timeStep
    omega_r = dphi_r / timeStep

    # 4. Wheel linear velocities (m/s)
    v_l = wheelRadius * omega_l
    v_r = wheelRadius * omega_r

    # 5. Differential drive kinematics
    vf = 0.5 * (v_r + v_l)
    omega = (v_r - v_l) / wheelBase

    return vf, omega


def dead_reckoning_from_encoders(leftEnc, rightEnc, dt, encoderResolution, wheelRadius, wheelBase, x0, y0, th0):
    """
    Dead-reckon robot pose from encoder ticks.

    Inputs
    ------
    leftEnc, rightEnc : np.ndarray, shape (N,)
        Accumulated ticks.
    dt : float
        Sampling period (seconds).
    encoderResolution : int
        ticks/rev
    wheelRadius : float
        meters
    wheelBase : float
        meters
    x0, y0, th0 : float
        initial pose

    Returns
    -------
    t : np.ndarray, shape (N,)
        timestamps starting at 0
    x, y, th : np.ndarray, shape (N,)
        estimated pose at each sample time (aligned with encoders)
    vf, omega : np.ndarray, shape (N-1,)
        estimated body velocities for each interval
    """
    # TODO: Implement
    # Convert to numpy
    leftEnc = np.asarray(leftEnc, dtype=float)
    rightEnc = np.asarray(rightEnc, dtype=float)

    N = len(leftEnc)
    if len(rightEnc) != N:
        raise ValueError("leftEnc and rightEnc must have same length")

    # 1. Compute body velocities
    vf, omega = ticks2vel(
        leftEnc, rightEnc,
        encoderResolution, dt,
        wheelRadius, wheelBase
    )

    # 2. Allocate state arrays
    x = np.zeros(N)
    y = np.zeros(N)
    th = np.zeros(N)

    # Initial condition
    x[0] = x0
    y[0] = y0
    th[0] = th0

    # 3. Integrate pose
    for k in range(N - 1):
        dtheta = omega[k] * dt
        ds = vf[k] * dt

        # Midpoint heading
        th_mid = th[k] + 0.5 * dtheta

        x[k + 1] = x[k] + ds * np.cos(th_mid)
        y[k + 1] = y[k] + ds * np.sin(th_mid)
        th[k + 1] = th[k] + dtheta

    # 4. Time vector
    t = np.arange(N) * dt

    return t, x, y, th, vf, omega
