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
    
