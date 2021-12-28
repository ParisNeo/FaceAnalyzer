"""=== Face Analyzer Helpers =>
    Module : ui
    Author : Saifeddine ALOUI (ParisNeo)
    Licence : MIT
    Description :
        Kalman filter, useful to do adaptive signal filtering
        It operates on linear systems in form of Marcov chains (a state x_t only depends on the state x_t-1 and the measurement z_t)
<================"""
import numpy as np

class KalmanFilter():
    """A simple linear Kalman filter
    """
    def __init__(self, Q:np.ndarray, R:np.ndarray, x0:np.ndarray, P0:np.ndarray, F, H) -> None:
        """Initializes a simple linear Kalman Filter

        Args:
            Q (np.ndarray): The state noise covariance matrix (how uncertain our state model is)
            R (np.ndarray): The measurement noise covariance matrix (how uncertain our measurement model is)
            x0 (np.ndarray): The initial state mean
            P0 (np.ndarray): The initial state covariance matrix (how uncertain our initial state is)
            F ([type]): state evolution matrix (x_n+1 = F*x_n)
            H ([type]): measurement matrix z_n = H*x_n
        """
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        self.F = F
        self.H = H


    def process(self, z:np.ndarray):
        """Processes a new measuremnt and updates the current state

        Args:
            z (np.ndarray): The measurement at current instant
        """
        # Predict
        self.x_ = self.F@self.x
        self.P_ = self.F@self.P@self.F.T+self.Q
        # Update
        y_ = z-self.H@self.x_                   # innovation
        S = self.R+ self.H@self.P_@self.H.T     # innovation covariance 
        K = self.P_@self.H@np.linalg.inv(S)     # Kalman gain
        self.x = self.x_ + K@y_                 # update state
        self.P = self.P_ - K@self.H@self.P_     # update state covariance

    