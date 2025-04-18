"""
Implementation of the differential equation operator estimator.

This module implements the mathematician's approach:
du/dt = G*F

Where:
- u is the user preference
- F is the external factor matrix (view time, click rate)
- G is the operator to be estimated
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class OperatorEstimator:
    def __init__(self, regularization=0.01):
        """
        Estimates operator G from differential equation du/dt = G*F
        
        Args:
            regularization (float): Regularization strength for stability
        """
        self.G = None
        self.model = LinearRegression()
        self.regularization = regularization
        
    def fit(self, preferences, factors, time_intervals):
        """
        Estimate G operator using time series of preferences and factors
        
        Args:
            preferences (array): Time series of user preferences (u)
            factors (array): Matrix of factors (F) with columns [view_time, click_rate]
            time_intervals (array): Time differences between consecutive interactions
            
        Returns:
            self: Fitted estimator
        """
        print(f"Fitting operator estimator with {len(preferences)} data points")
        
        # Ensure inputs are numpy arrays
        preferences = np.array(preferences)
        factors = np.array(factors)
        time_intervals = np.array(time_intervals)
        
        # Need at least 2 points to compute a derivative
        if len(preferences) < 2:
            print("Warning: Not enough preference data points")
            # Initialize with default values
            self.G = np.array([0.1, 0.1])
            return self
            
        # Calculate du/dt as finite differences
        du_dt = np.zeros(len(preferences) - 1)
        for i in range(len(preferences) - 1):
            # Avoid division by zero
            dt = max(time_intervals[i], 1e-8)
            du_dt[i] = (preferences[i + 1] - preferences[i]) / dt
        
        # Use factors[:-1] to align with du_dt
        F = factors[:-1]
        
        # Add regularization to handle potential instability
        F_reg = F + np.random.normal(0, self.regularization, F.shape)
        
        # Print diagnostics
        print(f"Preference range: [{preferences.min():.4f}, {preferences.max():.4f}]")
        print(f"du/dt range: [{du_dt.min():.4f}, {du_dt.max():.4f}]")
        print(f"Factors shape: {F.shape}")
        
        # Fit linear model: du/dt = G*F
        try:
            self.model.fit(F_reg, du_dt)
            self.G = self.model.coef_
            print(f"Estimated G: {self.G}")
        except Exception as e:
            print(f"Error fitting model: {e}")
            # Initialize with default values
            self.G = np.array([0.1, 0.1])
        
        return self
        
    def predict_preference_evolution(self, initial_preference, factors, time_intervals):
        """
        Predict user preference evolution using the estimated G operator
        
        Args:
            initial_preference (float): Starting user preference value
            factors (array): Matrix of factors [view_time, click_rate]
            time_intervals (array): Time differences between consecutive interactions
            
        Returns:
            array: Predicted preference values over time
        """
        if self.G is None:
            raise ValueError("Model must be fitted before prediction")
            
        # Ensure inputs are numpy arrays
        factors = np.array(factors)
        time_intervals = np.array(time_intervals)
        
        # Initialize preferences array
        preferences = np.zeros(len(factors) + 1)
        preferences[0] = initial_preference
        
        # Evolve preferences using the differential equation
        for i in range(len(factors)):
            # Calculate du/dt = G*F
            du_dt = np.dot(self.G, factors[i])
            
            # Ensure time interval is positive
            dt = max(time_intervals[i], 1e-8)
            
            # Update preference: u_{t+1} = u_t + (du/dt) * Δt
            preferences[i + 1] = preferences[i] + du_dt * dt
            
            # Ensure preference stays in [0, 1] range
            preferences[i + 1] = max(0.001, min(0.999, preferences[i + 1]))
            
        return preferences
    
    def calculate_preference_derivative(self, preference, factor):
        """
        Calculate the instantaneous rate of change of preference using G*F
        
        Args:
            preference (float): Current preference value
            factor (array): Current factor values [view_time, click_rate]
            
        Returns:
            float: Rate of change of preference (du/dt)
        """
        if self.G is None:
            raise ValueError("Model must be fitted before calculation")
            
        # Calculate du/dt = G*F
        du_dt = np.dot(self.G, factor)
        
        return du_dt
