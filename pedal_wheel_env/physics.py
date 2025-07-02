"""
Physics engine for the Pedal Wheel Environment.
Handles wheel dynamics, pedal forces, tilt, and energy calculations.
"""

import numpy as np
try:
    from .config import *
except ImportError:
    from config import *


class PedalWheelPhysics:
    """Physics engine for the pedal-controlled unicycle."""
    
    def __init__(self):
        """Initialize the physics engine."""
        self.reset()
    
    def reset(self):
        """Reset the physics state to initial conditions."""
        # Position and velocity
        self.x = 0.0  # Horizontal position
        self.y = GROUND_Y - WHEEL_RADIUS  # Vertical position (wheel on ground)
        self.vx = 0.0  # Horizontal velocity
        self.vy = 0.0  # Vertical velocity
        
        # Angular state
        self.theta = 0.0  # Tilt angle (0 = upright)
        self.omega = 0.0  # Angular velocity
        
        # Wheel rotation
        self.wheel_angle = 0.0  # Wheel rotation angle
        self.wheel_omega = 0.0  # Wheel angular velocity
        
        # Pedal state
        self.left_pedal_angle = 0.0
        self.right_pedal_angle = 0.0
        
        # Energy tracking
        self.total_energy_used = 0.0
    
    def step(self, left_pedal_force, right_pedal_force):
        """
        Advance physics simulation by one time step.
        
        Args:
            left_pedal_force (float): Force on left pedal [-1.0, 1.0]
            right_pedal_force (float): Force on right pedal [-1.0, 1.0]
        
        Returns:
            dict: Updated state information
        """
        # Convert normalized forces to actual forces
        left_force = left_pedal_force * MAX_PEDAL_FORCE
        right_force = right_pedal_force * MAX_PEDAL_FORCE
        
        # Calculate forward force from sum of pedal forces
        forward_force = (left_force + right_force) * 0.5
        
        # Calculate balance torque from difference of pedal forces
        balance_torque = (right_force - left_force) * PEDAL_LENGTH * 0.5
        
        # Calculate wheel torque and acceleration
        wheel_torque = forward_force * WHEEL_RADIUS - PEDAL_FRICTION * self.wheel_omega
        wheel_alpha = wheel_torque / (0.5 * WHEEL_MASS * WHEEL_RADIUS**2)
        
        # Update wheel angular velocity and angle
        self.wheel_omega += wheel_alpha * TIME_STEP
        self.wheel_angle += self.wheel_omega * TIME_STEP
        
        # Calculate forward acceleration from wheel rotation
        forward_acceleration = wheel_alpha * WHEEL_RADIUS
        
        # Calculate tilt dynamics
        # Gravity creates restoring torque when tilted
        gravity_torque = -TOTAL_MASS * GRAVITY * np.sin(self.theta) * WHEEL_RADIUS
        
        # Forward motion creates gyroscopic effect
        gyroscopic_torque = self.vx * self.wheel_omega * WHEEL_MASS * WHEEL_RADIUS
        
        # Net torque on the frame (include balance torque)
        frame_torque = gravity_torque + gyroscopic_torque + balance_torque
        
        # Moment of inertia for the frame (simplified)
        frame_moment_of_inertia = FRAME_MASS * WHEEL_RADIUS**2
        
        # Angular acceleration
        alpha = frame_torque / frame_moment_of_inertia
        
        # Update angular state
        self.omega += alpha * TIME_STEP
        self.theta += self.omega * TIME_STEP
        
        # Update linear motion
        self.vx += forward_acceleration * TIME_STEP
        
        # Apply friction to horizontal motion
        friction = -0.1 * self.vx
        self.vx += friction * TIME_STEP
        
        # Update position
        self.x += self.vx * TIME_STEP
        
        # Keep wheel on ground
        self.y = GROUND_Y - WHEEL_RADIUS
        
        # Update pedal angles based on wheel rotation
        self.left_pedal_angle = self.wheel_angle
        self.right_pedal_angle = self.wheel_angle + np.pi
        
        # Track energy usage
        energy_used = (abs(left_force) + abs(right_force)) * TIME_STEP
        self.total_energy_used += energy_used
        
        # Clamp values to prevent numerical instability
        self.vx = np.clip(self.vx, -MAX_FORWARD_VELOCITY, MAX_FORWARD_VELOCITY)
        self.omega = np.clip(self.omega, -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)
        self.wheel_omega = np.clip(self.wheel_omega, -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)
        
        # Keep position within world bounds
        self.x = np.clip(self.x, 0, WORLD_WIDTH)
        
        return self.get_state()
    
    def get_state(self):
        """
        Get the current physics state.
        
        Returns:
            dict: Current state information
        """
        return {
            'x': self.x,
            'y': self.y,
            'vx': self.vx,
            'vy': self.vy,
            'theta': self.theta,
            'omega': self.omega,
            'wheel_angle': self.wheel_angle,
            'wheel_omega': self.wheel_omega,
            'left_pedal_angle': self.left_pedal_angle,
            'right_pedal_angle': self.right_pedal_angle,
            'total_energy_used': self.total_energy_used
        }
    
    def is_fallen(self):
        """
        Check if the unicycle has fallen over.
        
        Returns:
            bool: True if fallen, False otherwise
        """
        return abs(self.theta) > MAX_TILT_ANGLE
    
    def get_observation(self):
        """
        Get the observation vector for the agent.
        
        Returns:
            np.ndarray: Observation vector
        """
        return np.array([
            self.x / WORLD_WIDTH,  # Normalized position
            self.vx / MAX_FORWARD_VELOCITY,  # Normalized velocity
            self.theta / MAX_TILT_ANGLE,  # Normalized tilt angle
            self.omega / MAX_ANGULAR_VELOCITY,  # Normalized angular velocity
            self.wheel_omega / MAX_ANGULAR_VELOCITY,  # Normalized wheel angular velocity
        ], dtype=np.float32)
    
    def get_reward(self):
        """
        Calculate the reward for the current state.
        
        Returns:
            float: Reward value
        """
        reward = REWARD_PER_STEP
        
        # Bonus for forward velocity
        if self.vx > 0:
            reward += VELOCITY_BONUS_FACTOR * self.vx
        
        # Penalty for energy usage
        energy_penalty = ENERGY_PENALTY_FACTOR * self.total_energy_used
        reward -= energy_penalty
        
        # Penalty for falling
        if self.is_fallen():
            reward += FALL_PENALTY
        
        return reward 