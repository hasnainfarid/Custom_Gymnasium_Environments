#!/usr/bin/env python3
"""
Explosive Turret Environment with Destructible Targets
A realistic Box2D + Pygame environment where a turret shoots explosive shells at destructible targets.
"""

import gymnasium as gym
import numpy as np
import pygame
import Box2D
from Box2D import b2World, b2BodyDef, b2Body, b2CircleShape, b2PolygonShape, b2Vec2, b2_staticBody, b2_dynamicBody
from gymnasium import spaces
import random
import math
import time

class ExplosionParticle:
    """Simple particle for explosion effects"""
    def __init__(self, x, y, vx, vy, color, lifetime=60):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.lifetime = lifetime
        self.initial_lifetime = lifetime
        self.size = random.randint(2, 5)
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.5  # Gravity effect
        self.lifetime -= 1
        self.vx *= 0.98  # Air resistance
        
    def draw(self, screen):
        if self.lifetime > 0:
            alpha = self.lifetime / self.initial_lifetime
            size = int(self.size * alpha)
            if size > 0:
                # Create a surface with alpha for transparency
                particle_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                alpha_value = int(255 * alpha)
                color_with_alpha = (*self.color, alpha_value)
                try:
                    pygame.draw.circle(particle_surf, color_with_alpha, (size, size), size)
                except Exception as e:
                    print(f"Error drawing explosion particle: {e}, color: {color_with_alpha}, base color: {self.color}")
                    raise
                screen.blit(particle_surf, (int(self.x) - size, int(self.y) - size))

class TurretEnv(gym.Env):
    """
    Explosive Turret shooting environment using Box2D physics and Pygame visualization.
    
    Actions:
        - angle: float [0, pi/2] - angle to fire the shell
        - force: float [0, 100] - force applied to the shell
    
    Rewards:
        - +10 for direct hit on target
        - +5 bonus for complete destruction
        - 0 for miss
    """
    
    def __init__(self, render_mode="human"):
        super().__init__()
        
        # Environment parameters
        self.screen_width = 800
        self.screen_height = 600
        self.ground_height = 20  # Visual and physical ground height
        self.scale = 30  # pixels per meter
        self.time_step = 1.0 / 60.0  # 60 FPS physics
        
        # Box2D world setup with realistic gravity
        self.world = b2World(gravity=(0, -9.81))  # Standard Earth gravity
        
        # Game objects
        self.turret = None
        self.shell = None
        self.ground = None
        self.target_pieces = []  # List of target debris pieces
        self.target_destroyed = False
        
        # Turret parameters (bottom on ground)
        self.turret_x = 50  # pixels from left
        self.turret_width = 60
        self.turret_height = 100
        self.turret_y = self.screen_height - self.ground_height - self.turret_height // 2
        
        # Target parameters (bottom on ground)
        self.target_width = 80
        self.target_height = 300  # much taller for vertical wall
        self.target_x = self.screen_width - 100  # pixels from right
        self.target_y = self.screen_height - self.ground_height - self.target_height / 2
        
        # Shell parameters - much heavier for dramatic impact
        self.shell_radius = 20  # pixels (bigger shell)
        self.shell_mass = 5.0  # kg (much heavier shell)
        
        # Explosion and visual effects
        self.explosion_particles = []
        self.explosion_occurred = False
        self.impact_point = None
        self.screen_shake = 0
        self.shell_trail = []  # Trail effect for shell
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),  # [angle, force]
            high=np.array([np.pi/2, 100.0]),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -200, -200, 0, 0, 0]),  # Added target_destroyed flag
            high=np.array([self.screen_width, self.screen_height, 200, 200, 
                          self.screen_width, self.screen_height, 1]),
            dtype=np.float32
        )
        
        # Pygame setup
        self.render_mode = render_mode
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Explosive Turret Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
            self.big_font = pygame.font.Font(None, 72)
        
        # Episode tracking
        self.episode_step = 0
        self.max_episode_steps = 600  # 10 seconds at 60 FPS
        self.shell_launched = False
        self.shell_landed = False
        self.shell_out_of_bounds = False
        self.firing_angle = 0.0
        self.firing_force = 0.0
        self.total_reward = 0
        
        # Initialize the environment
        self._create_world()
        self.barrel_radius = 28  # For thick round barrel
    
    def _create_world(self):
        """Create Box2D world objects."""
        # Create ground (at very bottom)
        ground_def = b2BodyDef()
        ground_def.position = (self.screen_width / 2 / self.scale, 0)
        ground_def.type = b2_staticBody
        self.ground = self.world.CreateBody(ground_def)
        
        ground_shape = b2PolygonShape(box=(self.screen_width / self.scale, self.ground_height / self.scale))
        self.ground.CreateFixture(shape=ground_shape, density=0, friction=0.8, restitution=0.1)
        
        # Create turret (static body)
        turret_def = b2BodyDef()
        turret_def.position = (self.turret_x / self.scale, (self.screen_height - self.turret_y) / self.scale)
        turret_def.type = b2_staticBody
        self.turret = self.world.CreateBody(turret_def)
        
        turret_shape = b2PolygonShape(box=(self.turret_width / 2 / self.scale, self.turret_height / 2 / self.scale))
        self.turret.CreateFixture(shape=turret_shape, density=0)
        
        # Create initial target structure
        self._create_target()
    
    def _create_target(self):
        """Create the target as separate pieces that will be dynamic from the start."""
        # Clear any existing pieces
        for piece in self.target_pieces:
            self.world.DestroyBody(piece['body'])
        self.target_pieces = []
        self.target_destroyed = False
        
        # Create target as multiple pieces (for realistic destruction)
        # Each piece starts as dynamic but with very low mass for easy destruction
        piece_width = self.target_width / 2  # fewer columns for a wall
        piece_height = self.target_height / 10  # more rows for a tall wall
        num_cols = 2
        num_rows = 10
        for i in range(num_cols):
            for j in range(num_rows):
                piece_x = self.target_x - self.target_width/2 + piece_width/2 + i * piece_width
                # Place bottom row on ground
                piece_y = self.screen_height - self.ground_height - piece_height/2 - j * piece_height
                piece_def = b2BodyDef()
                piece_def.position = (piece_x / self.scale, (self.screen_height - piece_y) / self.scale)
                piece_def.type = b2_dynamicBody  # Dynamic from start
                piece_def.linearDamping = 0.05  # Less damping for more dramatic movement
                piece_body = self.world.CreateBody(piece_def)
                piece_shape = b2PolygonShape(box=(piece_width/2 / self.scale, piece_height/2 / self.scale))
                piece_body.CreateFixture(shape=piece_shape, density=0.3, friction=0.3, restitution=0.4)  # Even lower density
                
                # Initially set very low velocity to keep pieces in place
                piece_body.linearVelocity = b2Vec2(0, 0)
                piece_body.angularVelocity = 0
                
                self.target_pieces.append({
                    'body': piece_body,
                    'width': piece_width,
                    'height': piece_height,
                    'color': (139, 69, 19),  # Brown
                    'is_dynamic': True,  # Always dynamic now
                    'damage': 0.0,  # Track damage level
                    'hit_time': 0.0  # Track when hit
                })
    
    def _create_shell(self, angle, force):
        """Create and launch an explosive shell."""
        if self.shell is not None:
            self.world.DestroyBody(self.shell)
        
        # Shell starting position (at the edge of the round barrel)
        tip_x = self.turret_x + self.turret_width // 2 + self.barrel_radius * math.cos(angle)
        tip_y = self.turret_y - self.barrel_radius * math.sin(angle)
        shell_x = tip_x
        shell_y = tip_y
        
        # Create shell body
        shell_def = b2BodyDef()
        shell_def.position = (shell_x / self.scale, (self.screen_height - shell_y) / self.scale)
        shell_def.type = b2_dynamicBody
        shell_def.linearDamping = 0.1
        shell_def.allowSleep = False
        shell_def.bullet = True  # For better collision detection
        self.shell = self.world.CreateBody(shell_def)
        
        # Create shell shape - much heavier for dramatic impact
        shell_shape = b2CircleShape(radius=self.shell_radius / self.scale)
        self.shell.CreateFixture(
            shape=shell_shape, 
            density=self.shell_mass * 3,  # Much heavier shell
            friction=0.9,  # High friction to stop quickly
            restitution=0.05  # Very low bounce
        )
        
        # Apply initial velocity
        velocity_x = float(force * math.cos(angle))
        velocity_y = float(force * math.sin(angle))
        self.shell.linearVelocity = b2Vec2(velocity_x, velocity_y)
        
        self.shell_launched = True
        self.shell_landed = False
        self.shell_out_of_bounds = False
        self.explosion_occurred = False
        self.shell_trail = []  # Reset trail
        
        # Store firing parameters
        self.firing_angle = angle
        self.firing_force = force
    
    def _check_shell_collision(self):
        """Check if shell hit the target and trigger explosion."""
        if self.shell is None or self.explosion_occurred:
            return False
        
        shell_pos = self.shell.position
        shell_x = shell_pos.x * self.scale
        shell_y = self.screen_height - (shell_pos.y * self.scale)
        
        # Check collision with each target piece
        hit_pieces = []
        for piece in self.target_pieces:
            piece_pos = piece['body'].position
            piece_x = piece_pos.x * self.scale
            piece_y = self.screen_height - (piece_pos.y * self.scale)
            
            # More precise collision check
            dx = abs(shell_x - piece_x)
            dy = abs(shell_y - piece_y)
            
            if dx < (self.shell_radius + piece['width']/2) and dy < (self.shell_radius + piece['height']/2):
                # Calculate distance for damage calculation
                distance = math.sqrt(dx*dx + dy*dy)
                hit_pieces.append((piece, distance))
        
        if hit_pieces:
            # Sort by distance (closest first)
            hit_pieces.sort(key=lambda x: x[1])
            closest_piece, distance = hit_pieces[0]
            
            # Apply direct impact force to the closest piece
            shell_velocity = math.sqrt(self.shell.linearVelocity.x**2 + self.shell.linearVelocity.y**2)
            impact_force = shell_velocity * 1000  # Direct impact force
            
            # Apply force in the direction of shell travel
            shell_vx = self.shell.linearVelocity.x
            shell_vy = -self.shell.linearVelocity.y
            closest_piece['body'].ApplyLinearImpulse(
                b2Vec2(shell_vx * 500, shell_vy * 500), 
                closest_piece['body'].worldCenter, 
                True
            )
            
            # Add rotation from impact
            closest_piece['body'].ApplyAngularImpulse(random.uniform(-200, 200), True)
            
            # Mark as damaged
            closest_piece['damage'] = 1.0
            closest_piece['hit_time'] = time.time()
            
            # Trigger explosion at impact point
            self._trigger_explosion(shell_x, shell_y, hit_pieces)
            return True
        
        return False
    
    def _trigger_explosion(self, x, y, hit_pieces=None):
        """Trigger explosion at given position with realistic damage physics."""
        self.explosion_occurred = True
        self.impact_point = (x, y)
        self.screen_shake = 20  # Screen shake intensity
        
        # Create explosion particles
        for _ in range(50):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 10)
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)
            
            # Mix of colors for explosion
            if random.random() < 0.3:
                color = (255, random.randint(200, 255), 0)  # Yellow
            elif random.random() < 0.6:
                color = (255, random.randint(100, 200), 0)  # Orange
            else:
                color = (255, 0, 0)  # Red
                
            particle = ExplosionParticle(x, y, vx, vy, color, lifetime=random.randint(20, 40))
            self.explosion_particles.append(particle)
        
        # Apply realistic explosion damage to target pieces
        explosion_force = 200.0  # Much stronger explosion
        explosion_radius = 200.0  # Larger damage radius in pixels
        
        for piece in self.target_pieces:
            piece_pos = piece['body'].position
            piece_x = piece_pos.x * self.scale
            piece_y = self.screen_height - (piece_pos.y * self.scale)
            
            dx = piece_x - x
            dy = piece_y - y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Check if piece is within explosion radius
            if distance <= explosion_radius:
                # Calculate damage based on distance and shell velocity
                shell_velocity = math.sqrt(self.shell.linearVelocity.x**2 + self.shell.linearVelocity.y**2)
                damage_factor = max(0.3, 1.0 - (distance / explosion_radius))  # Minimum damage increased
                
                # Apply much stronger explosion force
                force_magnitude = explosion_force * 2000 * damage_factor / (distance * distance + 1)  # Much stronger
                fx = (dx / distance) * force_magnitude
                fy = -(dy / distance) * force_magnitude  # Negative for Box2D coordinates
                
                # Add shell velocity influence (much stronger)
                shell_vx = self.shell.linearVelocity.x
                shell_vy = -self.shell.linearVelocity.y
                fx += shell_vx * 2.0 * damage_factor  # Much stronger shell influence
                fy += shell_vy * 2.0 * damage_factor
                
                # Add upward force for dramatic effect
                fy -= 5000 * damage_factor  # Strong upward force
                
                piece['body'].ApplyLinearImpulse(
                    b2Vec2(fx, fy), 
                    piece['body'].worldCenter, 
                    True
                )
                
                # Add much stronger rotation based on impact
                angular_impulse = random.uniform(-500, 500) * damage_factor  # Much stronger rotation
                piece['body'].ApplyAngularImpulse(angular_impulse, True)
                
                # Add damage state to piece
                piece['damage'] = damage_factor
                piece['hit_time'] = time.time()
        
        self.target_destroyed = True
    
    def _get_observation(self):
        """Get current observation."""
        if self.shell is None:
            shell_x = self.turret_x + self.turret_width // 2
            shell_y = self.turret_y - self.turret_height // 2
            shell_vx, shell_vy = 0, 0
        else:
            shell_pos = self.shell.position
            shell_vel = self.shell.linearVelocity
            shell_x = shell_pos.x * self.scale
            shell_y = self.screen_height - (shell_pos.y * self.scale)
            shell_vx = shell_vel.x
            shell_vy = -shell_vel.y
        
        return np.array([
            shell_x, shell_y, shell_vx, shell_vy,
            self.target_x, self.target_y,
            float(self.target_destroyed)
        ], dtype=np.float32)
    
    def _check_shell_landed(self):
        """Check if shell has landed."""
        if self.shell is None:
            return False
        
        velocity = self.shell.linearVelocity
        speed = math.sqrt(velocity.x**2 + velocity.y**2)
        
        shell_y = self.screen_height - (self.shell.position.y * self.scale)
        on_ground = shell_y >= self.screen_height - 40
        
        return speed < 1.0 and on_ground
    
    def _check_out_of_bounds(self):
        """Check if shell is out of bounds."""
        if self.shell is None:
            return False
        
        shell_pos = self.shell.position
        shell_x = shell_pos.x * self.scale
        shell_y = self.screen_height - (shell_pos.y * self.scale)
        
        out_left = shell_x < -50
        out_right = shell_x > self.screen_width + 50
        out_top = shell_y < -100
        
        return out_left or out_right or out_top
    
    def step(self, action):
        """Take a step in the environment."""
        # Only apply action if shell hasn't been launched yet
        if not self.shell_launched:
            angle, force = action
            self._create_shell(angle, force)
        
        # Store shell velocity BEFORE collision for impact calculation
        shell_impact_velocity = 0
        if self.shell is not None and self.shell_launched and not self.explosion_occurred:
            shell_impact_velocity = math.sqrt(self.shell.linearVelocity.x**2 + self.shell.linearVelocity.y**2)
        
        # Update shell trail
        if self.shell is not None and self.shell_launched and not self.explosion_occurred:
            shell_pos = self.shell.position
            shell_x = shell_pos.x * self.scale
            shell_y = self.screen_height - (shell_pos.y * self.scale)
            self.shell_trail.append((shell_x, shell_y))
            if len(self.shell_trail) > 20:  # Keep trail length limited
                self.shell_trail.pop(0)
        
        # Check for collision before physics step
        collision_occurred_this_step = False
        if self.shell_launched and not self.explosion_occurred:
            # Store velocity before collision
            if self._check_shell_collision():
                collision_occurred_this_step = True
                # Store the impact velocity
                self.impact_velocity = shell_impact_velocity
                # Mark explosion time
                self.explosion_time = self.episode_step
        
        # Step the physics world
        self.world.Step(self.time_step, 6, 2)
        self.world.ClearForces()
        
        # Update particles
        for particle in self.explosion_particles[:]:
            particle.update()
            if particle.lifetime <= 0:
                self.explosion_particles.remove(particle)
        
        # Update screen shake
        if self.screen_shake > 0:
            self.screen_shake -= 1
        
        # Update episode step
        self.episode_step += 1
        
        # Calculate reward AFTER physics step when pieces have actually moved
        reward = 0
        if collision_occurred_this_step:
            # Base reward for impact velocity (higher velocity = more reward)
            reward = min(50, self.impact_velocity * 0.5)  # Cap at 50 for velocity
            
            # Store initial reward
            self.initial_impact_reward = reward
            self.pieces_affected_count = 0
            self.total_piece_velocity = 0
        
        # Continue calculating reward for 2 seconds after explosion
        if self.explosion_occurred and hasattr(self, 'explosion_time'):
            time_since_explosion = self.episode_step - self.explosion_time
            
            if time_since_explosion <= 120:  # 2 seconds at 60 FPS
                # Calculate movement-based reward each frame
                frame_reward = 0
                moving_pieces = 0
                total_velocity = 0
                
                for piece in self.target_pieces:
                    v = piece['body'].linearVelocity
                    speed = math.sqrt(v.x**2 + v.y**2)
                    
                    # Count pieces that are moving significantly
                    if speed > 2.0:
                        moving_pieces += 1
                        total_velocity += speed
                        
                        # Bonus for pieces flying high
                        if v.y < -5:  # Negative y is upward in Box2D
                            frame_reward += 0.1
                
                # Add frame reward based on destruction
                if moving_pieces > 0:
                    frame_reward += moving_pieces * 0.2 + total_velocity * 0.05
                    reward += frame_reward
                    
                    # Update tracking
                    self.pieces_affected_count = max(self.pieces_affected_count, moving_pieces)
                    self.total_piece_velocity = max(self.total_piece_velocity, total_velocity)
            
            # Final reward summary when 2 seconds have passed
            if time_since_explosion == 120:
                # Bonus rewards
                if self.pieces_affected_count >= 15:  # Most pieces affected
                    reward += 20  # Massive destruction bonus
                elif self.pieces_affected_count >= 10:
                    reward += 10  # Good destruction bonus
                elif self.pieces_affected_count >= 5:
                    reward += 5   # Decent hit bonus
                
                print(f"  Impact Summary: Velocity={self.impact_velocity:.1f}, Pieces={self.pieces_affected_count}, Total Reward={reward:.1f}")
        
        # Store total reward
        if reward > 0:
            self.total_reward += reward
        
        # Check if shell has landed (but don't end episode immediately after explosion)
        if self.shell_launched and not self.shell_landed:
            if self._check_shell_landed() and not self.explosion_occurred:
                self.shell_landed = True
        
        # Check if shell is out of bounds
        if self.shell_launched and not self.shell_out_of_bounds:
            if self._check_out_of_bounds():
                self.shell_out_of_bounds = True
        
        # Get observation
        observation = self._get_observation()
        
        # Check if episode is done
        done = False
        truncated = False
        
        # Wait 2 seconds after explosion before ending episode
        if self.explosion_occurred and hasattr(self, 'explosion_time'):
            time_since_explosion = self.episode_step - self.explosion_time
            if time_since_explosion >= 120:  # 2 seconds at 60 FPS
                done = True
        elif (self.shell_landed or self.shell_out_of_bounds or 
              self.episode_step >= self.max_episode_steps):
            done = True
        
        # Additional info
        info = {
            'shell_launched': self.shell_launched,
            'shell_landed': self.shell_landed,
            'shell_out_of_bounds': self.shell_out_of_bounds,
            'target_hit': self.explosion_occurred,
            'target_destroyed': self.target_destroyed,
            'total_reward': self.total_reward,
            'moving_pieces': sum(1 for piece in self.target_pieces 
                               if math.sqrt(piece['body'].linearVelocity.x**2 + piece['body'].linearVelocity.y**2) > 2.0)
        }
        
        return observation, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset episode variables
        self.episode_step = 0
        self.shell_launched = False
        self.shell_landed = False
        self.shell_out_of_bounds = False
        self.explosion_occurred = False
        self.target_destroyed = False
        self.total_reward = 0
        self._explosion_rewarded = False
        
        # Reset impact tracking
        self.impact_velocity = 0
        self.explosion_time = None
        self.initial_impact_reward = 0
        self.pieces_affected_count = 0
        self.total_piece_velocity = 0
        
        # Reset damage tracking
        for piece in self.target_pieces:
            piece['damage'] = 0.0
            piece['hit_time'] = 0.0
        
        # Clear visual effects
        self.explosion_particles = []
        self.shell_trail = []
        self.screen_shake = 0
        self.impact_point = None
        
        # Remove old shell if it exists
        if self.shell is not None:
            self.world.DestroyBody(self.shell)
            self.shell = None
        
        # Recreate target
        self._create_target()
        
        # Reset physics world
        self.world.ClearForces()
        
        # Let physics stabilize for a few steps
        for _ in range(10):
            self.world.Step(self.time_step, 6, 2)
            self.world.ClearForces()
        
        return self._get_observation(), {}
    
    def render(self):
        """Render the environment using Pygame."""
        if self.render_mode != "human":
            return
        
        # Create display surface
        display_surface = pygame.Surface((self.screen_width, self.screen_height))
        
        # Castle/fortress background
        # Sky gradient (darker for serious feel)
        for y in range(self.screen_height // 2):
            color_value = int(80 + (y / (self.screen_height // 2)) * 40)
            r = min(255, color_value)
            g = min(255, color_value + 30)
            b = min(255, color_value + 60)
            pygame.draw.line(display_surface, (r, g, b), 
                           (0, y), (self.screen_width, y))
        
        # Draw distant mountains/hills
        mountain_points = [
            (0, self.screen_height // 2),
            (100, self.screen_height // 2 - 80),
            (200, self.screen_height // 2 - 40),
            (300, self.screen_height // 2 - 100),
            (400, self.screen_height // 2 - 60),
            (500, self.screen_height // 2 - 120),
            (600, self.screen_height // 2 - 80),
            (700, self.screen_height // 2 - 100),
            (800, self.screen_height // 2 - 60),
            (800, self.screen_height // 2),
            (0, self.screen_height // 2)
        ]
        pygame.draw.polygon(display_surface, (60, 60, 80), mountain_points)
        
        # Draw castle/fortress in background
        castle_x = 650
        castle_y = self.screen_height // 2
        castle_width = 120
        castle_height = 200
        
        # Main castle structure
        pygame.draw.rect(display_surface, (80, 80, 90), 
                        (castle_x, castle_y, castle_width, castle_height))
        
        # Castle towers
        tower_width = 25
        tower_height = 60
        # Left tower
        pygame.draw.rect(display_surface, (70, 70, 80), 
                        (castle_x - 15, castle_y - tower_height, tower_width, tower_height))
        # Right tower
        pygame.draw.rect(display_surface, (70, 70, 80), 
                        (castle_x + castle_width - 10, castle_y - tower_height, tower_width, tower_height))
        
        # Castle battlements (top edge)
        for i in range(0, castle_width, 15):
            pygame.draw.rect(display_surface, (60, 60, 70), 
                           (castle_x + i, castle_y, 8, 12))
        
        # Tower battlements
        for i in range(0, tower_width, 8):
            pygame.draw.rect(display_surface, (60, 60, 70), 
                           (castle_x - 15 + i, castle_y - tower_height, 6, 10))
            pygame.draw.rect(display_surface, (60, 60, 70), 
                           (castle_x + castle_width - 10 + i, castle_y - tower_height, 6, 10))
        
        # Castle windows
        pygame.draw.rect(display_surface, (40, 40, 50), 
                        (castle_x + 20, castle_y + 40, 15, 20))
        pygame.draw.rect(display_surface, (40, 40, 50), 
                        (castle_x + 60, castle_y + 40, 15, 20))
        pygame.draw.rect(display_surface, (40, 40, 50), 
                        (castle_x + 40, castle_y + 100, 15, 20))
        
        # Castle entrance
        pygame.draw.rect(display_surface, (30, 30, 40), 
                        (castle_x + 45, castle_y + castle_height - 40, 30, 40))
        
        # Draw some clouds for atmosphere
        cloud_positions = [(100, 80), (300, 60), (500, 90), (700, 70)]
        for cloud_x, cloud_y in cloud_positions:
            pygame.draw.circle(display_surface, (120, 120, 140), (cloud_x, cloud_y), 20)
            pygame.draw.circle(display_surface, (120, 120, 140), (cloud_x + 15, cloud_y), 15)
            pygame.draw.circle(display_surface, (120, 120, 140), (cloud_x + 30, cloud_y), 18)
        
        # Ground area (darker for serious feel)
        for y in range(self.screen_height // 2, self.screen_height):
            color_value = int(40 + ((y - self.screen_height // 2) / (self.screen_height // 2)) * 30)
            pygame.draw.line(display_surface, (color_value, color_value + 10, color_value + 5), 
                           (0, y), (self.screen_width, y))
        
        # Draw ground with texture
        pygame.draw.rect(display_surface, (34, 80, 34), 
                        (0, self.screen_height - self.ground_height, self.screen_width, self.ground_height))
        pygame.draw.rect(display_surface, (20, 50, 20), 
                        (0, self.screen_height - self.ground_height, self.screen_width, 5))
        
        # Draw military turret with enhanced appearance
        turret_rect = pygame.Rect(
            self.turret_x - self.turret_width // 2, self.turret_y - self.turret_height // 2, 
            self.turret_width, self.turret_height
        )
        
        # Main turret body (darker military green)
        pygame.draw.rect(display_surface, (50, 60, 50), turret_rect)
        
        # Turret armor plates
        pygame.draw.rect(display_surface, (70, 80, 70), 
                        (turret_rect.x + 5, turret_rect.y + 5, 15, turret_rect.height - 10))
        pygame.draw.rect(display_surface, (70, 80, 70), 
                        (turret_rect.x + turret_rect.width - 20, turret_rect.y + 5, 15, turret_rect.height - 10))
        
        # Turret details
        pygame.draw.rect(display_surface, (30, 40, 30), turret_rect, 3)
        
        # Draw thick military barrel
        barrel_center_x = self.turret_x + self.turret_width // 2
        barrel_center_y = self.turret_y
        
        # Barrel outer casing (dark metal)
        pygame.draw.circle(display_surface, (40, 40, 45), (int(barrel_center_x), int(barrel_center_y)), self.barrel_radius)
        
        # Barrel inner structure
        pygame.draw.circle(display_surface, (60, 60, 65), (int(barrel_center_x), int(barrel_center_y)), self.barrel_radius-6)
        
        # Barrel rifling/barrel rings
        for i in range(3):
            ring_radius = self.barrel_radius - 8 - i * 4
            pygame.draw.circle(display_surface, (80, 80, 85), (int(barrel_center_x), int(barrel_center_y)), ring_radius, 2)
        
        # Barrel muzzle
        pygame.draw.circle(display_surface, (20, 20, 25), (int(barrel_center_x), int(barrel_center_y)), self.barrel_radius-12)
        
        # Barrel outline
        pygame.draw.circle(display_surface, (0, 0, 0), (int(barrel_center_x), int(barrel_center_y)), self.barrel_radius, 2)
        
        # Draw firing direction indicator
        highlight_x = barrel_center_x + (self.barrel_radius-4) * math.cos(self.firing_angle if self.shell_launched else math.pi/4)
        highlight_y = barrel_center_y - (self.barrel_radius-4) * math.sin(self.firing_angle if self.shell_launched else math.pi/4)
        pygame.draw.circle(display_surface, (255, 255, 255), (int(highlight_x), int(highlight_y)), 4)
        
        # Add turret base/platform
        base_width = self.turret_width + 20
        base_height = 15
        base_rect = pygame.Rect(
            self.turret_x - base_width // 2, 
            self.turret_y + self.turret_height // 2, 
            base_width, base_height
        )
        pygame.draw.rect(display_surface, (60, 70, 60), base_rect)
        pygame.draw.rect(display_surface, (40, 50, 40), base_rect, 2)
        
        # Draw firing angle indicator
        if not self.shell_launched:
            start_x = self.turret_x + self.turret_width // 2 + 50
            start_y = self.turret_y
            default_angle = math.pi / 4
            for i in range(5):
                end_x = start_x + (20 + i * 20) * math.cos(default_angle)
                end_y = start_y - (20 + i * 20) * math.sin(default_angle)
                alpha = 255 - i * 40
                # Create surface for transparent circle
                indicator_surf = pygame.Surface((10, 10), pygame.SRCALPHA)
                pygame.draw.circle(indicator_surf, (255, 0, 0, alpha), 
                                 (5, 5), 3 - i // 2)
                display_surface.blit(indicator_surf, (int(end_x) - 5, int(end_y) - 5))
        
        # Draw target pieces
        for piece in self.target_pieces:
            piece_pos = piece['body'].position
            piece_x = piece_pos.x * self.scale
            piece_y = self.screen_height - (piece_pos.y * self.scale)
            angle = piece['body'].angle
            
            # Check if piece is moving
            velocity = piece['body'].linearVelocity
            speed = math.sqrt(velocity.x**2 + velocity.y**2)
            is_moving = speed > 2.0
            
            # Create rotated surface for piece
            piece_surf = pygame.Surface((piece['width'], piece['height']), pygame.SRCALPHA)
            
            # Draw piece with damage texture if dynamic
            if piece['is_dynamic']:
                # Calculate damage-based appearance
                damage = piece.get('damage', 0.0)
                hit_time = piece.get('hit_time', 0.0)
                time_since_hit = time.time() - hit_time
                
                # Add moving indicator
                if is_moving:
                    # Draw red outline for moving pieces
                    pygame.draw.rect(piece_surf, (255, 0, 0), 
                                   (0, 0, piece['width'], piece['height']), 3)
                
                # Damaged appearance based on damage level
                if damage > 0.7:
                    # Heavily damaged - dark and cracked
                    base_color = (60, 30, 10)
                    crack_color = (40, 20, 5)
                elif damage > 0.4:
                    # Moderately damaged - brown with cracks
                    base_color = (100, 50, 20)
                    crack_color = (80, 40, 15)
                else:
                    # Lightly damaged - lighter brown
                    base_color = (120, 60, 25)
                    crack_color = (100, 50, 20)
                
                try:
                    pygame.draw.rect(piece_surf, base_color, 
                                   (0, 0, piece['width'], piece['height']))
                    
                    # Add damage cracks
                    for _ in range(int(damage * 5) + 1):
                        crack_x = random.randint(2, int(piece['width'] - 2))
                        crack_y = random.randint(2, int(piece['height'] - 2))
                        crack_length = random.randint(3, int(piece['width'] // 2))
                        crack_angle = random.uniform(0, 2 * math.pi)
                        
                        end_x = crack_x + crack_length * math.cos(crack_angle)
                        end_y = crack_y + crack_length * math.sin(crack_angle)
                        
                        pygame.draw.line(piece_surf, crack_color, 
                                       (crack_x, crack_y), (end_x, end_y), 1)
                    
                    # Add fire effect for recently hit pieces
                    if time_since_hit < 2.0:  # Fire effect for 2 seconds
                        fire_intensity = max(0, 1.0 - time_since_hit / 2.0)
                        num_fire_particles = int(3 * fire_intensity)
                        
                        for _ in range(num_fire_particles):
                            fx = random.randint(0, int(piece['width']))
                            fy = random.randint(0, int(piece['height']))
                            fire_color = (255, random.randint(100, 200), 0)
                            try:
                                pygame.draw.circle(piece_surf, fire_color, (fx, fy), 2)
                            except Exception as e:
                                print(f"Error drawing fire circle: {e}, color: {fire_color}")
                                raise
                                
                except Exception as e:
                    print(f"Error drawing damaged piece rect: {e}, color: {base_color}")
                    raise
            else:
                # Fortress wall appearance
                try:
                    # Stone wall color
                    stone_color = (120, 110, 100)
                    pygame.draw.rect(piece_surf, stone_color, 
                                   (0, 0, piece['width'], piece['height']))
                    
                    # Add stone texture/bricks
                    brick_width = 8
                    brick_height = 6
                    for bx in range(0, int(piece['width']), brick_width):
                        for by in range(0, int(piece['height']), brick_height):
                            # Alternate brick pattern
                            if (bx // brick_width + by // brick_height) % 2 == 0:
                                brick_color = (100, 90, 80)
                            else:
                                brick_color = (140, 130, 120)
                            
                            pygame.draw.rect(piece_surf, brick_color,
                                           (bx, by, min(brick_width, int(piece['width']) - bx), 
                                            min(brick_height, int(piece['height']) - by)))
                    
                    # Add mortar lines
                    mortar_color = (80, 70, 60)
                    for bx in range(0, int(piece['width']), brick_width):
                        pygame.draw.line(piece_surf, mortar_color, 
                                       (bx, 0), (bx, piece['height']), 1)
                    for by in range(0, int(piece['height']), brick_height):
                        pygame.draw.line(piece_surf, mortar_color, 
                                       (0, by), (piece['width'], by), 1)
                        
                except Exception as e:
                    print(f"Error drawing fortress wall: {e}")
                    raise
            
            pygame.draw.rect(piece_surf, (0, 0, 0), 
                           (0, 0, piece['width'], piece['height']), 2)
            
            # Rotate and blit
            rotated_surf = pygame.transform.rotate(piece_surf, -math.degrees(angle))
            rot_rect = rotated_surf.get_rect(center=(int(piece_x), int(piece_y)))
            display_surface.blit(rotated_surf, rot_rect)
        
        # Draw shell trail
        if len(self.shell_trail) > 1:
            for i in range(1, len(self.shell_trail)):
                start_pos = self.shell_trail[i-1]
                end_pos = self.shell_trail[i]
                alpha = int(255 * (i / len(self.shell_trail)))
                width = 1 + i // 5
                # Create surface for transparent line
                trail_surf = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
                pygame.draw.line(trail_surf, (255, 200, 100, alpha), 
                               start_pos, end_pos, width)
                display_surface.blit(trail_surf, (0, 0))
        
        # Draw shell with glowing effect
        if self.shell is not None and self.shell_launched and not self.explosion_occurred:
            shell_pos = self.shell.position
            shell_x = int(shell_pos.x * self.scale)
            shell_y = int(self.screen_height - (shell_pos.y * self.scale))
            
            # Glow effect
            for i in range(3):
                glow_radius = self.shell_radius + 5 - i * 2
                glow_alpha = 50 + i * 30
                # Create surface for transparent glow
                glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (255, 150, 50, glow_alpha), 
                                 (glow_radius, glow_radius), glow_radius, 2)
                display_surface.blit(glow_surf, (shell_x - glow_radius, shell_y - glow_radius))
            
            # Shell body
            pygame.draw.circle(display_surface, (50, 50, 50), (shell_x, shell_y), self.shell_radius)
            pygame.draw.circle(display_surface, (30, 30, 30), (shell_x, shell_y), self.shell_radius, 3)
            pygame.draw.circle(display_surface, (255, 255, 255), 
                             (shell_x - 5, shell_y - 5), 3)
        
        # Draw explosion particles
        for particle in self.explosion_particles:
            particle.draw(display_surface)
        
        # Draw impact flash
        if self.explosion_occurred and self.impact_point and self.screen_shake > 15:
            flash_radius = 50 + (20 - self.screen_shake) * 5
            flash_alpha = int(min(255, self.screen_shake * 10))
            flash_surf = pygame.Surface((flash_radius * 2, flash_radius * 2), pygame.SRCALPHA)
            try:
                pygame.draw.circle(flash_surf, (255, 255, 200, flash_alpha), 
                                 (flash_radius, flash_radius), flash_radius)
            except Exception as e:
                print(f"Error drawing impact flash: {e}, color: {(255, 255, 200, flash_alpha)}")
                raise
            display_surface.blit(flash_surf, 
                               (self.impact_point[0] - flash_radius, 
                                self.impact_point[1] - flash_radius))
        
        # Apply screen shake
        if self.screen_shake > 0:
            offset_x = random.randint(-self.screen_shake, self.screen_shake)
            offset_y = random.randint(-self.screen_shake, self.screen_shake)
            self.screen.blit(display_surface, (offset_x, offset_y))
        else:
            self.screen.blit(display_surface, (0, 0))
        
        # Draw UI elements
        ui_y = 10
        
        # Episode info
        episode_text = self.font.render(f"Episode: {getattr(self, 'current_episode', 0)}", True, (255, 255, 255))
        self.screen.blit(episode_text, (10, ui_y))
        ui_y += 40
        
        # Firing parameters
        if self.shell_launched:
            angle_text = self.font.render(f"Angle: {self.firing_angle*180/math.pi:.1f}°", True, (255, 255, 255))
            force_text = self.font.render(f"Force: {self.firing_force:.1f} N", True, (255, 255, 255))
            self.screen.blit(angle_text, (10, ui_y))
            self.screen.blit(force_text, (10, ui_y + 30))
            ui_y += 70
        
        # Status
        if self.shell_launched:
            if self.explosion_occurred:
                status = "IMPACT!"
                status_color = (255, 100, 0)
            elif self.shell_out_of_bounds:
                status = "OUT OF BOUNDS"
                status_color = (255, 0, 0)
            elif self.shell_landed:
                status = "Missed"
                status_color = (255, 255, 0)
            else:
                status = "In Flight"
                status_color = (0, 255, 255)
        else:
            status = "Ready to Fire"
            status_color = (0, 255, 0)
        
        status_text = self.font.render(f"Status: {status}", True, status_color)
        self.screen.blit(status_text, (10, ui_y))
        ui_y += 40
        
        # Detailed reward information
        if self.explosion_occurred and hasattr(self, 'impact_velocity'):
            # Impact velocity
            impact_text = self.font.render(f"Impact Velocity: {self.impact_velocity:.1f} m/s", True, (255, 200, 0))
            self.screen.blit(impact_text, (10, ui_y))
            ui_y += 30
            
            # Pieces affected
            if hasattr(self, 'pieces_affected_count'):
                pieces_text = self.font.render(f"Pieces Affected: {self.pieces_affected_count}/20", True, (255, 150, 0))
                self.screen.blit(pieces_text, (10, ui_y))
                ui_y += 30
            
            # Total piece velocity
            if hasattr(self, 'total_piece_velocity'):
                velocity_text = self.font.render(f"Total Velocity: {self.total_piece_velocity:.1f} m/s", True, (255, 100, 0))
                self.screen.blit(velocity_text, (10, ui_y))
                ui_y += 30
        
        # Score display with enhanced styling
        if self.total_reward > 0:
            # Determine score color based on performance
            if self.total_reward >= 80:
                score_color = (0, 255, 0)  # Green for excellent
                score_bg_color = (0, 100, 0)
            elif self.total_reward >= 50:
                score_color = (255, 255, 0)  # Yellow for good
                score_bg_color = (100, 100, 0)
            elif self.total_reward >= 20:
                score_color = (255, 165, 0)  # Orange for decent
                score_bg_color = (100, 65, 0)
            else:
                score_color = (255, 100, 100)  # Red for poor
                score_bg_color = (100, 0, 0)
            
            score_text = self.big_font.render(f"SCORE: {self.total_reward:.0f}", True, score_color)
            text_rect = score_text.get_rect(center=(self.screen_width // 2, 100))
            
            # Draw background for score with color coding
            pygame.draw.rect(self.screen, score_bg_color, text_rect.inflate(20, 10))
            pygame.draw.rect(self.screen, score_color, text_rect.inflate(20, 10), 3)
            self.screen.blit(score_text, text_rect)
            
            # Add performance indicator
            if self.total_reward >= 80:
                perf_text = self.font.render("PERFECT DESTRUCTION!", True, (0, 255, 0))
            elif self.total_reward >= 50:
                perf_text = self.font.render("GREAT HIT!", True, (255, 255, 0))
            elif self.total_reward >= 20:
                perf_text = self.font.render("GOOD HIT!", True, (255, 165, 0))
            else:
                perf_text = self.font.render("WEAK HIT", True, (255, 100, 100))
            
            perf_rect = perf_text.get_rect(center=(self.screen_width // 2, 150))
            pygame.draw.rect(self.screen, (0, 0, 0), perf_rect.inflate(20, 10))
            self.screen.blit(perf_text, perf_rect)
        
        # Instructions
        if not self.shell_launched:
            instruction_text = self.font.render("Press SPACE to fire!", True, (255, 255, 255))
            inst_rect = instruction_text.get_rect(center=(self.screen_width // 2, self.screen_height - 40))
            pygame.draw.rect(self.screen, (0, 0, 0), inst_rect.inflate(20, 10))
            self.screen.blit(instruction_text, inst_rect)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(60)
    
    def close(self):
        """Close the environment."""
        if self.render_mode == "human":
            pygame.quit()

''' Random Agent Test

def random_agent(env, num_episodes=5):
    """Run a random agent for the specified number of episodes."""
    print(f"Starting Explosive Turret Environment - {num_episodes} episodes")
    print("=" * 50)
    
    total_reward = 0
    successful_hits = 0
    perfect_destructions = 0
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print("-" * 30)
        
        env.current_episode = episode + 1
        observation, info = env.reset()
        
        episode_reward = 0
        step = 0
        
        while True:
            # Render the environment
            env.render()
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
            
            # Take random action
            action = env.action_space.sample()
            angle, force = action
            
            if step == 0:  # Only print once per episode
                print(f"  Firing: Angle={angle*180/math.pi:.1f}°, Force={force:.1f}N")
            
            observation, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            # Check results
            if done:
                if info['target_hit']:
                    print(f"  TARGET HIT! Explosion triggered!")
                    successful_hits += 1
                    
                    # Enhanced reward analysis
                    if hasattr(env, 'impact_velocity'):
                        print(f"  Impact Velocity: {env.impact_velocity:.1f} m/s")
                    if hasattr(env, 'pieces_affected_count'):
                        print(f"  Pieces Affected: {env.pieces_affected_count}/20")
                    if hasattr(env, 'total_piece_velocity'):
                        print(f"  Total Piece Velocity: {env.total_piece_velocity:.1f} m/s")
                    
                    if info['total_reward'] >= 80:
                        print(f"  PERFECT DESTRUCTION! Complete annihilation!")
                        perfect_destructions += 1
                    elif info['total_reward'] >= 50:
                        print(f"  GREAT HIT! Massive destruction!")
                    elif info['total_reward'] >= 20:
                        print(f"  GOOD HIT! Significant damage!")
                    else:
                        print(f"  WEAK HIT! Minimal damage.")
                elif info['shell_out_of_bounds']:
                    print(f"  MISS! Shell went out of bounds")
                else:
                    print(f"  MISS! Shell landed without hitting target")
                
                print(f"  Episode reward: {episode_reward:.1f}")
                break
            
            # Small delay to see the trajectory
            time.sleep(0.016)
        
        total_reward += episode_reward
        
        # Show result for 2 seconds
        start_time = time.time()
        while time.time() - start_time < 2:
            env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
            time.sleep(0.016)
    
    print("\n" + "=" * 50)
    print(f"Training Complete!")
    print(f"Total episodes: {num_episodes}")
    print(f"Successful hits: {successful_hits} ({successful_hits/num_episodes*100:.1f}%)")
    print(f"Perfect destructions: {perfect_destructions}")
    print(f"Average reward: {total_reward/num_episodes:.2f}")
    
    # Keep window open
    print("\nPress CTRL+C or close window to exit.")
    try:
        while True:
            env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
            time.sleep(0.016)
    except KeyboardInterrupt:
        pass
    
    env.close()


if __name__ == "__main__":
    # Create and run the environment
    env = TurretEnv(render_mode="human")
    
    try:
        random_agent(env, num_episodes=5)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        env.close()
    except Exception as e:
        print(f"\nError during training: {e}")
        env.close()
'''