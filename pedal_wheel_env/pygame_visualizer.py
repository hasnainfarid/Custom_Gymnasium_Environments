"""
Pygame visualizer for the Pedal Wheel Environment.
Provides real-time visualization of the unicycle simulation.
"""

import pygame
import numpy as np
try:
    from .config import *
except ImportError:
    from config import *


class PygameVisualizer:
    """Pygame-based visualizer for the pedal wheel environment."""
    
    def __init__(self):
        """Initialize the Pygame visualizer."""
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pedal Wheel Environment")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (128, 128, 128)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.BROWN = (139, 69, 19)
        self.LIGHT_GRAY = (220, 220, 220)
        
        # Camera settings
        self.camera_x = 0.0
    
    def world_to_screen(self, x, y):
        """
        Convert world coordinates to screen coordinates.
        Invert y-axis so that increasing world y goes up the screen.
        """
        screen_x = int((x - self.camera_x) * PIXELS_PER_METER)
        # Invert y: 0 world y is bottom of screen
        screen_y = SCREEN_HEIGHT - int(y * PIXELS_PER_METER)
        return screen_x, screen_y
    
    def render(self, state):
        """
        Render the current state of the environment.
        
        Args:
            state (dict): Current physics state
            
        Returns:
            np.ndarray: RGB array of the rendered frame
        """
        # Clear screen with light gray background
        self.screen.fill(self.LIGHT_GRAY)
        
        # Draw a big green rectangle at the center for debug
        pygame.draw.rect(self.screen, (0,255,0), (SCREEN_WIDTH//2-50, SCREEN_HEIGHT//2-50, 100, 100))
        
        # Update camera to follow the wheel
        target_camera_x = state['x'] - SCREEN_WIDTH / (2 * PIXELS_PER_METER)
        self.camera_x += 0.1 * (target_camera_x - self.camera_x)
        self.camera_x = max(0, self.camera_x)
        
        # Draw ground line
        ground_y = GROUND_Y * PIXELS_PER_METER
        ground_screen_y = SCREEN_HEIGHT - int(ground_y)
        print(f"[DEBUG] Ground screen y: {ground_screen_y}")
        pygame.draw.line(
            self.screen,
            self.BLACK,
            (0, ground_screen_y),
            (SCREEN_WIDTH, ground_screen_y),
            5
        )
        
        # Draw wheel
        wheel_x, wheel_y = self.world_to_screen(state['x'], state['y'])
        print(f"[DEBUG] Wheel screen x: {wheel_x}, y: {wheel_y}")
        wheel_radius_pixels = int(WHEEL_RADIUS * PIXELS_PER_METER)
        
        # Make sure wheel is on screen
        if 0 <= wheel_x <= SCREEN_WIDTH and 0 <= wheel_y <= SCREEN_HEIGHT:
            # Draw wheel outline
            pygame.draw.circle(
                self.screen,
                self.BLACK,
                (wheel_x, wheel_y),
                wheel_radius_pixels,
                3
            )
            
            # Draw wheel spokes
            for i in range(8):
                angle = state['wheel_angle'] + i * np.pi / 4
                spoke_x = wheel_x + int(wheel_radius_pixels * 0.7 * np.cos(angle))
                spoke_y = wheel_y + int(wheel_radius_pixels * 0.7 * np.sin(angle))
                pygame.draw.line(
                    self.screen,
                    self.GRAY,
                    (wheel_x, wheel_y),
                    (spoke_x, spoke_y),
                    2
                )
            
            # Draw wheel center
            pygame.draw.circle(
                self.screen,
                self.RED,
                (wheel_x, wheel_y),
                5
            )
            
            # Draw frame (simple rectangle above wheel)
            frame_height = int(0.6 * WHEEL_RADIUS * PIXELS_PER_METER)
            frame_x = wheel_x - 10
            frame_y = wheel_y - frame_height
            
            # Apply tilt to frame
            tilt = state['theta']
            frame_offset_x = int(frame_height * 0.3 * np.sin(tilt))
            
            frame_rect = pygame.Rect(
                frame_x + frame_offset_x,
                frame_y,
                20,
                frame_height
            )
            pygame.draw.rect(self.screen, self.BLUE, frame_rect)
            pygame.draw.rect(self.screen, self.BLACK, frame_rect, 2)
            
            # Draw pedals
            self.draw_pedal(wheel_x, wheel_y, state['left_pedal_angle'], self.GREEN, "L")
            self.draw_pedal(wheel_x, wheel_y, state['right_pedal_angle'], self.YELLOW, "R")
        
        # Draw UI
        self.draw_ui(state)
        
        # Draw debug border
        pygame.draw.rect(self.screen, self.RED, self.screen.get_rect(), 2)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(20)  # 20 FPS
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
        
        # Return RGB array
        return pygame.surfarray.array3d(self.screen).transpose([1, 0, 2])
    
    def draw_pedal(self, wheel_x, wheel_y, angle, color, label):
        """Draw a single pedal."""
        pedal_length_pixels = int(PEDAL_LENGTH * PIXELS_PER_METER)
        pedal_x = wheel_x + int(pedal_length_pixels * np.cos(angle))
        pedal_y = wheel_y + int(pedal_length_pixels * np.sin(angle))
        
        # Draw pedal arm
        pygame.draw.line(
            self.screen,
            color,
            (wheel_x, wheel_y),
            (pedal_x, pedal_y),
            4
        )
        
        # Draw pedal platform
        pygame.draw.circle(
            self.screen,
            color,
            (pedal_x, pedal_y),
            8
        )
        pygame.draw.circle(
            self.screen,
            self.BLACK,
            (pedal_x, pedal_y),
            8,
            2
        )
        
        # Draw pedal label
        text = self.font.render(label, True, self.BLACK)
        text_rect = text.get_rect(center=(pedal_x, pedal_y))
        self.screen.blit(text, text_rect)
    
    def draw_ui(self, state):
        """Draw UI elements."""
        # Draw stats
        stats = [
            f"Position: {state['x']:.1f}m",
            f"Velocity: {state['vx']:.1f}m/s",
            f"Tilt: {np.degrees(state['theta']):.1f}°",
            f"Energy: {state['total_energy_used']:.1f}J"
        ]
        
        for i, stat in enumerate(stats):
            text = self.font.render(stat, True, self.BLACK)
            self.screen.blit(text, (10, 10 + i * 25))
        
        # Draw tilt indicator
        self.draw_tilt_indicator(state['theta'])
    
    def draw_tilt_indicator(self, tilt):
        """Draw a visual tilt indicator."""
        indicator_x = SCREEN_WIDTH - 100
        indicator_y = 50
        indicator_size = 40
        
        # Draw background circle
        pygame.draw.circle(
            self.screen,
            self.GRAY,
            (indicator_x, indicator_y),
            indicator_size
        )
        
        # Draw tilt line
        line_length = indicator_size - 5
        end_x = indicator_x + int(line_length * np.sin(tilt))
        end_y = indicator_y + int(line_length * np.cos(tilt))
        
        color = self.RED if abs(tilt) > MAX_TILT_ANGLE else self.GREEN
        pygame.draw.line(
            self.screen,
            color,
            (indicator_x, indicator_y),
            (end_x, end_y),
            3
        )
        
        # Draw center dot
        pygame.draw.circle(
            self.screen,
            self.BLACK,
            (indicator_x, indicator_y),
            3
        )
    
    def close(self):
        """Close the visualizer and clean up Pygame."""
        pygame.quit() 