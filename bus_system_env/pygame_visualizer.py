"""
Pygame-based visualization for the Bus System Environment.
Designed for demo purposes with configurable speed and on/off capability.
"""

import pygame
import sys
from typing import Dict, Any
from config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, STOP_RADIUS, BUS_SIZE, FONT_SIZE, COLORS,
    NUM_STOPS, NUM_BUSES
)


class BusSystemVisualizer:
    def __init__(self, demo_mode: bool = True, demo_speed: float = 1.0):
        """
        Initialize the Pygame visualizer.
        
        Args:
            demo_mode: If True, slows down rendering for human viewing
            demo_speed: Speed multiplier (1.0 = normal, 0.5 = half speed, etc.)
        """
        self.demo_mode = demo_mode
        self.demo_speed = demo_speed
        self.clock = None
        self.screen = None
        self.font = None
        
        if demo_mode:
            try:
                pygame.init()
                self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
                pygame.display.set_caption("Bus System Environment - Demo Mode")
                self.clock = pygame.time.Clock()
                self.font = pygame.font.Font(None, FONT_SIZE)
                print("Pygame visualization initialized for demo mode")
            except Exception as e:
                print(f"Pygame initialization failed: {e}")
                self.demo_mode = False
    
    def render(self, env) -> None:
        """Render the current environment state."""
        if not self.demo_mode:
            return
        
        try:
            # Clear screen
            self.screen.fill(COLORS['background'])
            
            # Draw route
            self._draw_route()
            
            # Draw stops
            self._draw_stops(env)
            
            # Draw buses
            self._draw_buses(env)
            
            # Draw info panel
            self._draw_info_panel(env)
            
            # Update display
            pygame.display.flip()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_SPACE:
                        # Pause/resume
                        self._handle_pause()
            
            # Demo mode: slow down rendering
            if self.demo_mode:
                self.clock.tick(2 * self.demo_speed)  # 2 FPS * speed multiplier
                
        except Exception as e:
            print(f"Rendering error: {e}")
    
    def _draw_route(self):
        """Draw the circular bus route."""
        center_x, center_y = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
        route_radius = 150
        
        # Draw circular route
        pygame.draw.circle(self.screen, COLORS['route'], (center_x, center_y), 
                          route_radius, 3)
        
        # Draw route arrows
        for i in range(NUM_STOPS):
            angle = (i * 90 - 90) * 3.14159 / 180  # Convert to radians
            x = center_x + int(route_radius * 0.8 * pygame.math.Vector2(1, 0).rotate(i * 90 - 90)[0])
            y = center_y + int(route_radius * 0.8 * pygame.math.Vector2(1, 0).rotate(i * 90 - 90)[1])
            
            # Draw arrow
            arrow_points = [
                (x, y),
                (x - 10, y - 5),
                (x - 10, y + 5)
            ]
            pygame.draw.polygon(self.screen, COLORS['route'], arrow_points)
    
    def _draw_stops(self, env):
        """Draw bus stops with waiting passengers."""
        center_x, center_y = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
        route_radius = 150
        
        for i in range(NUM_STOPS):
            # Calculate stop position
            angle = (i * 90 - 90) * 3.14159 / 180
            x = center_x + int(route_radius * pygame.math.Vector2(1, 0).rotate(i * 90 - 90)[0])
            y = center_y + int(route_radius * pygame.math.Vector2(1, 0).rotate(i * 90 - 90)[1])
            
            # Draw stop circle
            pygame.draw.circle(self.screen, COLORS['stop'], (x, y), STOP_RADIUS)
            
            # Draw stop number
            text = self.font.render(str(i), True, COLORS['text'])
            text_rect = text.get_rect(center=(x, y))
            self.screen.blit(text, text_rect)
            
            # Draw waiting passengers count
            waiting_count = len(env.waiting_passengers[i])
            if waiting_count > 0:
                # Draw passenger indicator
                passenger_x = x + STOP_RADIUS + 10
                passenger_y = y - 10
                pygame.draw.circle(self.screen, COLORS['waiting_passengers'], 
                                 (passenger_x, passenger_y), 8)
                
                # Draw count
                count_text = self.font.render(str(waiting_count), True, COLORS['text'])
                self.screen.blit(count_text, (passenger_x + 15, passenger_y - 8))
    
    def _draw_buses(self, env):
        """Draw buses with their current states."""
        center_x, center_y = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
        route_radius = 150
        
        for i, bus in enumerate(env.buses):
            # Debug: Print bus state
            print(f"DEBUG: Bus {i}: Stop {bus.current_stop}, Stopped: {bus.is_stopped}, Remaining: {bus.remaining_time}")
            
            # Calculate bus position
            if bus.is_stopped:
                # Bus is at a stop
                angle = (bus.current_stop * 90 - 90) * 3.14159 / 180
                x = center_x + int(route_radius * pygame.math.Vector2(1, 0).rotate(bus.current_stop * 90 - 90)[0])
                y = center_y + int(route_radius * pygame.math.Vector2(1, 0).rotate(bus.current_stop * 90 - 90)[1])
                
                # Offset bus slightly from stop
                x += 20
                y += 20
                print(f"DEBUG: Bus {i} at STOP {bus.current_stop}, pos: ({x}, {y})")
            else:
                # Bus is traveling - interpolate between stops
                # When traveling, bus is going from current_stop to next_stop
                current_stop = bus.current_stop
                next_stop = (current_stop + 1) % NUM_STOPS
                progress = 1 - (bus.remaining_time / 5.0)  # 5 is travel time
                
                # Get positions of current and next stops
                current_x = center_x + int(route_radius * pygame.math.Vector2(1, 0).rotate(current_stop * 90 - 90)[0])
                current_y = center_y + int(route_radius * pygame.math.Vector2(1, 0).rotate(current_stop * 90 - 90)[1])
                next_x = center_x + int(route_radius * pygame.math.Vector2(1, 0).rotate(next_stop * 90 - 90)[0])
                next_y = center_y + int(route_radius * pygame.math.Vector2(1, 0).rotate(next_stop * 90 - 90)[1])
                
                # Interpolate between current and next stop positions
                x = int(current_x + (next_x - current_x) * progress)
                y = int(current_y + (next_y - current_y) * progress)
                
                # Add small offset for traveling buses
                x += 10
                y += 10
                print(f"DEBUG: Bus {i} TRAVELING from {current_stop} to {next_stop}, progress: {progress:.2f}, pos: ({x}, {y})")
            
            # Draw bus
            bus_color = COLORS['bus'] if bus.is_stopped else (255, 100, 100)  # Different color when traveling
            pygame.draw.rect(self.screen, bus_color, (x - BUS_SIZE//2, y - BUS_SIZE//2, BUS_SIZE, BUS_SIZE))
            
            # Draw bus number
            bus_text = self.font.render(f"B{i}", True, COLORS['text'])
            self.screen.blit(bus_text, (x - 10, y - 8))
            
            # Draw passenger count
            passenger_count = len(bus.onboard_passengers)
            if passenger_count > 0:
                p_text = self.font.render(str(passenger_count), True, COLORS['text'])
                self.screen.blit(p_text, (x - 10, y + 10))
    
    def _draw_info_panel(self, env):
        """Draw information panel."""
        panel_x = 10
        panel_y = 10
        line_height = 20
        
        # Background panel
        panel_width = 300
        panel_height = 200
        pygame.draw.rect(self.screen, (240, 240, 240), 
                        (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, COLORS['text'], 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Info text
        info_lines = [
            f"Timestep: {env.current_timestep}",
            f"Delivered: {env.total_delivered}",
            f"Waiting: {env.total_waiting}",
            f"Onboard: {env.total_onboard}",
            "",
            "Controls:",
            "ESC - Exit",
            "SPACE - Pause"
        ]
        
        for i, line in enumerate(info_lines):
            text = self.font.render(line, True, COLORS['text'])
            self.screen.blit(text, (panel_x + 5, panel_y + 5 + i * line_height))
    
    def _handle_pause(self):
        """Handle pause/resume functionality."""
        paused = True
        while paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_SPACE:
                        paused = False
            self.clock.tick(10)
    
    def close(self):
        """Clean up Pygame resources."""
        if self.demo_mode and pygame.get_init():
            pygame.quit() 