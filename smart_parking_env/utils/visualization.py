"""
Pygame visualization for the Smart Parking Lot RL Environment with minute-based system.
"""

import pygame
import sys
import time
from typing import Dict, List, Tuple, Optional
# Paperspace-compatible imports
try:
    from ..core.config import ZONE_CONFIG, VISUALIZATION_CONFIG
    from ..core.parking_lot import ParkingLot
    from ..core.pricing import PricingManager
    from ..core.customer import Customer
except ImportError:
    # Fallback for Paperspace
    import sys
    import os
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from core.config import ZONE_CONFIG, VISUALIZATION_CONFIG
    from core.parking_lot import ParkingLot
    from core.pricing import PricingManager
    from core.customer import Customer
import numpy as np


class ParkingVisualization:
    """
    Pygame visualization for the parking lot environment.
    """
    
    def __init__(self, parking_lot: ParkingLot, pricing_manager: PricingManager):
        """
        Initialize the visualization.
        
        Args:
            parking_lot: Parking lot to visualize
            pricing_manager: Pricing manager for price information
        """
        self.parking_lot = parking_lot
        self.pricing_manager = pricing_manager
        self.screen = None
        self.font = None
        self.small_font = None
        self.clock = None
        self.running = False
        self.paused = False
        self.speed_multiplier = 1
        
    def initialize(self) -> bool:
        """
        Initialize pygame and create the window.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            pygame.init()
            
            # Create window
            width = VISUALIZATION_CONFIG['window_width']
            height = VISUALIZATION_CONFIG['window_height']
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Smart Parking Lot RL Environment - Minute-based")
            
            # Initialize fonts
            font_size = VISUALIZATION_CONFIG['font_size']
            self.font = pygame.font.Font(None, font_size)
            self.small_font = pygame.font.Font(None, font_size - 4)
            
            # Initialize clock
            self.clock = pygame.time.Clock()
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize visualization: {e}")
            return False
            
    def handle_events(self) -> bool:
        """
        Handle pygame events.
        
        Returns:
            True if should continue running, False if should quit
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_1:
                    self.speed_multiplier = 1
                elif event.key == pygame.K_2:
                    self.speed_multiplier = 2
                elif event.key == pygame.K_3:
                    self.speed_multiplier = 3
                elif event.key == pygame.K_4:
                    self.speed_multiplier = 5
                    
        return True
        
    def draw_parking_lot(self) -> None:
        """Draw the complete parking lot with gates and zones."""
        colors = VISUALIZATION_CONFIG['colors']
        
        # Draw background
        self.screen.fill(colors['background'])
        
        # Draw entrance gate
        self.draw_entrance_gate()
        
        # Draw exit gate
        self.draw_exit_gate()
        
        # Draw parking zones
        self.draw_parking_zones()
        
        # Draw road
        self.draw_road()
        
    def draw_entrance_gate(self) -> None:
        """Draw the entrance gate."""
        gate_x = 50
        gate_y = 20
        gate_width = 100
        gate_height = 30
        
        pygame.draw.rect(self.screen, (0, 100, 0), (gate_x, gate_y, gate_width, gate_height))
        pygame.draw.rect(self.screen, (0, 0, 0), (gate_x, gate_y, gate_width, gate_height), 2)
        
        gate_text = self.font.render("ENTRANCE", True, (255, 255, 255))
        text_rect = gate_text.get_rect(center=(gate_x + gate_width//2, gate_y + gate_height//2))
        self.screen.blit(gate_text, text_rect)
        
    def draw_exit_gate(self) -> None:
        """Draw the exit gate."""
        gate_x = 50
        gate_y = 550
        gate_width = 100
        gate_height = 30
        
        pygame.draw.rect(self.screen, (100, 0, 0), (gate_x, gate_y, gate_width, gate_height))
        pygame.draw.rect(self.screen, (0, 0, 0), (gate_x, gate_y, gate_width, gate_height), 2)
        
        gate_text = self.font.render("EXIT", True, (255, 255, 255))
        text_rect = gate_text.get_rect(center=(gate_x + gate_width//2, gate_y + gate_height//2))
        self.screen.blit(gate_text, text_rect)
        
    def draw_road(self) -> None:
        """Draw the road connecting entrance to parking zones."""
        road_color = (100, 100, 100)
        
        # Main road from entrance
        pygame.draw.rect(self.screen, road_color, (150, 35, 400, 20))
        
        # Road to exit
        pygame.draw.rect(self.screen, road_color, (150, 550, 400, 20))
        
        # Connecting roads
        pygame.draw.rect(self.screen, road_color, (530, 55, 20, 500))
        
    def draw_parking_zones(self) -> None:
        """Draw the parking zones with cars using actual parking lot data."""
        colors = VISUALIZATION_CONFIG['colors']
        
        zone_positions = {
            'A': {'x': 200, 'y': 100, 'rows': 3, 'cols': 5},  # Top zone
            'B': {'x': 200, 'y': 250, 'rows': 4, 'cols': 5},  # Middle zone (same x as A)
            'C': {'x': 200, 'y': 450, 'rows': 3, 'cols': 5}   # Bottom zone (same x as A and B)
        }
        
        for zone, config in ZONE_CONFIG.items():
            pos = zone_positions[zone]
            
            # Draw zone background
            zone_width = pos['cols'] * 45
            zone_height = pos['rows'] * 35
            zone_color = colors[f'zone_{zone.lower()}']
            
            pygame.draw.rect(self.screen, zone_color, 
                           (pos['x'] - 5, pos['y'] - 5, zone_width + 10, zone_height + 10))
            
            # Draw zone label
            zone_label = f"Zone {zone} ({config['name']})"
            label_surface = self.font.render(zone_label, True, (0, 0, 0))
            self.screen.blit(label_surface, (pos['x'], pos['y'] - 25))
            
            # Draw parking spots
            spot_size = 35
            for i in range(config['spots']):
                row = i // pos['cols']
                col = i % pos['cols']
                spot_x = pos['x'] + col * 45
                spot_y = pos['y'] + row * 35
                
                # Get actual spot data
                spot_id = self.get_spot_id_for_zone(zone, i)
                if spot_id in self.parking_lot.spots:
                    spot = self.parking_lot.spots[spot_id]
                    if spot.is_occupied:
                        self.draw_car(spot_x, spot_y, spot_size, spot.customer_id, spot.remaining_hours)
                    else:
                        pygame.draw.rect(self.screen, (200, 200, 200), (spot_x, spot_y, spot_size, spot_size))
                        pygame.draw.rect(self.screen, (100, 100, 100), (spot_x, spot_y, spot_size, spot_size), 1)
                else:
                    pygame.draw.rect(self.screen, (200, 200, 200), (spot_x, spot_y, spot_size, spot_size))
                    pygame.draw.rect(self.screen, (100, 100, 100), (spot_x, spot_y, spot_size, spot_size), 1)
                    
    def get_spot_id_for_zone(self, zone: str, index: int) -> int:
        """Get spot ID for a zone and index."""
        if zone not in self.parking_lot.zone_spots:
            return -1
        zone_spots = self.parking_lot.zone_spots[zone]
        if index < len(zone_spots):
            return zone_spots[index]
        return -1
        
    def draw_car(self, x: int, y: int, size: int, customer_id: int, remaining_hours: float) -> None:
        """Draw a car in a parking spot."""
        # Car body
        car_color = (255, 0, 0)  # Red for parked cars
        pygame.draw.rect(self.screen, car_color, (x + 2, y + 2, size - 4, size - 4))
        pygame.draw.rect(self.screen, (0, 0, 0), (x + 2, y + 2, size - 4, size - 4), 1)
        
        # Customer ID
        id_text = str(customer_id)
        id_surface = self.small_font.render(id_text, True, (255, 255, 255))
        id_rect = id_surface.get_rect(center=(x + size//2, y + size//2))
        self.screen.blit(id_surface, id_rect)
        
    def draw_small_car(self, x: int, y: int, customer_id: int, intensity: float = 1.0, is_exiting: bool = False) -> None:
        """Draw a small car for incoming/exiting customers."""
        car_color = (0, 0, 255) if not is_exiting else (100, 100, 100)  # Blue for incoming, gray for exiting
        car_size = 20
        
        pygame.draw.rect(self.screen, car_color, (x, y, car_size, car_size))
        pygame.draw.rect(self.screen, (0, 0, 0), (x, y, car_size, car_size), 1)
        
        # Customer ID
        id_text = str(customer_id)
        id_surface = self.small_font.render(id_text, True, (255, 255, 255))
        id_rect = id_surface.get_rect(center=(x + car_size//2, y + car_size//2))
        self.screen.blit(id_surface, id_rect)
        
    def draw_customers(self, customers: List[Customer]) -> None:
        """Draw arriving and departing customers as cars using real environment data."""
        colors = VISUALIZATION_CONFIG['colors']
        
        # Use only real customers from environment (no fake demo customers)
        incoming_customers = list(customers)
        
        # Draw arriving customers as cars near entrance (increase capacity)
        max_entrance_cars = 8  # Increased from 5
        for i, customer in enumerate(incoming_customers[:max_entrance_cars]):
            # Position cars in 2 rows for better capacity
            row = i // 4
            col = i % 4
            car_x = 50 + col * 35
            car_y = 80 + row * 25
            
            # Draw car (not person)
            self.draw_small_car(car_x, car_y, customer.customer_id)
            
            # Draw customer info (smaller text for more cars)
            info_text = f"C{customer.customer_id}:{customer.zone_preference}"
            text_surface = self.small_font.render(info_text, True, (0, 0, 0))
            self.screen.blit(text_surface, (car_x - 15, car_y + 20))
            
        # Show queue count if more cars than display capacity
        if len(incoming_customers) > max_entrance_cars:
            remaining = len(incoming_customers) - max_entrance_cars
            queue_text = f"+{remaining} more in queue"
            queue_surface = self.small_font.render(queue_text, True, (255, 0, 0))
            self.screen.blit(queue_surface, (50, 130))
            
        # Draw exiting cars near exit gate (increase capacity)
        exiting_cars = self.get_exiting_cars()
        max_exit_cars = 6  # Increased from 3
        for i, car_info in enumerate(exiting_cars[:max_exit_cars]):
            # Position cars in 2 rows
            row = i // 3
            col = i % 3
            car_x = 50 + col * 40
            car_y = 520 + row * 25
            
            # Draw exiting car
            self.draw_small_car(car_x, car_y, car_info['customer_id'], 0.5, is_exiting=True)
            
            # Draw exit info
            exit_text = f"Exit C{car_info['customer_id']}"
            text_surface = self.small_font.render(exit_text, True, (0, 0, 0))
            self.screen.blit(text_surface, (car_x - 15, car_y - 15))
            
        # Show exit count if more cars than display capacity
        if len(exiting_cars) > max_exit_cars:
            remaining = len(exiting_cars) - max_exit_cars
            exit_text = f"+{remaining} more exiting"
            exit_surface = self.small_font.render(exit_text, True, (100, 100, 100))
            self.screen.blit(exit_surface, (50, 570))
            
    def get_exiting_cars(self) -> List[Dict]:
        """Get cars that are about to exit (low remaining time)."""
        exiting_cars = []
        for spot in self.parking_lot.spots.values():
            if spot.is_occupied and spot.remaining_hours <= 1:  # Cars with 1 hour or less
                exiting_cars.append({
                    'customer_id': spot.customer_id,
                    'remaining_hours': spot.remaining_hours
                })
        return exiting_cars
        
    def draw_info_panel(self, current_hour: int, current_minute: int, current_reward: float = 0.0, 
                       customer_stats: Dict = None) -> None:
        """
        Draw the information panel.
        
        Args:
            current_hour: Current hour of the day
            current_minute: Current minute of the hour
            current_reward: Current step reward
            customer_stats: Customer statistics
        """
        colors = VISUALIZATION_CONFIG['colors']
        info_panel_width = VISUALIZATION_CONFIG['info_panel_width']
        window_width = VISUALIZATION_CONFIG['window_width']
        
        # Draw info panel background
        panel_x = window_width - info_panel_width
        panel_rect = pygame.Rect(panel_x, 0, info_panel_width, VISUALIZATION_CONFIG['window_height'])
        pygame.draw.rect(self.screen, colors['background'], panel_rect)
        pygame.draw.rect(self.screen, colors['text'], panel_rect, 2)
        
        # Draw information
        y_offset = 20
        line_height = 25
        
        # Title
        title = self.font.render("Smart Parking Lot", True, colors['text'])
        self.screen.blit(title, (panel_x + 10, y_offset))
        y_offset += line_height * 2
        
        # Time (minute-based)
        time_text = f"Time: {current_hour:02d}:{current_minute:02d}"
        time_surface = self.font.render(time_text, True, colors['text'])
        self.screen.blit(time_surface, (panel_x + 10, y_offset))
        y_offset += line_height
        
        # Current Reward
        reward_color = (0, 255, 0) if current_reward > 0 else (255, 0, 0)
        reward_text = f"Reward: {current_reward:.2f}"
        reward_surface = self.font.render(reward_text, True, reward_color)
        self.screen.blit(reward_surface, (panel_x + 10, y_offset))
        y_offset += line_height
        
        # Revenue
        total_revenue = self.pricing_manager.get_total_revenue()
        revenue_text = f"Revenue: ${total_revenue:.2f}"
        revenue_surface = self.font.render(revenue_text, True, (0, 255, 0))
        self.screen.blit(revenue_surface, (panel_x + 10, y_offset))
        y_offset += line_height
        
        # Queue
        queue_length = len(self.parking_lot.queue)
        queue_text = f"Queue: {queue_length}"
        queue_surface = self.font.render(queue_text, True, colors['text'])
        self.screen.blit(queue_surface, (panel_x + 10, y_offset))
        y_offset += line_height * 2
        
        # Zone information
        zone_title = self.font.render("Zone Details:", True, colors['text'])
        self.screen.blit(zone_title, (panel_x + 10, y_offset))
        y_offset += line_height
        
        for zone in ['A', 'B', 'C']:
            price = self.pricing_manager.get_current_price(zone)
            occupied, total = self.parking_lot.get_zone_occupancy(zone)
            occupancy_rate = occupied / total if total > 0 else 0.0
            
            zone_text = f"{zone}: ${price:.2f} ({occupancy_rate:.1%})"
            zone_surface = self.font.render(zone_text, True, colors['text'])
            self.screen.blit(zone_surface, (panel_x + 10, y_offset))
            y_offset += line_height
            
        y_offset += line_height
        
        # Customer statistics
        stats_title = self.font.render("Customer Stats:", True, colors['text'])
        self.screen.blit(stats_title, (panel_x + 10, y_offset))
        y_offset += line_height
        
        if customer_stats:
            total_customers = customer_stats.get('total_customers', 0)
            satisfaction_rate = customer_stats.get('satisfaction_rate', 0.0)
            avg_wait = customer_stats.get('avg_wait_time', 0.0)
            
            total_text = f"Total: {total_customers}"
            total_surface = self.small_font.render(total_text, True, colors['text'])
            self.screen.blit(total_surface, (panel_x + 10, y_offset))
            y_offset += line_height - 5
            
            satisfaction_text = f"Satisfaction: {satisfaction_rate:.1%}"
            satisfaction_surface = self.small_font.render(satisfaction_text, True, colors['text'])
            self.screen.blit(satisfaction_surface, (panel_x + 10, y_offset))
            y_offset += line_height - 5
            
            wait_text = f"Avg Wait: {avg_wait:.1f}min"
            wait_surface = self.small_font.render(wait_text, True, colors['text'])
            self.screen.blit(wait_surface, (panel_x + 10, y_offset))
            y_offset += line_height - 5
            
        y_offset += line_height
        
        # Controls
        controls_title = self.font.render("Controls:", True, colors['text'])
        self.screen.blit(controls_title, (panel_x + 10, y_offset))
        y_offset += line_height
        
        controls = [
            "SPACE: Pause/Resume",
            "1,2,3,4: Speed (1x,2x,3x,5x)",
            "ESC: Quit"
        ]
        
        for control in controls:
            control_surface = self.small_font.render(control, True, colors['text'])
            self.screen.blit(control_surface, (panel_x + 10, y_offset))
            y_offset += line_height - 5
            
        # Status
        status_text = "PAUSED" if self.paused else f"Speed: {self.speed_multiplier}x"
        status_color = (255, 0, 0) if self.paused else colors['text']
        status_surface = self.font.render(status_text, True, status_color)
        self.screen.blit(status_surface, (panel_x + 10, y_offset))
        
    def draw_pause_overlay(self) -> None:
        """Draw pause overlay when paused."""
        if not self.paused:
            return
            
        # Semi-transparent overlay
        overlay = pygame.Surface((VISUALIZATION_CONFIG['window_width'], 
                                VISUALIZATION_CONFIG['window_height']))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # Pause text
        pause_font = pygame.font.Font(None, 72)
        pause_text = pause_font.render("PAUSED", True, (255, 255, 255))
        pause_rect = pause_text.get_rect(center=(VISUALIZATION_CONFIG['window_width']//2, 
                                                VISUALIZATION_CONFIG['window_height']//2))
        self.screen.blit(pause_text, pause_rect)
        
    def run_visualization(self, current_hour: int, current_minute: int, customers: List[Customer], 
                         customer_stats: Dict = None, current_reward: float = 0.0) -> bool:
        """
        Run one frame of visualization.
        
        Args:
            current_hour: Current hour
            current_minute: Current minute  
            customers: List of customers
            customer_stats: Customer statistics
            current_reward: Current reward
            
        Returns:
            True if should continue, False if should quit
        """
        if not self.handle_events():
            return False
            
        if not self.paused:
            # Draw everything
            self.draw_parking_lot()
            self.draw_customers(customers)
            self.draw_info_panel(current_hour, current_minute, current_reward, customer_stats)
        else:
            # When paused, just draw pause overlay
            self.draw_pause_overlay()
            pygame.time.wait(100)  # Small delay when paused
            
        pygame.display.flip()
        self.clock.tick(60)
        
        return True
        
    def __enter__(self):
        """Context manager entry."""
        if self.initialize():
            return self
        else:
            raise RuntimeError("Failed to initialize visualization")
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pygame.quit()
        
    def close(self):
        """Close the visualization."""
        pygame.quit() 