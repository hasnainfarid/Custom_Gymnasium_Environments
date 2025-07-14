import pygame
import numpy as np
from typing import Dict, List, Any, Optional
import sys
import os

# Add the current directory to the path so we can import from entities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from entities import TableState, WaiterState, TaskType, CustomerState

def action_to_str(action: int) -> str:
    if action == 0:
        return "Seat Customer"
    elif action == 1:
        return "Serve Food"
    elif action == 2:
        return "Clean Table"
    elif action == 3:
        return "Do Nothing"
    return str(action)

class RestaurantVisualization:
    """Pygame-based visualization for the restaurant environment"""
    
    def __init__(self, env, window_size=(1200, 800)):
        self.env = env
        self.window_size = window_size
        self.screen = None
        self.font = None
        self.small_font = None
        self.tiny_font = None
        
        # Colors
        self.colors = {
            'background': (240, 240, 240),
            'grid': (200, 200, 200),
            'text': (50, 50, 50),
            'table_free_clean': (100, 200, 100),      # Green
            'table_free_dirty': (255, 255, 100),      # Yellow
            'table_occupied_clean': (255, 100, 100),   # Red
            'table_occupied_dirty': (150, 50, 50),    # Dark Red
            'waiter_idle': (100, 150, 255),           # Blue
            'waiter_busy': (255, 150, 50),            # Orange
            'customer_waiting': (150, 100, 200),      # Purple
            'kitchen': (180, 180, 180),               # Gray
            'stats_bg': (255, 255, 255),
            'stats_border': (100, 100, 100),
            'agent_panel': (230, 230, 255),
            'agent_border': (60, 60, 180)
        }
        
        # Layout parameters
        self.table_size = 60
        self.waiter_size = 30
        self.customer_size = 25
        self.margin = 50
        self.table_spacing = 80
        
        # For showing last action/reward
        self.last_action = None
        self.last_action_params = None
        self.last_reward = None
        
        # Initialize pygame
        self._init_pygame()
    
    def _init_pygame(self):
        """Initialize pygame components"""
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Restaurant Management Environment")
        
        # Initialize fonts
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.tiny_font = pygame.font.Font(None, 18)
    
    def render(self, observation: Dict[str, np.ndarray], info: Dict[str, Any], last_actions: Optional[list]=None, last_reward: Optional[float]=None):
        self.last_actions = last_actions or []
        self.last_reward = last_reward
        self.screen.fill(self.colors['background'])
        self._draw_restaurant_layout()
        self._draw_idle_waiter_area()
        self._draw_tables()
        self._draw_waiters()
        self._draw_customers()
        self._draw_waiting_customers()
        self._draw_kitchen()
        self._draw_statistics(info)
        self._draw_agent_panel()
        self._draw_legend()
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    
    def _draw_restaurant_layout(self):
        """Draw the basic restaurant layout"""
        # Draw grid lines
        for x in range(0, self.window_size[0], 50):
            pygame.draw.line(self.screen, self.colors['grid'], (x, 0), (x, self.window_size[1]), 1)
        for y in range(0, self.window_size[1], 50):
            pygame.draw.line(self.screen, self.colors['grid'], (0, y), (self.window_size[0], y), 1)
        
        # Draw restaurant title
        title = self.font.render("Restaurant Management Environment", True, self.colors['text'])
        self.screen.blit(title, (20, 20))
    
    def _draw_tables(self):
        """Draw all tables with appropriate colors and centered numbers"""
        start_x = self.margin + 200
        start_y = self.margin + 100
        # Draw 'Tables' label above the tables area
        tables_label = self.small_font.render("Tables", True, self.colors['text'])
        label_x = start_x + (self.table_spacing * 2)  # Centered above 5 tables
        label_y = start_y - 35
        self.screen.blit(tables_label, (label_x, label_y))
        for i, table in enumerate(self.env.tables):
            # Calculate table position
            row = i // 5
            col = i % 5
            x = start_x + col * self.table_spacing
            y = start_y + row * self.table_spacing
            
            # Determine table color based on state
            state = table.get_state()
            if state == TableState.FREE_CLEAN:
                color = self.colors['table_free_clean']
            elif state == TableState.FREE_DIRTY:
                color = self.colors['table_free_dirty']
            elif state == TableState.OCCUPIED_CLEAN:
                color = self.colors['table_occupied_clean']
            else:  # OCCUPIED_DIRTY
                color = self.colors['table_occupied_dirty']
            
            # Draw table
            pygame.draw.rect(self.screen, color, (x, y, self.table_size, self.table_size))
            pygame.draw.rect(self.screen, self.colors['text'], (x, y, self.table_size, self.table_size), 2)
            
            # Draw centered table number
            table_text = self.small_font.render(str(i), True, self.colors['text'])
            text_rect = table_text.get_rect(center=(x + self.table_size // 2, y + self.table_size // 2))
            self.screen.blit(table_text, text_rect)
            
            # Draw customer indicator if occupied
            if table.occupied and table.customer_id:
                customer = self.env.customers.get(table.customer_id)
                if customer:
                    # Draw eating progress
                    if customer.state.value == 'eating':
                        eating_progress = min(1.0, (self.env.current_timestep - customer.eating_start_time) / customer.eating_duration)
                        progress_width = int(self.table_size * eating_progress)
                        pygame.draw.rect(self.screen, (0, 255, 0), (x, y + self.table_size - 5, progress_width, 5))
    
    def _draw_waiters(self):
        """Draw all waiters with their current states"""
        # Draw waiters on tables if busy with a table, else in idle area
        idle_start_x = self.margin + 30  # Left margin
        idle_start_y = self.margin + 280  # Centered vertically
        table_positions = {}
        for i, table in enumerate(self.env.tables):
            row = i // 5
            col = i % 5
            x = self.margin + 200 + col * self.table_spacing
            y = self.margin + 100 + row * self.table_spacing
            table_positions[i] = (x, y)
        for i, waiter in enumerate(self.env.waiters):
            if waiter.state == WaiterState.BUSY and waiter.task_type in [TaskType.SEAT_CUSTOMER, TaskType.SERVE_FOOD, TaskType.CLEAN_TABLE]:
                # Draw on top of relevant table
                if waiter.task_target is not None:
                    if waiter.task_type == TaskType.SEAT_CUSTOMER and isinstance(waiter.task_target, tuple):
                        _, table_id = waiter.task_target
                    else:
                        table_id = waiter.task_target if isinstance(waiter.task_target, int) else None
                    if table_id is not None and table_id in table_positions:
                        x, y = table_positions[table_id]
                        wx = x + self.table_size // 2
                        wy = y + self.table_size // 2
                        color = self.colors['waiter_busy']
                        pygame.draw.circle(self.screen, color, (wx, wy), self.waiter_size//2)
                        pygame.draw.circle(self.screen, self.colors['text'], (wx, wy), self.waiter_size//2, 2)
                        waiter_text = self.tiny_font.render(str(i), True, self.colors['text'])
                        waiter_rect = waiter_text.get_rect(center=(wx, wy))
                        self.screen.blit(waiter_text, waiter_rect)
                        continue
            # Idle waiters: draw in new idle area (center left)
            row = i // 5
            col = i % 5
            x = idle_start_x + col * 80
            y = idle_start_y + row * 80
            color = self.colors['waiter_idle']
            pygame.draw.circle(self.screen, color, (x + self.waiter_size//2, y + self.waiter_size//2), self.waiter_size//2)
            pygame.draw.circle(self.screen, self.colors['text'], (x + self.waiter_size//2, y + self.waiter_size//2), self.waiter_size//2, 2)
            waiter_text = self.tiny_font.render(str(i), True, self.colors['text'])
            waiter_rect = waiter_text.get_rect(center=(x + self.waiter_size//2, y + self.waiter_size//2))
            self.screen.blit(waiter_text, waiter_rect)
            # Draw task indicator if busy
            if waiter.state == WaiterState.BUSY:
                task_symbol = self._get_task_symbol(waiter.task_type)
                task_text = self.tiny_font.render(task_symbol, True, self.colors['text'])
                self.screen.blit(task_text, (x + 15, y + 25))
                # Draw remaining time
                time_text = self.tiny_font.render(str(waiter.task_remaining_time), True, self.colors['text'])
                self.screen.blit(time_text, (x + 15, y + 40))
    
    def _get_task_symbol(self, task_type):
        """Get symbol for task type"""
        if task_type == TaskType.SEAT_CUSTOMER:
            return "S"
        elif task_type == TaskType.SERVE_FOOD:
            return "F"
        elif task_type == TaskType.CLEAN_TABLE:
            return "C"
        return "?"
    
    def _draw_customers(self):
        """Draw waiting customers"""
        # Draw only customers who are seated at tables
        table_positions = {}
        for i, table in enumerate(self.env.tables):
            row = i // 5
            col = i % 5
            x = self.margin + 200 + col * self.table_spacing
            y = self.margin + 100 + row * self.table_spacing
            table_positions[i] = (x, y)
        for customer in self.env.customers.values():
            if customer.state in [CustomerState.SEATED, CustomerState.ORDERED, CustomerState.EATING]:
                table_id = customer.table_id
                if table_id is not None and table_id in table_positions:
                    x, y = table_positions[table_id]
                    cx = x + self.table_size // 2
                    cy = y + self.table_size // 2
                    pygame.draw.circle(self.screen, self.colors['customer_waiting'], (cx, cy), self.customer_size//2)
                    pygame.draw.circle(self.screen, self.colors['text'], (cx, cy), self.customer_size//2, 1)
    
    def _draw_waiting_customers(self):
        # Draw waiting customers queue (purple circles with wait times)
        queue_x = self.margin + 600
        queue_y = self.margin + 100
        queue_w = 300
        queue_h = 80
        # Draw area
        pygame.draw.rect(self.screen, (200, 180, 255), (queue_x - 10, queue_y - 30, queue_w, queue_h), 2)
        queue_title = self.small_font.render("Waiting Customers", True, self.colors['text'])
        self.screen.blit(queue_title, (queue_x, queue_y - 30))
        for i, customer_id in enumerate(self.env.waiting_customers[:10]):
            customer = self.env.customers.get(customer_id)
            if customer:
                x = queue_x + (i % 10) * 28 + 20
                y = queue_y + 20
                pygame.draw.circle(self.screen, self.colors['customer_waiting'], (x, y), self.customer_size//2)
                pygame.draw.circle(self.screen, self.colors['text'], (x, y), self.customer_size//2, 1)
                wait_text = self.tiny_font.render(str(customer.wait_time), True, self.colors['text'])
                self.screen.blit(wait_text, (x - 10, y + 15))

    def _draw_kitchen(self):
        """Draw kitchen status"""
        kitchen_x = self.margin + 600
        kitchen_y = self.margin + 300
        
        # Draw kitchen title
        kitchen_title = self.small_font.render("Kitchen", True, self.colors['text'])
        self.screen.blit(kitchen_title, (kitchen_x, kitchen_y - 30))
        
        # Draw kitchen area
        pygame.draw.rect(self.screen, self.colors['kitchen'], (kitchen_x, kitchen_y, 200, 150))
        pygame.draw.rect(self.screen, self.colors['text'], (kitchen_x, kitchen_y, 200, 150), 2)
        
        # Draw cooking orders
        cooking_text = self.small_font.render(f"Cooking: {self.env.kitchen.get_queue_length()}", True, self.colors['text'])
        self.screen.blit(cooking_text, (kitchen_x + 10, kitchen_y + 10))
        
        # Draw ready orders
        ready_text = self.small_font.render(f"Ready: {self.env.kitchen.get_ready_count()}", True, self.colors['text'])
        self.screen.blit(ready_text, (kitchen_x + 10, kitchen_y + 30))
        
        # Draw cooking progress bars
        for i, order in enumerate(self.env.kitchen.orders[:5]):  # Show first 5
            progress = order.cooking_progress / order.cooking_duration
            bar_width = 150
            bar_height = 8
            bar_x = kitchen_x + 10
            bar_y = kitchen_y + 60 + i * 15
            
            # Background bar
            pygame.draw.rect(self.screen, (200, 200, 200), (bar_x, bar_y, bar_width, bar_height))
            # Progress bar
            progress_width = int(bar_width * progress)
            pygame.draw.rect(self.screen, (255, 100, 100), (bar_x, bar_y, progress_width, bar_height))
    
    def _draw_statistics(self, info: Dict[str, Any]):
        """Draw real-time statistics, including current step reward if available"""
        stats_x = self.margin + 50
        stats_y = self.margin + 500
        
        # Draw statistics background
        pygame.draw.rect(self.screen, self.colors['stats_bg'], (stats_x - 10, stats_y - 10, 400, 220))
        pygame.draw.rect(self.screen, self.colors['stats_border'], (stats_x - 10, stats_y - 10, 400, 220), 2)
        
        # Draw statistics
        stats = [
            f"Timestep: {info.get('current_timestep', 0)}",
            f"Step Reward: {self.last_reward if self.last_reward is not None else 0.0:.2f}",
            f"Total Reward: {info.get('total_reward', 0):.2f}",
            f"Customers Served: {info.get('episode_stats', {}).get('customers_served', 0)}",
            f"Customers Left: {info.get('episode_stats', {}).get('customers_left', 0)}",
            f"Tables Cleaned: {info.get('episode_stats', {}).get('tables_cleaned', 0)}",
            f"Orders Served: {info.get('episode_stats', {}).get('orders_served', 0)}",
            f"Waiting Customers: {info.get('waiting_customers', 0)}",
            f"Idle Waiters: {info.get('idle_waiters', 0)}",
            f"Kitchen Queue: {info.get('kitchen_queue_length', 0)}",
            f"Ready Orders: {info.get('ready_orders', 0)}",
            f"Dirty Tables: {info.get('dirty_tables', 0)}",
            f"Avg Wait Time: {info.get('average_wait_time', 0):.1f}"
        ]
        
        for i, stat in enumerate(stats):
            stat_text = self.small_font.render(stat, True, self.colors['text'])
            self.screen.blit(stat_text, (stats_x, stats_y + i * 16))
    
    def _draw_agent_panel(self):
        # Move agent panel left and make it taller
        panel_x = self.margin + 550
        panel_y = self.margin + 220
        panel_w = 320
        panel_h = 320
        pygame.draw.rect(self.screen, self.colors['agent_panel'], (panel_x, panel_y, panel_w, panel_h))
        pygame.draw.rect(self.screen, self.colors['agent_border'], (panel_x, panel_y, panel_w, panel_h), 2)
        agent_box = pygame.Rect(panel_x + 20, panel_y + 20, 40, 40)
        pygame.draw.rect(self.screen, (80, 120, 255), agent_box)
        pygame.draw.rect(self.screen, self.colors['text'], agent_box, 2)
        agent_label = self.tiny_font.render("Agent", True, self.colors['text'])
        self.screen.blit(agent_label, (panel_x + 22, panel_y + 5))
        reward_text = self.small_font.render(f"Step Reward: {self.last_reward if self.last_reward is not None else 0.0:.2f}", True, self.colors['text'])
        self.screen.blit(reward_text, (panel_x + 70, panel_y + 20))
        actions_title = self.small_font.render("Actions this step:", True, self.colors['text'])
        self.screen.blit(actions_title, (panel_x + 20, panel_y + 70))
        y_offset = 100
        for idx, action in enumerate(self.last_actions or []):
            action_str = str(action)
            action_text = self.tiny_font.render(action_str, True, self.colors['text'])
            self.screen.blit(action_text, (panel_x + 20, panel_y + y_offset + idx * 18))
    
    def _draw_legend(self):
        """Draw color legend"""
        legend_x = self.margin + 600
        legend_y = self.margin + 500
        
        # Draw legend background
        pygame.draw.rect(self.screen, self.colors['stats_bg'], (legend_x - 10, legend_y - 10, 300, 200))
        pygame.draw.rect(self.screen, self.colors['stats_border'], (legend_x - 10, legend_y - 10, 300, 200), 2)
        
        # Draw legend title
        legend_title = self.small_font.render("Legend", True, self.colors['text'])
        self.screen.blit(legend_title, (legend_x, legend_y))
        
        # Draw table states
        table_legend = [
            ("Free + Clean", self.colors['table_free_clean']),
            ("Free + Dirty", self.colors['table_free_dirty']),
            ("Occupied + Clean", self.colors['table_occupied_clean']),
            ("Occupied + Dirty", self.colors['table_occupied_dirty'])
        ]
        
        for i, (label, color) in enumerate(table_legend):
            y = legend_y + 30 + i * 20
            pygame.draw.rect(self.screen, color, (legend_x, y, 15, 15))
            pygame.draw.rect(self.screen, self.colors['text'], (legend_x, y, 15, 15), 1)
            legend_text = self.tiny_font.render(label, True, self.colors['text'])
            self.screen.blit(legend_text, (legend_x + 20, y))
        
        # Draw waiter states
        waiter_legend = [
            ("Idle Waiter", self.colors['waiter_idle']),
            ("Busy Waiter", self.colors['waiter_busy'])
        ]
        
        for i, (label, color) in enumerate(waiter_legend):
            y = legend_y + 110 + i * 20
            pygame.draw.circle(self.screen, color, (legend_x + 7, y + 7), 7)
            pygame.draw.circle(self.screen, self.colors['text'], (legend_x + 7, y + 7), 7, 1)
            legend_text = self.tiny_font.render(label, True, self.colors['text'])
            self.screen.blit(legend_text, (legend_x + 20, y))
    
    def _draw_idle_waiter_area(self):
        # Draw only the label for Idle Waiters (no green rectangle)
        start_x = self.margin + 30
        start_y = self.margin + 280
        label = self.small_font.render("Idle Waiters", True, (0, 120, 0))
        self.screen.blit(label, (start_x + 10, start_y - 25))

    def close(self):
        """Close the visualization"""
        if self.screen:
            pygame.quit() 