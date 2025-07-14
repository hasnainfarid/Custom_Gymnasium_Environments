import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any, Union
import random
import uuid
from collections import defaultdict

from entities import Customer, Waiter, Table, Kitchen, TaskType, CustomerState, TableState

class RestaurantEnv(gym.Env):
    """
    Restaurant Management Environment (Multi-action per step)
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.num_tables = 10
        self.num_waiters = 10
        self.max_episode_steps = 500
        self.seat_customer_duration = 2
        self.serve_food_duration = 1
        self.clean_table_duration = 3
        self.customer_patience = 20
        self.eating_duration = 10
        self.rewards = {
            'seat_customer': 2.0,
            'serve_food': 1.5,
            'clean_table': 1.0,
            'quick_service_bonus': 0.5,
            'customer_leaves_penalty': -5.0,
            'dirty_table_penalty': -1.5,
            'per_timestep_penalty': -0.1,
            'all_tables_clean_bonus': 0.5,
            'no_waiting_customers_bonus': 0.3,
            'kitchen_queue_bonus': 0.2
        }
        # Action space: Single action dict
        # {type: int, waiter_id: int, customer_id: int, table_id: int}
        self.action_space = spaces.Dict({
            'type': spaces.Discrete(4),
            'waiter_id': spaces.Discrete(self.num_waiters),
            'customer_id': spaces.Discrete(50),
            'table_id': spaces.Discrete(self.num_tables)
        })
        self.observation_space = spaces.Dict({
            'waiting_customers': spaces.Box(low=0, high=100, shape=(50, 2), dtype=np.int32),
            'waiter_status': spaces.Box(low=0, high=10, shape=(10, 3), dtype=np.int32),
            'table_occupancy': spaces.Box(low=0, high=1, shape=(10,), dtype=np.int32),
            'table_cleanliness': spaces.Box(low=0, high=1, shape=(10,), dtype=np.int32),
            'kitchen_queue': spaces.Box(low=0, high=100, shape=(50, 3), dtype=np.int32),
            'ready_orders': spaces.Box(low=0, high=100, shape=(20, 2), dtype=np.int32),
            'current_timestep': spaces.Box(low=0, high=500, shape=(1,), dtype=np.int32)
        })
        self.tables: List[Table] = []
        self.waiters: List[Waiter] = []
        self.customers: Dict[str, Customer] = {}
        self.kitchen: Kitchen = None
        self.waiting_customers: List[str] = []
        self.current_timestep = 0
        self.total_reward = 0.0
        self.episode_stats = defaultdict(int)
        self.arrival_probs = {
            (1, 150): 0.12,
            (151, 350): 0.20,
            (351, 500): 0.08
        }
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.last_action_details = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tables = [Table(i) for i in range(self.num_tables)]
        self.waiters = [Waiter(i) for i in range(self.num_waiters)]
        self.customers = {}
        self.kitchen = Kitchen()
        self.waiting_customers = []
        self.current_timestep = 0
        self.total_reward = 0.0
        self.episode_stats = defaultdict(int)
        self.episode_stats = {
            'customers_served': 0,
            'customers_left': 0,
            'tables_cleaned': 0,
            'orders_served': 0,
            'total_wait_time': 0,
            'average_wait_time': 0.0
        }
        self.last_action_details = []
        return self._get_observation(), self._get_info()

    def step(self, action: dict):
        """
        Accepts a single action dict: {type: int, waiter_id: int, customer_id: int, table_id: int}
        """
        reward = 0.0
        terminated = False
        truncated = False
        self.last_action_details = []
        waiter_id = action.get('waiter_id')
        if waiter_id is None or waiter_id >= self.num_waiters:
            self.last_action_details.append({'action': action, 'result': 'invalid_waiter'})
        else:
            waiter = self.waiters[waiter_id]
            if not waiter.is_idle():
                self.last_action_details.append({'action': action, 'result': 'waiter_not_idle'})
            else:
                action_type = action.get('type')
                if action_type == 0:  # Seat Customer
                    customer_idx = action.get('customer_id')
                    table_id = action.get('table_id')
                    if (customer_idx is not None and customer_idx < len(self.waiting_customers)
                        and table_id is not None and table_id < self.num_tables):
                        customer_id = self.waiting_customers[customer_idx]
                        table = self.tables[table_id]
                        if table.dirty:
                            reward += self.rewards['dirty_table_penalty']
                            self.last_action_details.append({'action': action, 'result': 'table_dirty'})
                        elif table.is_available_for_seating():
                            if waiter.assign_task(TaskType.SEAT_CUSTOMER, (customer_id, table_id), self.seat_customer_duration):
                                self.waiting_customers.pop(customer_idx)
                                self.last_action_details.append({'action': action, 'result': 'assigned'})
                            else:
                                self.last_action_details.append({'action': action, 'result': 'assign_failed'})
                        else:
                            self.last_action_details.append({'action': action, 'result': 'invalid_seat'})
                    else:
                        self.last_action_details.append({'action': action, 'result': 'invalid_seat'})
                elif action_type == 1:  # Serve Food
                    table_id = action.get('table_id')
                    if table_id is not None and table_id < self.num_tables:
                        table = self.tables[table_id]
                        if table.occupied and table.customer_id:
                            customer = self.customers.get(table.customer_id)
                            ready_order = self.kitchen.get_ready_order_for_table(table_id)
                            if customer and customer.state == CustomerState.ORDERED and ready_order:
                                if waiter.assign_task(TaskType.SERVE_FOOD, table_id, self.serve_food_duration):
                                    self.last_action_details.append({'action': action, 'result': 'assigned'})
                                else:
                                    self.last_action_details.append({'action': action, 'result': 'assign_failed'})
                            else:
                                self.last_action_details.append({'action': action, 'result': 'invalid_serve'})
                        else:
                            self.last_action_details.append({'action': action, 'result': 'invalid_serve'})
                    else:
                        self.last_action_details.append({'action': action, 'result': 'invalid_serve'})
                elif action_type == 2:  # Clean Table
                    table_id = action.get('table_id')
                    if table_id is not None and table_id < self.num_tables:
                        table = self.tables[table_id]
                        if table.is_available_for_cleaning():
                            if waiter.assign_task(TaskType.CLEAN_TABLE, table_id, self.clean_table_duration):
                                self.last_action_details.append({'action': action, 'result': 'assigned'})
                            else:
                                self.last_action_details.append({'action': action, 'result': 'assign_failed'})
                        else:
                            self.last_action_details.append({'action': action, 'result': 'invalid_clean'})
                    else:
                        self.last_action_details.append({'action': action, 'result': 'invalid_clean'})
                elif action_type == 3:  # Do Nothing
                    self.last_action_details.append({'action': action, 'result': 'noop'})
                else:
                    self.last_action_details.append({'action': action, 'result': 'unknown_action'})
        self._update_environment()
        efficiency_reward = self._calculate_efficiency_rewards()
        reward += efficiency_reward
        self.current_timestep += 1
        if self.current_timestep >= self.max_episode_steps:
            truncated = True
        reward += self.rewards['per_timestep_penalty']
        self.total_reward += reward
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _execute_action(self, action: int) -> float:
        """Execute the agent's action and return reward"""
        reward = 0.0
        
        if action == 0:  # Seat Customer
            reward = self._action_seat_customer()
        elif action == 1:  # Serve Food
            reward = self._action_serve_food()
        elif action == 2:  # Clean Table
            reward = self._action_clean_table()
        elif action == 3:  # Do Nothing
            pass
        
        return reward
    
    def _action_seat_customer(self) -> float:
        """Action: Seat a customer at a table"""
        # Find idle waiter
        idle_waiters = [w for w in self.waiters if w.is_idle()]
        if not idle_waiters:
            return 0.0
        
        # Find waiting customer
        if not self.waiting_customers:
            return 0.0
        
        # Find available table
        available_tables = [t for t in self.tables if t.is_available_for_seating()]
        if not available_tables:
            return 0.0
        
        # Select first available waiter, customer, and table
        waiter = idle_waiters[0]
        customer_id = self.waiting_customers[0]
        table = available_tables[0]
        
        # Check if table is dirty (penalty)
        if table.dirty:
            return self.rewards['dirty_table_penalty']
        
        # Assign task to waiter
        if waiter.assign_task(TaskType.SEAT_CUSTOMER, (customer_id, table.table_id), self.seat_customer_duration):
            # Remove customer from waiting list
            self.waiting_customers.pop(0)
            return 0.0  # Reward will be given when task completes
        
        return 0.0
    
    def _action_serve_food(self) -> float:
        """Action: Serve food to a table"""
        # Find idle waiter
        idle_waiters = [w for w in self.waiters if w.is_idle()]
        if not idle_waiters:
            return 0.0
        
        # Find table with ready order
        for table in self.tables:
            if table.occupied and table.customer_id:
                customer = self.customers.get(table.customer_id)
                if customer and customer.state == CustomerState.ORDERED:
                    # Check if order is ready
                    ready_order = self.kitchen.get_ready_order_for_table(table.table_id)
                    if ready_order:
                        waiter = idle_waiters[0]
                        if waiter.assign_task(TaskType.SERVE_FOOD, table.table_id, self.serve_food_duration):
                            return 0.0  # Reward will be given when task completes
        
        return 0.0
    
    def _action_clean_table(self) -> float:
        """Action: Clean a table"""
        # Find idle waiter
        idle_waiters = [w for w in self.waiters if w.is_idle()]
        if not idle_waiters:
            return 0.0
        
        # Find dirty table
        dirty_tables = [t for t in self.tables if t.is_available_for_cleaning()]
        if not dirty_tables:
            return 0.0
        
        # Select first available waiter and dirty table
        waiter = idle_waiters[0]
        table = dirty_tables[0]
        
        # Assign task to waiter
        if waiter.assign_task(TaskType.CLEAN_TABLE, table.table_id, self.clean_table_duration):
            return 0.0  # Reward will be given when task completes
        
        return 0.0
    
    def _update_environment(self):
        """Update all environment components"""
        # Update waiters
        self._update_waiters()
        
        # Update customers
        self._update_customers()
        
        # Update kitchen
        self.kitchen.update_cooking()
        
        # Handle customer arrivals
        self._handle_customer_arrivals()
        
        # Handle impatient customers
        self._handle_impatient_customers()
    
    def _update_waiters(self):
        """Update waiter tasks and handle completions"""
        for waiter in self.waiters:
            completion = waiter.update_task()
            if completion:
                self._handle_task_completion(completion)
    
    def _handle_task_completion(self, completion: Dict):
        """Handle completed waiter tasks"""
        task_type = completion['task_type']
        task_target = completion['task_target']
        
        if task_type == TaskType.SEAT_CUSTOMER:
            customer_id, table_id = task_target
            self._complete_seat_customer(customer_id, table_id)
        elif task_type == TaskType.SERVE_FOOD:
            table_id = task_target
            self._complete_serve_food(table_id)
        elif task_type == TaskType.CLEAN_TABLE:
            table_id = task_target
            self._complete_clean_table(table_id)
    
    def _complete_seat_customer(self, customer_id: str, table_id: int):
        """Complete seating a customer (only by waiter)"""
        customer = self.customers.get(customer_id)
        table = self.tables[table_id]
        # Only seat if table is available and customer is still waiting
        if customer and table and table.is_available_for_seating() and customer.state == CustomerState.WAITING:
            # Seat customer
            customer.seat_at_table(table_id)
            table.seat_customer(customer_id)
            # Customer places order immediately
            customer.place_order(self.current_timestep)
            order_id = self.kitchen.add_order(table_id, self.current_timestep)
            # Give reward
            self.total_reward += self.rewards['seat_customer']
            self.episode_stats['customers_served'] += 1
    
    def _complete_serve_food(self, table_id: int):
        """Complete serving food to a table"""
        table = self.tables[table_id]
        if table.occupied and table.customer_id:
            customer = self.customers.get(table.customer_id)
            ready_order = self.kitchen.get_ready_order_for_table(table_id)
            
            if customer and ready_order:
                # Serve food
                customer.serve_food(self.current_timestep)
                self.kitchen.remove_order(ready_order)
                
                # Calculate quick service bonus
                service_time = self.current_timestep - ready_order.order_time
                quick_service_bonus = self.rewards['quick_service_bonus'] if service_time <= 5 else 0.0
                
                # Give reward
                self.total_reward += self.rewards['serve_food'] + quick_service_bonus
                self.episode_stats['orders_served'] += 1
    
    def _complete_clean_table(self, table_id: int):
        """Complete cleaning a table"""
        table = self.tables[table_id]
        if table.is_available_for_cleaning():
            if table.clean_table():
                # Give reward
                self.total_reward += self.rewards['clean_table']
                self.episode_stats['tables_cleaned'] += 1
    
    def _update_customers(self):
        """Update customer states"""
        customers_to_remove = []
        
        for customer_id, customer in self.customers.items():
            # Update wait time for waiting customers
            if customer.state == CustomerState.WAITING:
                customer.update_wait_time()
            
            # Update eating progress
            if customer.state == CustomerState.EATING:
                if customer.update_eating(self.current_timestep):
                    # Customer finished eating
                    if customer.table_id is not None:
                        table = self.tables[customer.table_id]
                        table.customer_leaves()
                    customers_to_remove.append(customer_id)
        
        # Remove customers who have left
        for customer_id in customers_to_remove:
            del self.customers[customer_id]
    
    def _handle_customer_arrivals(self):
        """Handle new customer arrivals based on timestep"""
        arrival_prob = 0.0
        for (start, end), prob in self.arrival_probs.items():
            if start <= self.current_timestep <= end:
                arrival_prob = prob
                break
        
        if random.random() < arrival_prob:
            # Create new customer
            customer_id = str(uuid.uuid4())
            customer = Customer(customer_id, self.current_timestep)
            self.customers[customer_id] = customer
            self.waiting_customers.append(customer_id)
    
    def _handle_impatient_customers(self):
        """Handle customers who leave due to impatience"""
        customers_to_remove = []
        
        for customer_id in self.waiting_customers[:]:
            customer = self.customers.get(customer_id)
            if customer and customer.is_impatient():
                # Customer leaves due to impatience
                self.waiting_customers.remove(customer_id)
                customers_to_remove.append(customer_id)
                self.total_reward += self.rewards['customer_leaves_penalty']
                self.episode_stats['customers_left'] += 1
        
        # Remove impatient customers
        for customer_id in customers_to_remove:
            if customer_id in self.customers:
                del self.customers[customer_id]
    
    def _calculate_efficiency_rewards(self) -> float:
        """Calculate efficiency bonuses"""
        reward = 0.0
        
        # All tables clean bonus
        dirty_tables = sum(1 for table in self.tables if table.dirty)
        if dirty_tables == 0:
            reward += self.rewards['all_tables_clean_bonus']
        
        # No waiting customers bonus
        if len(self.waiting_customers) == 0:
            reward += self.rewards['no_waiting_customers_bonus']
        
        # Kitchen queue bonus
        if self.kitchen.get_queue_length() <= 2:
            reward += self.rewards['kitchen_queue_bonus']
        
        return reward
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation"""
        # Waiting customers
        waiting_customers_data = np.zeros((50, 2), dtype=np.int32)
        for i, customer_id in enumerate(self.waiting_customers[:50]):
            customer = self.customers.get(customer_id)
            if customer:
                waiting_customers_data[i] = [hash(customer_id) % 100, customer.wait_time]

        # Waiter status
        waiter_status = np.zeros((10, 3), dtype=np.int32)
        def encode_task_type(task_type):
            if task_type is None:
                return 0
            if task_type.value == 'seat_customer':
                return 1
            if task_type.value == 'serve_food':
                return 2
            if task_type.value == 'clean_table':
                return 3
            return 0
        for i, waiter in enumerate(self.waiters):
            waiter_status[i] = [
                0 if waiter.state.value == 'idle' else 1,
                encode_task_type(waiter.task_type),
                waiter.task_remaining_time
            ]

        # Table occupancy and cleanliness
        table_occupancy = np.array([1 if table.occupied else 0 for table in self.tables], dtype=np.int32)
        table_cleanliness = np.array([1 if table.dirty else 0 for table in self.tables], dtype=np.int32)

        # Kitchen queue
        kitchen_queue_data = np.zeros((50, 3), dtype=np.int32)
        for i, order in enumerate(self.kitchen.orders[:50]):
            kitchen_queue_data[i] = [hash(order.order_id) % 100, order.table_id, order.cooking_progress]

        # Ready orders
        ready_orders_data = np.zeros((20, 2), dtype=np.int32)
        for i, order in enumerate(self.kitchen.ready_orders[:20]):
            ready_orders_data[i] = [hash(order.order_id) % 100, order.table_id]

        return {
            'waiting_customers': waiting_customers_data,
            'waiter_status': waiter_status,
            'table_occupancy': table_occupancy,
            'table_cleanliness': table_cleanliness,
            'kitchen_queue': kitchen_queue_data,
            'ready_orders': ready_orders_data,
            'current_timestep': np.array([self.current_timestep], dtype=np.int32)
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information"""
        total_wait_time = sum(customer.wait_time for customer in self.customers.values())
        num_customers = len(self.customers)
        avg_wait_time = total_wait_time / max(num_customers, 1)
        
        return {
            'total_reward': self.total_reward,
            'current_timestep': self.current_timestep,
            'waiting_customers': len(self.waiting_customers),
            'idle_waiters': len([w for w in self.waiters if w.is_idle()]),
            'kitchen_queue_length': self.kitchen.get_queue_length(),
            'ready_orders': self.kitchen.get_ready_count(),
            'dirty_tables': sum(1 for table in self.tables if table.dirty),
            'episode_stats': dict(self.episode_stats),
            'average_wait_time': avg_wait_time
        }
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_human(self):
        """Render for human viewing using pygame visualization"""
        if not hasattr(self, 'visualization'):
            from visualization import RestaurantVisualization
            self.visualization = RestaurantVisualization(self)
        
        # Get current observation and info
        obs = self._get_observation()
        info = self._get_info()
        
        # Render using the visualization
        self.visualization.render(obs, info, self.last_action_details, self.total_reward)
        return f"Restaurant State - Timestep: {self.current_timestep}, Reward: {self.total_reward:.2f}"
    
    def _render_rgb_array(self):
        """Render as RGB array"""
        # This would be implemented with pygame visualization
        # For now, return a simple array
        return np.zeros((400, 600, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'visualization'):
            self.visualization.close()
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit() 