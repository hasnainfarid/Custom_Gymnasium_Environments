import numpy as np
from typing import List, Dict, Optional, Tuple
from enum import Enum
import uuid

class WaiterState(Enum):
    IDLE = "idle"
    BUSY = "busy"

class TaskType(Enum):
    SEAT_CUSTOMER = "seat_customer"
    SERVE_FOOD = "serve_food"
    CLEAN_TABLE = "clean_table"

class TableState(Enum):
    FREE_CLEAN = "free_clean"
    FREE_DIRTY = "free_dirty"
    OCCUPIED_CLEAN = "occupied_clean"
    OCCUPIED_DIRTY = "occupied_dirty"

class CustomerState(Enum):
    WAITING = "waiting"
    SEATED = "seated"
    ORDERED = "ordered"
    EATING = "eating"
    LEFT = "left"

class Customer:
    def __init__(self, customer_id: str, arrival_time: int):
        self.customer_id = customer_id
        self.arrival_time = arrival_time
        self.state = CustomerState.WAITING
        self.wait_time = 0
        self.patience_limit = 20
        self.table_id = None
        self.order_placed_time = None
        self.food_served_time = None
        self.eating_start_time = None
        self.eating_duration = 10
        
    def update_wait_time(self):
        """Update customer wait time and check patience"""
        if self.state == CustomerState.WAITING:
            self.wait_time += 1
            return self.wait_time >= self.patience_limit
        return False
    
    def is_impatient(self) -> bool:
        """Check if customer has exceeded patience limit"""
        return self.wait_time >= self.patience_limit
    
    def seat_at_table(self, table_id: int):
        """Seat customer at specified table"""
        self.table_id = table_id
        self.state = CustomerState.SEATED
    
    def place_order(self, current_time: int):
        """Customer places order"""
        self.order_placed_time = current_time
        self.state = CustomerState.ORDERED
    
    def serve_food(self, current_time: int):
        """Food is served to customer"""
        self.food_served_time = current_time
        self.state = CustomerState.EATING
        self.eating_start_time = current_time
    
    def update_eating(self, current_time: int) -> bool:
        """Update eating progress, return True if finished eating"""
        if self.state == CustomerState.EATING:
            eating_time = current_time - self.eating_start_time
            if eating_time >= self.eating_duration:
                self.state = CustomerState.LEFT
                return True
        return False
    
    def get_state_dict(self) -> Dict:
        """Get customer state as dictionary"""
        return {
            'customer_id': self.customer_id,
            'state': self.state.value,
            'wait_time': self.wait_time,
            'table_id': self.table_id,
            'order_placed_time': self.order_placed_time,
            'food_served_time': self.food_served_time,
            'eating_start_time': self.eating_start_time
        }

class Waiter:
    def __init__(self, waiter_id: int):
        self.waiter_id = waiter_id
        self.state = WaiterState.IDLE
        self.current_task = None
        self.task_remaining_time = 0
        self.task_type = None
        self.task_target = None  # customer_id, table_id, or (customer_id, table_id)
        
    def assign_task(self, task_type: TaskType, target, duration: int):
        """Assign a task to the waiter"""
        if self.state == WaiterState.IDLE:
            self.state = WaiterState.BUSY
            self.task_type = task_type
            self.task_target = target
            self.task_remaining_time = duration
            return True
        return False
    
    def update_task(self) -> Optional[Dict]:
        """Update task progress, return task completion info if finished"""
        if self.state == WaiterState.BUSY:
            self.task_remaining_time -= 1
            if self.task_remaining_time <= 0:
                # Task completed
                completion_info = {
                    'waiter_id': self.waiter_id,
                    'task_type': self.task_type,
                    'task_target': self.task_target
                }
                self.state = WaiterState.IDLE
                self.task_type = None
                self.task_target = None
                self.task_remaining_time = 0
                return completion_info
        return None
    
    def is_idle(self) -> bool:
        """Check if waiter is available for new tasks"""
        return self.state == WaiterState.IDLE
    
    def get_state_dict(self) -> Dict:
        """Get waiter state as dictionary"""
        return {
            'waiter_id': self.waiter_id,
            'state': self.state.value,
            'task_type': self.task_type.value if self.task_type else None,
            'task_remaining_time': self.task_remaining_time,
            'task_target': self.task_target
        }

class Table:
    def __init__(self, table_id: int):
        self.table_id = table_id
        self.occupied = False
        self.dirty = False
        self.customer_id = None
        
    def get_state(self) -> TableState:
        """Get current table state"""
        if self.occupied and self.dirty:
            return TableState.OCCUPIED_DIRTY
        elif self.occupied and not self.dirty:
            return TableState.OCCUPIED_CLEAN
        elif not self.occupied and self.dirty:
            return TableState.FREE_DIRTY
        else:
            return TableState.FREE_CLEAN
    
    def seat_customer(self, customer_id: str):
        """Seat a customer at this table"""
        if not self.occupied and not self.dirty:
            self.occupied = True
            self.customer_id = customer_id
            return True
        return False
    
    def customer_leaves(self):
        """Customer leaves the table, making it dirty"""
        if self.occupied:
            self.occupied = False
            self.dirty = True
            self.customer_id = None
    
    def clean_table(self):
        """Clean the table"""
        if not self.occupied and self.dirty:
            self.dirty = False
            return True
        return False
    
    def is_available_for_seating(self) -> bool:
        """Check if table is available for seating"""
        return not self.occupied and not self.dirty
    
    def is_available_for_cleaning(self) -> bool:
        """Check if table is available for cleaning"""
        return not self.occupied and self.dirty
    
    def get_state_dict(self) -> Dict:
        """Get table state as dictionary"""
        return {
            'table_id': self.table_id,
            'occupied': self.occupied,
            'dirty': self.dirty,
            'customer_id': self.customer_id,
            'state': self.get_state().value
        }

class Order:
    def __init__(self, order_id: str, table_id: int, order_time: int):
        self.order_id = order_id
        self.table_id = table_id
        self.order_time = order_time
        self.cooking_progress = 0
        self.cooking_duration = 4
        self.is_ready = False
        
    def update_cooking(self) -> bool:
        """Update cooking progress, return True if ready"""
        if not self.is_ready:
            self.cooking_progress += 1
            if self.cooking_progress >= self.cooking_duration:
                self.is_ready = True
                return True
        return False
    
    def get_state_dict(self) -> Dict:
        """Get order state as dictionary"""
        return {
            'order_id': self.order_id,
            'table_id': self.table_id,
            'order_time': self.order_time,
            'cooking_progress': self.cooking_progress,
            'is_ready': self.is_ready
        }

class Kitchen:
    def __init__(self):
        self.orders: List[Order] = []
        self.ready_orders: List[Order] = []
        
    def add_order(self, table_id: int, order_time: int) -> str:
        """Add new order to kitchen queue"""
        order_id = str(uuid.uuid4())
        order = Order(order_id, table_id, order_time)
        self.orders.append(order)
        return order_id
    
    def update_cooking(self):
        """Update cooking progress for all orders"""
        completed_orders = []
        
        for order in self.orders:
            if order.update_cooking():
                completed_orders.append(order)
        
        # Move completed orders to ready queue
        for order in completed_orders:
            self.orders.remove(order)
            self.ready_orders.append(order)
    
    def get_ready_order_for_table(self, table_id: int) -> Optional[Order]:
        """Get ready order for specific table"""
        for order in self.ready_orders:
            if order.table_id == table_id:
                return order
        return None
    
    def remove_order(self, order: Order):
        """Remove order from ready queue"""
        if order in self.ready_orders:
            self.ready_orders.remove(order)
    
    def get_queue_length(self) -> int:
        """Get number of orders in cooking queue"""
        return len(self.orders)
    
    def get_ready_count(self) -> int:
        """Get number of ready orders"""
        return len(self.ready_orders)
    
    def get_state_dict(self) -> Dict:
        """Get kitchen state as dictionary"""
        return {
            'cooking_orders': [order.get_state_dict() for order in self.orders],
            'ready_orders': [order.get_state_dict() for order in self.ready_orders],
            'queue_length': self.get_queue_length(),
            'ready_count': self.get_ready_count()
        } 