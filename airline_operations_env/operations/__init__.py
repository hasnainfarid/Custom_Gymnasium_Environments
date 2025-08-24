"""
Operations module for airline data generation
Provides internal flight data, routes, and operational scenarios
"""

import random
from typing import List, Dict, Tuple
from enum import Enum


class RouteType(Enum):
    INTERNATIONAL = "international"
    DOMESTIC = "domestic"
    REGIONAL = "regional"
    COMMUTER = "commuter"


class DisruptionType(Enum):
    WEATHER = "weather"
    MECHANICAL = "mechanical"
    ATC_HOLD = "atc_hold"
    CREW_ISSUE = "crew_issue"
    SECURITY = "security"
    PASSENGER_MEDICAL = "passenger_medical"
    FUEL_SHORTAGE = "fuel_shortage"
    GATE_UNAVAILABLE = "gate_unavailable"


class SeasonalPattern(Enum):
    HOLIDAY_RUSH = "holiday_rush"
    SUMMER_VACATION = "summer_vacation"
    BUSINESS_PEAK = "business_peak"
    OFF_SEASON = "off_season"


class FlightDataGenerator:
    """Generates realistic flight data and patterns"""
    
    @staticmethod
    def generate_route_network(n_airports: int = 15) -> List[Dict]:
        """Generate hub-and-spoke route network"""
        routes = []
        
        # Hub to hub routes (high priority)
        for i in range(5):
            for j in range(i + 1, 5):
                routes.append({
                    'origin': i,
                    'destination': j,
                    'type': RouteType.DOMESTIC,
                    'frequency': random.randint(4, 8),  # Daily frequency
                    'distance': random.uniform(500, 2000),  # km
                    'base_price': random.uniform(200, 500),
                    'priority': 1.0
                })
        
        # Hub to spoke routes
        for hub in range(5):
            for spoke in range(5, n_airports):
                if random.random() < 0.7:  # Not all connections exist
                    routes.append({
                        'origin': hub,
                        'destination': spoke,
                        'type': RouteType.REGIONAL,
                        'frequency': random.randint(1, 4),
                        'distance': random.uniform(200, 1000),
                        'base_price': random.uniform(100, 300),
                        'priority': 0.7
                    })
        
        # Some direct spoke to spoke routes
        for i in range(5, n_airports):
            for j in range(i + 1, n_airports):
                if random.random() < 0.2:  # Few direct connections
                    routes.append({
                        'origin': i,
                        'destination': j,
                        'type': RouteType.COMMUTER,
                        'frequency': random.randint(1, 2),
                        'distance': random.uniform(100, 500),
                        'base_price': random.uniform(50, 150),
                        'priority': 0.4
                    })
        
        return routes
    
    @staticmethod
    def generate_passenger_demand(hour: float, route_type: RouteType, 
                                 season: SeasonalPattern) -> int:
        """Generate passenger demand based on time and season"""
        base_demand = 100
        
        # Time of day factor
        if 6 <= hour < 9 or 17 <= hour < 20:  # Peak hours
            time_factor = 1.5
        elif 9 <= hour < 17:  # Business hours
            time_factor = 1.2
        elif 20 <= hour < 23:  # Evening
            time_factor = 0.8
        else:  # Night/early morning
            time_factor = 0.3
        
        # Route type factor
        route_factors = {
            RouteType.INTERNATIONAL: 1.2,
            RouteType.DOMESTIC: 1.0,
            RouteType.REGIONAL: 0.8,
            RouteType.COMMUTER: 0.6
        }
        route_factor = route_factors.get(route_type, 1.0)
        
        # Seasonal factor
        season_factors = {
            SeasonalPattern.HOLIDAY_RUSH: 1.8,
            SeasonalPattern.SUMMER_VACATION: 1.4,
            SeasonalPattern.BUSINESS_PEAK: 1.2,
            SeasonalPattern.OFF_SEASON: 0.7
        }
        season_factor = season_factors.get(season, 1.0)
        
        demand = int(base_demand * time_factor * route_factor * season_factor)
        return max(10, min(300, demand + random.randint(-20, 20)))
    
    @staticmethod
    def generate_disruption_scenario() -> Dict:
        """Generate a random operational disruption"""
        disruption_type = random.choice(list(DisruptionType))
        
        scenarios = {
            DisruptionType.WEATHER: {
                'type': 'weather',
                'severity': random.uniform(0.3, 1.0),
                'duration_hours': random.uniform(2, 12),
                'affected_airports': random.randint(1, 5),
                'delay_minutes': random.randint(30, 240),
                'cancellation_probability': random.uniform(0.1, 0.5)
            },
            DisruptionType.MECHANICAL: {
                'type': 'mechanical',
                'severity': random.uniform(0.2, 0.8),
                'duration_hours': random.uniform(1, 6),
                'affected_aircraft': random.randint(1, 3),
                'delay_minutes': random.randint(60, 360),
                'replacement_needed': random.random() < 0.3
            },
            DisruptionType.ATC_HOLD: {
                'type': 'atc_hold',
                'severity': random.uniform(0.2, 0.6),
                'duration_hours': random.uniform(0.5, 3),
                'affected_region': random.randint(1, 3),
                'delay_minutes': random.randint(15, 90),
                'ground_stop': random.random() < 0.2
            },
            DisruptionType.CREW_ISSUE: {
                'type': 'crew_issue',
                'severity': random.uniform(0.3, 0.7),
                'duration_hours': random.uniform(1, 4),
                'crew_shortage': random.randint(1, 5),
                'delay_minutes': random.randint(30, 120),
                'overtime_required': random.random() < 0.4
            },
            DisruptionType.SECURITY: {
                'type': 'security',
                'severity': random.uniform(0.4, 0.9),
                'duration_hours': random.uniform(0.5, 2),
                'terminal_affected': random.randint(1, 3),
                'delay_minutes': random.randint(45, 180),
                'evacuation_required': random.random() < 0.1
            },
            DisruptionType.PASSENGER_MEDICAL: {
                'type': 'passenger_medical',
                'severity': random.uniform(0.2, 0.6),
                'duration_hours': random.uniform(0.5, 1.5),
                'diversion_required': random.random() < 0.3,
                'delay_minutes': random.randint(20, 60),
                'emergency_landing': random.random() < 0.1
            },
            DisruptionType.FUEL_SHORTAGE: {
                'type': 'fuel_shortage',
                'severity': random.uniform(0.3, 0.8),
                'duration_hours': random.uniform(2, 8),
                'affected_airports': random.randint(1, 3),
                'price_increase': random.uniform(1.2, 2.0),
                'tankering_required': random.random() < 0.5
            },
            DisruptionType.GATE_UNAVAILABLE: {
                'type': 'gate_unavailable',
                'severity': random.uniform(0.2, 0.5),
                'duration_hours': random.uniform(0.5, 2),
                'gates_blocked': random.randint(1, 4),
                'delay_minutes': random.randint(10, 45),
                'remote_parking': random.random() < 0.3
            }
        }
        
        return scenarios.get(disruption_type, {})
    
    @staticmethod
    def calculate_fuel_requirement(distance: float, aircraft_type: str, 
                                  weather_factor: float = 1.0) -> float:
        """Calculate fuel requirement for a flight"""
        # Base fuel consumption rates (kg per km)
        fuel_rates = {
            'widebody': 8.0,
            'narrowbody': 4.0,
            'regional': 2.0
        }
        
        base_rate = fuel_rates.get(aircraft_type, 4.0)
        
        # Add reserve fuel (30% extra)
        reserve_factor = 1.3
        
        # Weather impact
        weather_factor = max(1.0, weather_factor)
        
        # Calculate total fuel
        fuel_required = distance * base_rate * reserve_factor * weather_factor
        
        # Add taxi fuel
        taxi_fuel = 200 if aircraft_type == 'widebody' else 100
        
        return fuel_required + taxi_fuel
    
    @staticmethod
    def calculate_turnaround_time(aircraft_type: str, is_international: bool = False) -> int:
        """Calculate minimum turnaround time in minutes"""
        base_times = {
            'widebody': 90,
            'narrowbody': 45,
            'regional': 30
        }
        
        base_time = base_times.get(aircraft_type, 45)
        
        if is_international:
            base_time += 30  # Additional time for customs/immigration
        
        # Add some variability
        return base_time + random.randint(-5, 15)
    
    @staticmethod
    def generate_crew_schedule(n_crew: int = 40, max_duty_hours: float = 12) -> List[Dict]:
        """Generate crew scheduling constraints"""
        crew_schedules = []
        
        for i in range(n_crew):
            schedule = {
                'crew_id': i,
                'base_airport': random.randint(0, 4),  # Based at hubs
                'qualification': random.choice(['widebody', 'narrowbody', 'regional', 'all']),
                'duty_start': random.uniform(5, 10),  # Start time
                'max_duty': max_duty_hours,
                'rest_required': 10,  # Minimum rest hours
                'days_on': random.randint(3, 5),
                'days_off': random.randint(2, 3),
                'seniority': random.uniform(0, 1),  # Higher = more senior
                'overtime_cost': random.uniform(200, 500)  # Per hour
            }
            crew_schedules.append(schedule)
        
        return crew_schedules
    
    @staticmethod
    def calculate_delay_cost(delay_minutes: int, passengers: int, 
                            is_connection_hub: bool = False) -> float:
        """Calculate the cost of a delay"""
        # Base compensation per passenger
        if delay_minutes < 60:
            compensation = 0
        elif delay_minutes < 180:
            compensation = 50
        elif delay_minutes < 360:
            compensation = 200
        else:
            compensation = 500
        
        total_compensation = compensation * passengers
        
        # Additional costs
        crew_cost = delay_minutes * 10  # Crew overtime
        fuel_cost = delay_minutes * 5 if delay_minutes > 30 else 0  # Holding pattern fuel
        
        # Connection impact
        if is_connection_hub:
            missed_connections = int(passengers * 0.3 * min(delay_minutes / 60, 1))
            connection_cost = missed_connections * 300  # Rebooking and accommodation
        else:
            connection_cost = 0
        
        # Reputation cost (long-term impact)
        reputation_cost = delay_minutes * passengers * 0.5
        
        return total_compensation + crew_cost + fuel_cost + connection_cost + reputation_cost
    
    @staticmethod
    def generate_maintenance_schedule(aircraft_id: int) -> Dict:
        """Generate maintenance requirements for an aircraft"""
        return {
            'aircraft_id': aircraft_id,
            'last_a_check': random.randint(0, 500),  # Flight hours ago
            'last_b_check': random.randint(0, 3000),
            'last_c_check': random.randint(0, 6000),
            'last_d_check': random.randint(0, 25000),
            'a_check_due': 500,  # Every 500 flight hours
            'b_check_due': 3000,
            'c_check_due': 6000,
            'd_check_due': 25000,
            'unscheduled_probability': random.uniform(0.01, 0.05),
            'mел_items': random.randint(0, 3),  # Minimum equipment list items
            'next_scheduled': random.randint(10, 100)  # Hours until next check
        }
    
    @staticmethod
    def calculate_load_factor(passengers: int, capacity: int) -> float:
        """Calculate load factor percentage"""
        return min(1.0, passengers / capacity) if capacity > 0 else 0.0
    
    @staticmethod
    def generate_competitor_action() -> Dict:
        """Generate competitor airline actions that affect operations"""
        actions = [
            {
                'type': 'price_cut',
                'route': random.randint(0, 10),
                'discount': random.uniform(0.1, 0.3),
                'duration': random.randint(7, 30)
            },
            {
                'type': 'new_route',
                'origin': random.randint(0, 14),
                'destination': random.randint(0, 14),
                'frequency': random.randint(1, 4)
            },
            {
                'type': 'capacity_increase',
                'route': random.randint(0, 10),
                'additional_seats': random.randint(50, 200)
            },
            {
                'type': 'schedule_change',
                'route': random.randint(0, 10),
                'time_shift': random.uniform(-2, 2)
            }
        ]
        
        return random.choice(actions)
    
    @staticmethod
    def calculate_revenue(passengers: int, base_price: float, 
                         load_factor: float, season: SeasonalPattern) -> float:
        """Calculate flight revenue with dynamic pricing"""
        # Load factor pricing adjustment
        if load_factor > 0.9:
            price_multiplier = 1.3
        elif load_factor > 0.8:
            price_multiplier = 1.15
        elif load_factor > 0.7:
            price_multiplier = 1.0
        elif load_factor > 0.5:
            price_multiplier = 0.9
        else:
            price_multiplier = 0.8
        
        # Seasonal adjustment
        season_multipliers = {
            SeasonalPattern.HOLIDAY_RUSH: 1.5,
            SeasonalPattern.SUMMER_VACATION: 1.3,
            SeasonalPattern.BUSINESS_PEAK: 1.1,
            SeasonalPattern.OFF_SEASON: 0.8
        }
        season_multiplier = season_multipliers.get(season, 1.0)
        
        # Calculate total revenue
        adjusted_price = base_price * price_multiplier * season_multiplier
        revenue = passengers * adjusted_price
        
        # Add ancillary revenue (baggage, seats, food)
        ancillary = passengers * random.uniform(20, 50)
        
        return revenue + ancillary


class PerformanceMetrics:
    """Track and calculate airline performance metrics"""
    
    @staticmethod
    def calculate_otp(on_time_flights: int, total_flights: int) -> float:
        """Calculate On-Time Performance percentage"""
        return (on_time_flights / total_flights * 100) if total_flights > 0 else 0
    
    @staticmethod
    def calculate_completion_factor(completed_flights: int, scheduled_flights: int) -> float:
        """Calculate completion factor (% of flights not cancelled)"""
        return (completed_flights / scheduled_flights * 100) if scheduled_flights > 0 else 0
    
    @staticmethod
    def calculate_revenue_per_asm(revenue: float, seats: int, distance: float) -> float:
        """Calculate Revenue per Available Seat Mile"""
        asm = seats * distance * 0.621371  # Convert km to miles
        return (revenue / asm) if asm > 0 else 0
    
    @staticmethod
    def calculate_cost_per_asm(costs: float, seats: int, distance: float) -> float:
        """Calculate Cost per Available Seat Mile"""
        asm = seats * distance * 0.621371
        return (costs / asm) if asm > 0 else 0
    
    @staticmethod
    def calculate_yield(revenue: float, passengers: int, distance: float) -> float:
        """Calculate yield (revenue per passenger mile)"""
        passenger_miles = passengers * distance * 0.621371
        return (revenue / passenger_miles) if passenger_miles > 0 else 0
    
    @staticmethod
    def calculate_block_time(distance: float, aircraft_type: str) -> float:
        """Calculate block time (gate to gate) in hours"""
        speeds = {
            'widebody': 850,
            'narrowbody': 750,
            'regional': 550
        }
        
        cruise_speed = speeds.get(aircraft_type, 750)
        flight_time = distance / cruise_speed
        
        # Add taxi time
        taxi_time = 0.5  # 30 minutes total
        
        return flight_time + taxi_time
    
    @staticmethod
    def calculate_utilization(flight_hours: float, available_hours: float) -> float:
        """Calculate aircraft utilization percentage"""
        return (flight_hours / available_hours * 100) if available_hours > 0 else 0
    
    @staticmethod
    def generate_daily_report(metrics: Dict) -> str:
        """Generate a daily operations report"""
        report = f"""
        ===== DAILY OPERATIONS REPORT =====
        
        OPERATIONAL METRICS:
        - On-Time Performance: {metrics.get('otp', 0):.1f}%
        - Completion Factor: {metrics.get('completion', 0):.1f}%
        - Average Load Factor: {metrics.get('load_factor', 0):.1f}%
        - Aircraft Utilization: {metrics.get('utilization', 0):.1f}%
        
        FINANCIAL METRICS:
        - Total Revenue: ${metrics.get('revenue', 0):,.0f}
        - Total Costs: ${metrics.get('costs', 0):,.0f}
        - Profit/Loss: ${metrics.get('profit', 0):,.0f}
        - Revenue per ASM: ${metrics.get('rasm', 0):.3f}
        - Cost per ASM: ${metrics.get('casm', 0):.3f}
        - Yield: ${metrics.get('yield', 0):.3f}
        
        CUSTOMER METRICS:
        - Passengers Carried: {metrics.get('passengers', 0):,}
        - Customer Satisfaction: {metrics.get('satisfaction', 0):.1f}%
        - Complaints: {metrics.get('complaints', 0)}
        - Missed Connections: {metrics.get('missed_connections', 0)}
        
        OPERATIONAL ISSUES:
        - Delays: {metrics.get('delays', 0)}
        - Cancellations: {metrics.get('cancellations', 0)}
        - Diversions: {metrics.get('diversions', 0)}
        - Mechanical Issues: {metrics.get('mechanical', 0)}
        
        CREW METRICS:
        - Crew Timeouts: {metrics.get('timeouts', 0)}
        - Overtime Hours: {metrics.get('overtime', 0):.1f}
        - Crew Utilization: {metrics.get('crew_utilization', 0):.1f}%
        
        ====================================
        """
        return report


# Export main classes and functions
__all__ = [
    'RouteType',
    'DisruptionType', 
    'SeasonalPattern',
    'FlightDataGenerator',
    'PerformanceMetrics'
]




