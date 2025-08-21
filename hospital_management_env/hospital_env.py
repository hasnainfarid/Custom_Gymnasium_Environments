"""
Hospital Management Environment
A realistic hospital operations simulation with resource management and patient care dynamics.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import pygame
from collections import deque
import math

# Department types
class Department(Enum):
    EMERGENCY = 0
    ICU = 1
    SURGERY = 2
    GENERAL_WARD = 3
    PHARMACY = 4
    LABS = 5

# Patient severity levels
class Severity(Enum):
    MINOR = 1
    STANDARD = 2
    EMERGENCY = 3
    URGENT = 4
    CRITICAL = 5

@dataclass
class Patient:
    """Patient data structure"""
    id: int
    severity: int
    arrival_time: int
    department: Optional[int] = None
    bed_id: Optional[int] = None
    treatment_time: int = 0
    wait_time: int = 0
    disease_type: str = "general"
    age: int = 0
    insurance_delay: int = 0
    
@dataclass
class Doctor:
    """Doctor data structure"""
    id: int
    x: int
    y: int
    specialization: str
    department: int
    busy_until: int = 0
    fatigue: float = 0.0
    patients_treated: int = 0
    
@dataclass
class Nurse:
    """Nurse data structure"""
    id: int
    department: int
    fatigue: float = 0.0
    
@dataclass
class Bed:
    """Hospital bed data structure"""
    id: int
    department: int
    occupied: bool = False
    patient_id: Optional[int] = None
    patient_severity: int = 0

class HospitalManagementEnv(gym.Env):
    """
    Hospital Management Environment for reinforcement learning.
    Simulates realistic hospital operations with resource constraints.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Environment parameters
        self.width = 20
        self.height = 20
        self.render_mode = render_mode
        
        # Time tracking
        self.current_time = 0
        self.max_episode_length = 1440  # 24 hours in minutes
        
        # Department configurations
        self.department_configs = {
            Department.EMERGENCY: {"size": (4, 4), "position": (0, 0), "color": (255, 0, 0)},
            Department.ICU: {"size": (3, 3), "position": (5, 0), "color": (255, 128, 0)},
            Department.SURGERY: {"size": (4, 2), "position": (9, 0), "color": (128, 0, 255)},
            Department.GENERAL_WARD: {"size": (6, 4), "position": (0, 5), "color": (0, 255, 0)},
            Department.PHARMACY: {"size": (2, 2), "position": (7, 5), "color": (0, 255, 255)},
            Department.LABS: {"size": (3, 3), "position": (10, 5), "color": (255, 255, 0)}
        }
        
        # Staff configurations
        self.num_doctors = 15
        self.num_nurses = 25
        self.doctor_specializations = {
            "emergency": 3,
            "icu": 2,
            "general": 4,
            "surgeon": 3,
            "specialist": 2,
            "chief": 1
        }
        
        # Bed configurations
        self.bed_distribution = {
            Department.EMERGENCY: 8,
            Department.ICU: 6,
            Department.SURGERY: 4,
            Department.GENERAL_WARD: 22
        }
        self.total_beds = sum(self.bed_distribution.values())
        
        # Equipment and supplies
        self.equipment_types = [
            "ventilator", "xray", "mri", "ct_scanner", "ultrasound",
            "defibrillator", "ecg", "dialysis", "anesthesia", "surgical_robot"
        ]
        self.medicine_types = [
            "antibiotics", "painkillers", "anesthetics", "insulin", "blood_pressure",
            "anticoagulants", "steroids", "antivirals", "chemotherapy", "vaccines",
            "sedatives", "antidepressants", "antihistamines", "bronchodilators", "diuretics"
        ]
        
        # Patient arrival rates (per hour)
        self.severity_arrival_rates = {
            Severity.CRITICAL: 0.05,
            Severity.URGENT: 0.10,
            Severity.EMERGENCY: 0.20,
            Severity.STANDARD: 0.35,
            Severity.MINOR: 0.30
        }
        
        # Treatment times (in minutes)
        self.treatment_times = {
            Severity.CRITICAL: 120,
            Severity.URGENT: 60,
            Severity.EMERGENCY: 45,
            Severity.STANDARD: 30,
            Severity.MINOR: 15
        }
        
        # Define observation space (95 elements as specified)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(295,),  # Expanded for complete state representation
            dtype=np.float32
        )
        
        # Define action space (35 discrete actions)
        self.action_space = spaces.Discrete(35)
        
        # Initialize pygame if rendering
        self.screen = None
        self.clock = None
        self.font = None
        if render_mode == "human":
            pygame.init()
            pygame.font.init()
            
        # Episode tracking
        self.deaths = 0
        self.patients_treated = 0
        self.total_wait_time = 0
        self.critical_incidents = []
        
        # Initialize environment state
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.current_time = 0
        self.deaths = 0
        self.patients_treated = 0
        self.total_wait_time = 0
        self.critical_incidents = []
        
        # Initialize doctors
        self.doctors = []
        doc_id = 0
        for spec, count in self.doctor_specializations.items():
            for _ in range(count):
                dept = random.choice(list(Department))
                config = self.department_configs[dept]
                self.doctors.append(Doctor(
                    id=doc_id,
                    x=config["position"][0] + random.randint(0, config["size"][0]-1),
                    y=config["position"][1] + random.randint(0, config["size"][1]-1),
                    specialization=spec,
                    department=dept.value,
                    fatigue=random.uniform(0, 30)
                ))
                doc_id += 1
        
        # Initialize nurses
        self.nurses = []
        nurses_per_dept = self.num_nurses // len(Department)
        for i in range(self.num_nurses):
            dept = i // nurses_per_dept if i // nurses_per_dept < len(Department) else len(Department) - 1
            self.nurses.append(Nurse(
                id=i,
                department=dept,
                fatigue=random.uniform(0, 30)
            ))
        
        # Initialize beds
        self.beds = []
        bed_id = 0
        for dept, count in self.bed_distribution.items():
            for _ in range(count):
                self.beds.append(Bed(
                    id=bed_id,
                    department=dept.value
                ))
                bed_id += 1
        
        # Initialize equipment
        self.equipment_status = {eq: random.uniform(0.7, 1.0) for eq in self.equipment_types}
        self.equipment_in_use = {eq: False for eq in self.equipment_types}
        
        # Initialize medicine inventory
        self.medicine_inventory = {med: random.randint(50, 100) for med in self.medicine_types}
        
        # Initialize patient queues
        self.patient_queues = {dept: deque() for dept in Department}
        self.active_patients = {}
        self.next_patient_id = 0
        
        # Department metrics
        self.dept_wait_times = {dept: 0.0 for dept in Department}
        self.dept_utilization = {dept: 0.0 for dept in Department}
        
        # Special events
        self.outbreak_active = False
        self.outbreak_type = None
        self.mass_casualty_event = False
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Generate the current observation vector"""
        obs = []
        
        # Doctor locations and availability (45 elements)
        for doctor in self.doctors[:15]:  # Ensure we have exactly 15
            obs.extend([
                doctor.x / self.width,
                doctor.y / self.height,
                1.0 if doctor.busy_until > self.current_time else 0.0
            ])
        
        # Nurse assignments per department (6 elements)
        nurse_counts = [0] * 6
        for nurse in self.nurses:
            if nurse.department < 6:
                nurse_counts[nurse.department] += 1
        obs.extend([count / 10.0 for count in nurse_counts])
        
        # Bed occupancy (80 elements: 40 beds * 2)
        for bed in self.beds[:40]:  # Ensure we have exactly 40
            obs.extend([
                1.0 if bed.occupied else 0.0,
                bed.patient_severity / 5.0
            ])
        
        # Patient queues per department (30 elements: 6 depts * 5 severity levels)
        for dept in Department:
            severity_counts = [0] * 5
            for patient in self.patient_queues[dept]:
                if patient.severity <= 5:
                    severity_counts[patient.severity - 1] += 1
            obs.extend([min(count / 10.0, 1.0) for count in severity_counts])
        
        # Equipment availability (10 elements)
        for eq in self.equipment_types[:10]:
            obs.append(self.equipment_status[eq])
        
        # Medicine inventory (15 elements)
        for med in self.medicine_types[:15]:
            obs.append(self.medicine_inventory[med] / 100.0)
        
        # Department utilization (6 elements)
        for dept in Department:
            obs.append(self.dept_utilization[dept])
        
        # Average waiting times (6 elements)
        for dept in Department:
            obs.append(min(self.dept_wait_times[dept] / 60.0, 1.0))
        
        # Staff fatigue levels (40 elements: 15 doctors + 25 nurses)
        for doctor in self.doctors[:15]:
            obs.append(doctor.fatigue / 100.0)
        for nurse in self.nurses[:25]:
            obs.append(nurse.fatigue / 100.0)
        
        # Additional state information
        obs.extend([
            self.deaths / 10.0,
            self.patients_treated / 100.0,
            self.current_time / self.max_episode_length,
            1.0 if self.outbreak_active else 0.0,
            1.0 if self.mass_casualty_event else 0.0
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        """Execute one time step within the environment"""
        self.current_time += 1
        reward = 0
        
        # Process action
        action_reward = self._process_action(action)
        reward += action_reward
        
        # Generate new patients
        self._generate_patients()
        
        # Process patient treatment
        treatment_reward = self._process_treatments()
        reward += treatment_reward
        
        # Update patient queues and waiting times
        queue_reward = self._update_queues()
        reward += queue_reward
        
        # Update staff fatigue
        self._update_staff_fatigue()
        
        # Update equipment and supplies
        self._update_equipment()
        
        # Check for special events
        self._check_special_events()
        
        # Calculate department metrics
        self._update_department_metrics()
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_time >= self.max_episode_length
        
        # Get observation
        obs = self._get_observation()
        
        info = {
            "deaths": self.deaths,
            "patients_treated": self.patients_treated,
            "avg_wait_time": self.total_wait_time / max(self.patients_treated, 1),
            "time": self.current_time
        }
        
        return obs, reward, terminated, truncated, info
    
    def _process_action(self, action):
        """Process the selected action and return immediate reward"""
        reward = 0
        
        if action <= 5:
            # Assign additional nurse to department
            dept = action
            available_nurses = [n for n in self.nurses if n.fatigue < 80]
            if available_nurses:
                nurse = random.choice(available_nurses)
                nurse.department = dept
                reward += 10
        
        elif action <= 11:
            # Reassign doctor between departments
            if len(self.doctors) > 0:
                doctor = random.choice(self.doctors)
                new_dept = (action - 6) % len(Department)
                doctor.department = new_dept
                config = self.department_configs[Department(new_dept)]
                doctor.x = config["position"][0] + random.randint(0, config["size"][0]-1)
                doctor.y = config["position"][1] + random.randint(0, config["size"][1]-1)
                reward += 5
        
        elif action <= 17:
            # Adjust department priority levels
            dept_idx = action - 12
            if dept_idx < len(Department):
                dept = Department(dept_idx)
                # Prioritize critical patients in this department
                critical_patients = [p for p in self.patient_queues[dept] if p.severity >= 4]
                if critical_patients:
                    reward += 20 * len(critical_patients)
        
        elif action <= 23:
            # Schedule equipment maintenance/transfer
            eq_idx = action - 18
            if eq_idx < len(self.equipment_types):
                eq = self.equipment_types[eq_idx]
                if not self.equipment_in_use[eq]:
                    self.equipment_status[eq] = min(1.0, self.equipment_status[eq] + 0.2)
                    reward += 15
        
        elif action <= 29:
            # Order emergency supplies
            dept_idx = action - 24
            if dept_idx < len(self.medicine_types):
                med = self.medicine_types[dept_idx % len(self.medicine_types)]
                self.medicine_inventory[med] = min(100, self.medicine_inventory[med] + 20)
                reward -= 5  # Cost of supplies
        
        elif action == 30:
            # Call additional staff (overtime)
            for doctor in self.doctors:
                doctor.fatigue = max(0, doctor.fatigue - 10)
            for nurse in self.nurses:
                nurse.fatigue = max(0, nurse.fatigue - 10)
            reward -= 50  # Overtime costs
        
        elif action == 31:
            # Discharge stable patients early
            discharged = 0
            for bed in self.beds:
                if bed.occupied and bed.patient_severity <= 2:
                    bed.occupied = False
                    bed.patient_id = None
                    bed.patient_severity = 0
                    discharged += 1
                    if discharged >= 3:
                        break
            reward += discharged * 30
        
        elif action == 32:
            # Transfer patients to other hospitals
            for dept in Department:
                if len(self.patient_queues[dept]) > 10:
                    for _ in range(min(3, len(self.patient_queues[dept]))):
                        self.patient_queues[dept].popleft()
                        reward -= 200  # Transfer cost
        
        elif action == 33:
            # Activate emergency protocol
            self.mass_casualty_event = True
            # All staff on high alert
            for doctor in self.doctors:
                doctor.busy_until = max(0, doctor.busy_until - 10)
            reward -= 100  # Protocol activation cost
        
        elif action == 34:
            # Normal operation mode
            self.mass_casualty_event = False
            reward += 5
        
        return reward
    
    def _generate_patients(self):
        """Generate new patient arrivals based on arrival rates"""
        base_rate = random.randint(20, 35) / 60.0  # 20-35 patients per hour
        
        # Adjust for special events
        if self.outbreak_active:
            base_rate *= 1.5
        if self.mass_casualty_event:
            base_rate *= 2.0
        
        # Generate patients for this timestep
        if random.random() < base_rate:
            severity_roll = random.random()
            cumulative = 0
            severity = Severity.MINOR
            
            for sev, rate in self.severity_arrival_rates.items():
                cumulative += rate
                if severity_roll < cumulative:
                    severity = sev
                    break
            
            # Create new patient
            patient = Patient(
                id=self.next_patient_id,
                severity=severity.value,
                arrival_time=self.current_time,
                treatment_time=self.treatment_times[severity],
                age=random.randint(1, 90),
                disease_type=self._generate_disease_type()
            )
            
            # Add insurance delay for some patients
            if random.random() < 0.2:
                patient.insurance_delay = random.randint(10, 30)
            
            self.next_patient_id += 1
            
            # Assign to appropriate department queue
            if severity == Severity.CRITICAL:
                self.patient_queues[Department.ICU].append(patient)
            elif severity in [Severity.URGENT, Severity.EMERGENCY]:
                self.patient_queues[Department.EMERGENCY].append(patient)
            else:
                self.patient_queues[Department.GENERAL_WARD].append(patient)
    
    def _generate_disease_type(self):
        """Generate a disease type for a patient"""
        diseases = [
            "cardiac", "respiratory", "trauma", "neurological", "infection",
            "diabetes", "hypertension", "fracture", "poisoning", "burn",
            "stroke", "pneumonia", "appendicitis", "kidney_failure", "cancer"
        ]
        
        if self.outbreak_active and self.outbreak_type:
            # Higher chance of outbreak disease
            if random.random() < 0.6:
                return self.outbreak_type
        
        return random.choice(diseases)
    
    def _process_treatments(self):
        """Process ongoing patient treatments"""
        reward = 0
        
        # Process patients in beds
        for bed in self.beds:
            if bed.occupied and bed.patient_id in self.active_patients:
                patient = self.active_patients[bed.patient_id]
                
                # Check if treatment is complete
                if self.current_time - patient.arrival_time >= patient.treatment_time:
                    # Treatment successful
                    bed.occupied = False
                    bed.patient_id = None
                    bed.patient_severity = 0
                    
                    # Calculate reward based on severity
                    if patient.severity == 5:
                        reward += 1000  # Life saved
                    elif patient.severity == 4:
                        reward += 500
                    elif patient.severity == 3:
                        reward += 200
                    else:
                        reward += 100
                    
                    self.patients_treated += 1
                    del self.active_patients[patient.id]
                    
                    # Update doctor stats
                    for doctor in self.doctors:
                        if doctor.department == patient.department:
                            doctor.patients_treated += 1
                            break
        
        # Assign patients from queues to beds
        for dept in Department:
            if dept not in self.bed_distribution:
                continue
                
            # Get available beds in department
            available_beds = [b for b in self.beds 
                            if not b.occupied and b.department == dept.value]
            
            # Get available doctors
            available_doctors = [d for d in self.doctors 
                               if d.department == dept.value and d.busy_until <= self.current_time]
            
            # Process queue
            while available_beds and available_doctors and self.patient_queues[dept]:
                patient = self.patient_queues[dept].popleft()
                bed = available_beds.pop(0)
                doctor = available_doctors.pop(0)
                
                # Check for insurance delay
                if patient.insurance_delay > 0:
                    patient.insurance_delay -= 1
                    self.patient_queues[dept].appendleft(patient)
                    continue
                
                # Assign patient to bed
                bed.occupied = True
                bed.patient_id = patient.id
                bed.patient_severity = patient.severity
                patient.department = dept.value
                patient.bed_id = bed.id
                patient.wait_time = self.current_time - patient.arrival_time
                
                # Update doctor
                doctor.busy_until = self.current_time + patient.treatment_time // 2
                doctor.fatigue = min(100, doctor.fatigue + patient.severity * 2)
                
                # Add to active patients
                self.active_patients[patient.id] = patient
                
                # Update wait time tracking
                self.total_wait_time += patient.wait_time
        
        return reward
    
    def _update_queues(self):
        """Update patient queues and check for critical delays"""
        reward = 0
        
        for dept, queue in self.patient_queues.items():
            total_wait = 0
            patients_to_remove = []
            
            for patient in queue:
                wait_time = self.current_time - patient.arrival_time
                total_wait += wait_time
                
                # Check for critical delays
                if patient.severity == 5 and wait_time > 60:
                    # Critical patient waiting too long
                    if random.random() < 0.1:  # 10% chance of death per timestep after 60 min
                        self.deaths += 1
                        patients_to_remove.append(patient)
                        reward -= 2000
                        self.critical_incidents.append({
                            "time": self.current_time,
                            "type": "death",
                            "patient_id": patient.id,
                            "reason": "critical_delay"
                        })
                    else:
                        reward -= 500
                elif patient.severity == 4 and wait_time > 90:
                    reward -= 100
                elif patient.severity == 3 and wait_time > 30:
                    reward -= 50
            
            # Remove deceased patients
            for patient in patients_to_remove:
                queue.remove(patient)
            
            # Update department wait time
            if len(queue) > 0:
                self.dept_wait_times[dept] = total_wait / len(queue)
            else:
                self.dept_wait_times[dept] = 0
        
        return reward
    
    def _update_staff_fatigue(self):
        """Update staff fatigue levels"""
        # Natural fatigue increase
        for doctor in self.doctors:
            if doctor.busy_until > self.current_time:
                doctor.fatigue = min(100, doctor.fatigue + 0.5)
            else:
                doctor.fatigue = max(0, doctor.fatigue - 0.2)
        
        for nurse in self.nurses:
            # Get department load
            dept = Department(nurse.department)
            queue_size = len(self.patient_queues[dept])
            if queue_size > 5:
                nurse.fatigue = min(100, nurse.fatigue + 0.3)
            else:
                nurse.fatigue = max(0, nurse.fatigue - 0.1)
    
    def _update_equipment(self):
        """Update equipment status and medicine inventory"""
        # Equipment degradation
        for eq in self.equipment_types:
            if self.equipment_in_use[eq]:
                self.equipment_status[eq] = max(0, self.equipment_status[eq] - 0.01)
                if random.random() < 0.001:  # Small chance of breakdown
                    self.equipment_status[eq] = 0
            
            # Random equipment usage
            if random.random() < 0.1:
                self.equipment_in_use[eq] = not self.equipment_in_use[eq]
        
        # Medicine consumption
        for med in self.medicine_types:
            if self.patients_treated > 0:
                consumption = random.randint(0, 2)
                self.medicine_inventory[med] = max(0, self.medicine_inventory[med] - consumption)
    
    def _check_special_events(self):
        """Check and trigger special events"""
        # Outbreak event
        if not self.outbreak_active and random.random() < 0.001:
            self.outbreak_active = True
            self.outbreak_type = random.choice(["flu", "covid", "gastro", "respiratory"])
        elif self.outbreak_active and random.random() < 0.01:
            self.outbreak_active = False
            self.outbreak_type = None
        
        # Mass casualty event
        if not self.mass_casualty_event and random.random() < 0.0005:
            self.mass_casualty_event = True
            # Generate multiple critical patients
            for _ in range(random.randint(5, 10)):
                patient = Patient(
                    id=self.next_patient_id,
                    severity=random.randint(3, 5),
                    arrival_time=self.current_time,
                    treatment_time=random.randint(45, 120),
                    disease_type="trauma"
                )
                self.next_patient_id += 1
                self.patient_queues[Department.EMERGENCY].append(patient)
    
    def _update_department_metrics(self):
        """Calculate department utilization metrics"""
        for dept in Department:
            if dept not in self.bed_distribution:
                self.dept_utilization[dept] = 0
                continue
            
            total_beds = self.bed_distribution[dept]
            occupied_beds = sum(1 for b in self.beds 
                              if b.department == dept.value and b.occupied)
            
            self.dept_utilization[dept] = occupied_beds / max(total_beds, 1)
    
    def _check_termination(self):
        """Check if episode should terminate"""
        # Check for 3 preventable deaths
        if self.deaths >= 3:
            return True
        
        # Check for overcapacity
        total_capacity = sum(self.dept_utilization.values()) / len(self.dept_utilization)
        if total_capacity > 1.5:
            return True
        
        # Check for dangerous staff fatigue
        high_fatigue_doctors = sum(1 for d in self.doctors if d.fatigue > 95)
        if high_fatigue_doctors == len(self.doctors):
            return True
        
        return False
    
    def render(self):
        """Render the environment using pygame"""
        if self.render_mode is None:
            return None
        
        if self.screen is None:
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((1000, 800))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)
            pygame.display.set_caption("Hospital Management Environment")
        
        # Clear screen
        self.screen.fill((240, 240, 240))
        
        # Draw hospital layout
        self._draw_hospital_layout()
        
        # Draw staff
        self._draw_staff()
        
        # Draw patients
        self._draw_patients()
        
        # Draw metrics dashboard
        self._draw_dashboard()
        
        # Draw alerts
        self._draw_alerts()
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        
        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), 
                axes=(1, 0, 2)
            )
        
        return None
    
    def _draw_hospital_layout(self):
        """Draw the hospital floor plan"""
        cell_size = 30
        offset_x, offset_y = 50, 50
        
        # Draw grid
        for x in range(self.width + 1):
            pygame.draw.line(self.screen, (200, 200, 200),
                           (offset_x + x * cell_size, offset_y),
                           (offset_x + x * cell_size, offset_y + self.height * cell_size))
        for y in range(self.height + 1):
            pygame.draw.line(self.screen, (200, 200, 200),
                           (offset_x, offset_y + y * cell_size),
                           (offset_x + self.width * cell_size, offset_y + y * cell_size))
        
        # Draw departments
        for dept, config in self.department_configs.items():
            x, y = config["position"]
            w, h = config["size"]
            color = config["color"]
            
            # Draw department area
            rect = pygame.Rect(
                offset_x + x * cell_size,
                offset_y + y * cell_size,
                w * cell_size,
                h * cell_size
            )
            pygame.draw.rect(self.screen, color, rect, 3)
            
            # Draw department name
            text = self.small_font.render(dept.name, True, (0, 0, 0))
            text_rect = text.get_rect(center=(rect.centerx, rect.top - 10))
            self.screen.blit(text, text_rect)
            
            # Draw utilization bar
            if dept in self.dept_utilization:
                util = self.dept_utilization[dept]
                bar_rect = pygame.Rect(rect.left, rect.bottom + 5, 
                                      int(rect.width * util), 5)
                bar_color = (255, 0, 0) if util > 0.9 else (255, 255, 0) if util > 0.7 else (0, 255, 0)
                pygame.draw.rect(self.screen, bar_color, bar_rect)
    
    def _draw_staff(self):
        """Draw doctors and nurses on the hospital layout"""
        cell_size = 30
        offset_x, offset_y = 50, 50
        
        # Draw doctors (blue dots)
        for doctor in self.doctors:
            x = offset_x + doctor.x * cell_size + cell_size // 2
            y = offset_y + doctor.y * cell_size + cell_size // 2
            
            # Color based on status
            if doctor.busy_until > self.current_time:
                color = (0, 0, 128)  # Dark blue when busy
            elif doctor.fatigue > 80:
                color = (128, 0, 128)  # Purple when tired
            else:
                color = (0, 0, 255)  # Blue when available
            
            pygame.draw.circle(self.screen, color, (x, y), 8)
            
            # Draw specialization icon
            spec_text = doctor.specialization[0].upper()
            text = self.small_font.render(spec_text, True, (255, 255, 255))
            text_rect = text.get_rect(center=(x, y))
            self.screen.blit(text, text_rect)
        
        # Draw nurse indicators in departments
        nurse_counts = {}
        for nurse in self.nurses:
            if nurse.department not in nurse_counts:
                nurse_counts[nurse.department] = 0
            nurse_counts[nurse.department] += 1
        
        for dept_value, count in nurse_counts.items():
            if dept_value < len(Department):
                dept = Department(dept_value)
                config = self.department_configs[dept]
                x = offset_x + config["position"][0] * cell_size + 5
                y = offset_y + config["position"][1] * cell_size + 5
                
                # Draw nurse indicator
                pygame.draw.circle(self.screen, (0, 255, 0), (x, y), 5)
                text = self.small_font.render(str(count), True, (0, 128, 0))
                self.screen.blit(text, (x + 10, y - 5))
    
    def _draw_patients(self):
        """Draw patients in queues and beds"""
        cell_size = 30
        offset_x, offset_y = 50, 50
        
        # Draw queue sizes for each department
        queue_y = offset_y + self.height * cell_size + 50
        queue_x = offset_x
        
        for dept in Department:
            queue = self.patient_queues[dept]
            
            # Count by severity
            severity_counts = {i: 0 for i in range(1, 6)}
            for patient in queue:
                severity_counts[patient.severity] += 1
            
            # Draw queue visualization
            dept_text = self.font.render(f"{dept.name}:", True, (0, 0, 0))
            self.screen.blit(dept_text, (queue_x, queue_y))
            
            bar_x = queue_x
            bar_y = queue_y + 25
            
            severity_colors = {
                1: (0, 255, 0),    # Green - Minor
                2: (128, 255, 0),  # Yellow-green - Standard
                3: (255, 255, 0),  # Yellow - Emergency
                4: (255, 128, 0),  # Orange - Urgent
                5: (255, 0, 0)     # Red - Critical
            }
            
            for severity in range(1, 6):
                count = severity_counts[severity]
                if count > 0:
                    bar_width = count * 5
                    pygame.draw.rect(self.screen, severity_colors[severity],
                                   (bar_x, bar_y, bar_width, 15))
                    bar_x += bar_width + 2
            
            queue_x += 160
            if queue_x > offset_x + 800:
                queue_x = offset_x
                queue_y += 50
        
        # Draw bed occupancy
        bed_display_y = 650
        bed_display_x = offset_x
        
        beds_text = self.font.render("Bed Occupancy:", True, (0, 0, 0))
        self.screen.blit(beds_text, (bed_display_x, bed_display_y))
        
        for i, bed in enumerate(self.beds):
            if i % 20 == 0 and i > 0:
                bed_display_y += 20
                bed_display_x = offset_x
            
            color = (255, 0, 0) if bed.occupied else (0, 255, 0)
            pygame.draw.rect(self.screen, color,
                           (bed_display_x + 150 + i % 20 * 15, bed_display_y, 12, 12))
    
    def _draw_dashboard(self):
        """Draw the metrics dashboard"""
        dashboard_x = 700
        dashboard_y = 50
        
        # Title
        title = self.font.render("Hospital Metrics", True, (0, 0, 0))
        self.screen.blit(title, (dashboard_x, dashboard_y))
        
        # Metrics
        metrics = [
            f"Time: {self.current_time} / {self.max_episode_length}",
            f"Deaths: {self.deaths}",
            f"Treated: {self.patients_treated}",
            f"Avg Wait: {self.total_wait_time / max(self.patients_treated, 1):.1f} min",
            "",
            "Equipment Status:",
        ]
        
        y_offset = dashboard_y + 30
        for metric in metrics:
            text = self.small_font.render(metric, True, (0, 0, 0))
            self.screen.blit(text, (dashboard_x, y_offset))
            y_offset += 20
        
        # Equipment bars
        for eq in self.equipment_types[:5]:  # Show first 5
            status = self.equipment_status[eq]
            text = self.small_font.render(f"{eq[:8]}:", True, (0, 0, 0))
            self.screen.blit(text, (dashboard_x, y_offset))
            
            # Status bar
            bar_color = (0, 255, 0) if status > 0.7 else (255, 255, 0) if status > 0.3 else (255, 0, 0)
            pygame.draw.rect(self.screen, bar_color,
                           (dashboard_x + 80, y_offset, int(status * 100), 15))
            pygame.draw.rect(self.screen, (0, 0, 0),
                           (dashboard_x + 80, y_offset, 100, 15), 1)
            
            y_offset += 20
        
        # Medicine inventory
        y_offset += 10
        inv_text = self.small_font.render("Medicine Inventory:", True, (0, 0, 0))
        self.screen.blit(inv_text, (dashboard_x, y_offset))
        y_offset += 20
        
        for i, (med, count) in enumerate(list(self.medicine_inventory.items())[:5]):
            text = self.small_font.render(f"{med[:8]}: {count}", True, (0, 0, 0))
            self.screen.blit(text, (dashboard_x, y_offset))
            y_offset += 20
    
    def _draw_alerts(self):
        """Draw critical alerts and warnings"""
        alert_x = 700
        alert_y = 500
        
        alerts = []
        
        # Check for critical conditions
        critical_patients = sum(1 for dept in self.patient_queues.values() 
                               for p in dept if p.severity == 5)
        if critical_patients > 0:
            alerts.append(f"⚠️ {critical_patients} CRITICAL PATIENTS WAITING")
        
        if self.outbreak_active:
            alerts.append(f"🦠 OUTBREAK: {self.outbreak_type}")
        
        if self.mass_casualty_event:
            alerts.append("🚨 MASS CASUALTY EVENT")
        
        high_fatigue = sum(1 for d in self.doctors if d.fatigue > 80)
        if high_fatigue > 5:
            alerts.append(f"😴 {high_fatigue} DOCTORS EXHAUSTED")
        
        low_meds = [m for m, c in self.medicine_inventory.items() if c < 10]
        if low_meds:
            alerts.append(f"💊 LOW MEDICINE: {', '.join(low_meds[:3])}")
        
        # Draw alerts with flashing effect
        if alerts:
            # Flashing background
            if self.current_time % 20 < 10:
                pygame.draw.rect(self.screen, (255, 200, 200),
                               (alert_x - 10, alert_y - 10, 280, len(alerts) * 25 + 20))
            
            for i, alert in enumerate(alerts):
                color = (255, 0, 0) if "CRITICAL" in alert or "MASS" in alert else (255, 128, 0)
                text = self.small_font.render(alert, True, color)
                self.screen.blit(text, (alert_x, alert_y + i * 25))
    
    def close(self):
        """Clean up pygame resources"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()