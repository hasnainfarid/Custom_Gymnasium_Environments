"""
Emergency Response Environment Testing Suite

Comprehensive testing with 5 different emergency scenarios, performance metrics,
and strategy comparisons for emergency management training.

Author: Hasnain Fareed
License: MIT (2025)
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from emergency_env import EmergencyResponseEnv, EmergencyType, UnitType
import random


@dataclass
class TestResult:
    """Test result data structure"""
    scenario_name: str
    total_casualties: int
    lives_saved: int
    emergencies_resolved: int
    avg_response_time: float
    max_response_time: float
    episode_duration: int
    final_reward: float
    success: bool
    termination_reason: str
    resource_efficiency: float
    coordination_score: float


class EmergencyTestSuite:
    """Comprehensive testing suite for emergency response environment supporting multiple episodes per scenario"""

    def __init__(self, render_mode: str = "human", episodes_per_scenario: int = 5):
        self.render_mode = render_mode
        self.episodes_per_scenario = episodes_per_scenario
        self.test_results: List[TestResult] = []
        self.scenarios_path = os.path.join(os.path.dirname(__file__), "scenarios", "emergency_scenarios.json")

        # Load scenario configurations
        with open(self.scenarios_path, 'r') as f:
            self.scenarios = json.load(f)

    def run_all_tests(self, visualize: bool = True) -> Dict[str, Any]:
        """Run all test scenarios (multiple episodes each) and return comprehensive results"""
        print("=" * 80)
        print("EMERGENCY RESPONSE ENVIRONMENT - COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        print(f"Testing {len(self.scenarios)} different emergency scenarios...")
        print(f"Running {self.episodes_per_scenario} episodes per scenario.")
        print()

        # Test scenarios in order of complexity
        test_order = [
            "single_incident",
            "multiple_incidents",
            "natural_disaster",
            "technological_disaster",
            "mass_casualty_event"
        ]

        for scenario_name in test_order:
            if scenario_name in self.scenarios:
                print(f"Running scenario: {scenario_name.upper().replace('_', ' ')}")
                print("-" * 50)
                scenario_results = []
                # Only create and close the environment once for all episodes of this scenario
                env = EmergencyResponseEnv(render_mode=self.render_mode if visualize else None)
                for ep in range(self.episodes_per_scenario):
                    print(f"  Episode {ep+1}/{self.episodes_per_scenario}...", end="", flush=True)
                    result = self.run_scenario_test(scenario_name, visualize if ep == 0 else False, env=env)
                    scenario_results.append(result)
                    print(f" Done. Reward: {result.final_reward:.0f}, Success: {'✅' if result.success else '❌'}")
                env.close()
                # Aggregate results for this scenario
                agg_result = self._aggregate_scenario_results(scenario_name, scenario_results)
                self.test_results.append(agg_result)
                # Display immediate results
                self._display_scenario_result(agg_result)
                print()

        # Generate comprehensive report
        report = self._generate_comprehensive_report()

        if visualize:
            self._create_visualizations()

        return report

    def run_scenario_test(self, scenario_name: str, visualize: bool = False, env: EmergencyResponseEnv = None) -> TestResult:
        """Run a single episode of a specific scenario test. If env is provided, reuse it for multiple episodes."""
        scenario_config = self.scenarios[scenario_name]

        # Initialize environment if not provided
        close_env = False
        if env is None:
            env = EmergencyResponseEnv(render_mode=self.render_mode if visualize else None)
            close_env = True

        # Apply scenario configuration
        self._configure_scenario(env, scenario_config)

        # Reset environment with scenario
        observation, info = env.reset()

        # Run episode with intelligent agent
        episode_reward = 0
        step_count = 0
        max_steps = scenario_config.get("expected_duration", 600)

        # Strategy selection based on scenario type
        strategy = self._select_strategy(scenario_name, scenario_config)

        while step_count < max_steps:
            # Get intelligent action based on current state
            action = self._get_intelligent_action(env, observation, strategy)

            # Execute step
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            # Render if visualization enabled
            if visualize and step_count % 10 == 0:  # Render every 10 steps
                env.render()
                time.sleep(0.1)

            if terminated or truncated:
                break

        # Calculate performance metrics
        result = self._calculate_test_result(
            scenario_name, scenario_config, env, info,
            episode_reward, step_count
        )

        if close_env:
            env.close()
        return result

    def _aggregate_scenario_results(self, scenario_name: str, results: List[TestResult]) -> TestResult:
        """Aggregate multiple episode results for a scenario into a single TestResult (mean values)"""
        if not results:
            raise ValueError("No results to aggregate for scenario: " + scenario_name)
        # Compute means for numeric fields, majority for success, most common termination reason
        n = len(results)
        total_casualties = int(np.mean([r.total_casualties for r in results]))
        lives_saved = int(np.mean([r.lives_saved for r in results]))
        emergencies_resolved = int(np.mean([r.emergencies_resolved for r in results]))
        avg_response_time = float(np.mean([r.avg_response_time for r in results]))
        max_response_time = float(np.mean([r.max_response_time for r in results]))
        episode_duration = int(np.mean([r.episode_duration for r in results]))
        final_reward = float(np.mean([r.final_reward for r in results]))
        success = sum(r.success for r in results) > n // 2
        # Most common termination reason
        term_reasons = [r.termination_reason for r in results]
        termination_reason = max(set(term_reasons), key=term_reasons.count)
        resource_efficiency = float(np.mean([r.resource_efficiency for r in results]))
        coordination_score = float(np.mean([r.coordination_score for r in results]))
        return TestResult(
            scenario_name=scenario_name,
            total_casualties=total_casualties,
            lives_saved=lives_saved,
            emergencies_resolved=emergencies_resolved,
            avg_response_time=avg_response_time,
            max_response_time=max_response_time,
            episode_duration=episode_duration,
            final_reward=final_reward,
            success=success,
            termination_reason=termination_reason,
            resource_efficiency=resource_efficiency,
            coordination_score=coordination_score
        )

    # --- The rest of the class is unchanged from the original, except for docstrings mentioning multiple episodes ---

    def _configure_scenario(self, env: EmergencyResponseEnv, config: Dict[str, Any]):
        """Configure environment for specific scenario"""
        # Set weather conditions
        if "weather" in config:
            weather = config["weather"]
            env.weather.visibility = weather.get("visibility", 1.0)
            env.weather.road_conditions = weather.get("road_conditions", 1.0)
            env.weather.wind_speed = weather.get("wind_speed", 0.0)
            env.weather.precipitation = weather.get("precipitation", 0.0)
            env.weather.temperature = weather.get("temperature", 0.5)

        # Set traffic conditions
        if "traffic_multiplier" in config:
            env.traffic_congestion *= config["traffic_multiplier"]

        # Apply infrastructure effects
        if "infrastructure_damage" in config:
            damage = config["infrastructure_damage"]

            if "hospital_capacity_reduction" in damage:
                reduction = damage["hospital_capacity_reduction"]
                for hospital in env.hospitals:
                    hospital.beds_available = int(hospital.beds_available * (1 - reduction))
                    hospital.trauma_capacity = int(hospital.trauma_capacity * (1 - reduction))

            if "communication_disruption" in damage:
                disruption = damage["communication_disruption"]
                env.cellular_coverage *= (1 - disruption)
                env.emergency_broadcast_reach *= (1 - disruption)

        # Set initial emergencies
        if "initial_emergencies" in config:
            env.active_emergencies.clear()
            env.emergency_counter = 0

            for emergency_data in config["initial_emergencies"]:
                from emergency_env import EmergencyIncident
                incident = EmergencyIncident(
                    x=emergency_data["x"],
                    y=emergency_data["y"],
                    emergency_type=EmergencyType[emergency_data["type"]],
                    severity=emergency_data["severity"],
                    start_time=0,
                    casualties=emergency_data.get("casualties", 0)
                )
                incident.required_units = env._get_required_units(
                    incident.emergency_type, incident.severity
                )
                env.active_emergencies[env.emergency_counter] = incident
                env.emergency_counter += 1
                env.total_casualties += incident.casualties

    def _select_strategy(self, scenario_name: str, config: Dict[str, Any]) -> str:
        """Select appropriate strategy based on scenario"""
        if scenario_name == "single_incident":
            return "focused_response"
        elif scenario_name == "multiple_incidents":
            return "resource_optimization"
        elif scenario_name == "natural_disaster":
            return "disaster_management"
        elif scenario_name == "technological_disaster":
            return "hazmat_protocol"
        elif scenario_name == "mass_casualty_event":
            return "mass_casualty"
        else:
            return "adaptive"

    def _get_intelligent_action(self, env: EmergencyResponseEnv, observation: np.ndarray,
                              strategy: str) -> int:
        """Get intelligent action based on current state and strategy"""
        # Analyze current situation
        active_emergencies = len(env.active_emergencies)
        available_units = sum(1 for unit in env.response_units if not unit.busy)
        high_severity_incidents = sum(1 for inc in env.active_emergencies.values()
                                    if inc.severity >= 7)

        # Strategy-based decision making
        if strategy == "focused_response":
            return self._focused_response_action(env, observation)
        elif strategy == "resource_optimization":
            return self._resource_optimization_action(env, observation)
        elif strategy == "disaster_management":
            return self._disaster_management_action(env, observation)
        elif strategy == "hazmat_protocol":
            return self._hazmat_protocol_action(env, observation)
        elif strategy == "mass_casualty":
            return self._mass_casualty_action(env, observation)
        else:
            return self._adaptive_action(env, observation)

    def _focused_response_action(self, env: EmergencyResponseEnv, observation: np.ndarray) -> int:
        """Focused response for single incident scenarios"""
        # Priority: Dispatch appropriate units quickly
        for incident_id, incident in env.active_emergencies.items():
            required_units = incident.required_units
            current_units = len(incident.active_units)

            # Find best available unit for this incident
            for unit_idx, unit in enumerate(env.response_units):
                if not unit.busy and unit.unit_type in required_units:
                    return unit_idx  # Dispatch this unit

        return 49  # Normal operations if no dispatch needed

    def _resource_optimization_action(self, env: EmergencyResponseEnv, observation: np.ndarray) -> int:
        """Resource optimization for multiple incidents"""
        # Priority: Balance resources across incidents
        incident_priorities = []

        for incident_id, incident in env.active_emergencies.items():
            priority = incident.severity * 10 + incident.casualties * 5
            if incident.escalation_risk > 0.7:
                priority += 50
            incident_priorities.append((priority, incident_id, incident))

        # Sort by priority
        incident_priorities.sort(reverse=True)

        # Dispatch to highest priority incident
        if incident_priorities:
            _, incident_id, incident = incident_priorities[0]
            for unit_idx, unit in enumerate(env.response_units):
                if not unit.busy and unit.unit_type in incident.required_units:
                    return unit_idx

        # Establish evacuation if needed
        if len(env.active_emergencies) >= 3:
            return 20  # Establish evacuation zone

        return 49

    def _disaster_management_action(self, env: EmergencyResponseEnv, observation: np.ndarray) -> int:
        """Disaster management for natural disasters"""
        # Priority: Declare emergency, coordinate resources
        if not env.state_of_emergency and len(env.active_emergencies) >= 2:
            return 44  # Declare state of emergency

        if not env.mutual_aid_requested and len(env.active_emergencies) >= 3:
            return 26  # Request mutual aid

        if not env.national_guard_active and env.state_of_emergency:
            return 46  # Request National Guard

        # Establish incident command
        if env.cellular_coverage < 0.8:
            return 48  # Establish incident command post

        # Dispatch search and rescue first
        for unit_idx, unit in enumerate(env.response_units):
            if not unit.busy and unit.unit_type == UnitType.SEARCH_RESCUE:
                return unit_idx

        return self._focused_response_action(env, observation)

    def _hazmat_protocol_action(self, env: EmergencyResponseEnv, observation: np.ndarray) -> int:
        """Hazmat protocol for chemical/technological disasters"""
        # Priority: Hazmat response and evacuation
        has_chemical_incident = any(inc.emergency_type == EmergencyType.CHEMICAL_SPILL
                                 for inc in env.active_emergencies.values())

        if has_chemical_incident:
            # Establish evacuation zones immediately
            for zone_id in range(6):
                if not env.evacuation_zones[zone_id]:
                    return 20 + zone_id

            # Dispatch hazmat units
            for unit_idx, unit in enumerate(env.response_units):
                if not unit.busy and unit.unit_type == UnitType.HAZMAT:
                    return unit_idx

        # Coordinate with utilities
        return 47

    def _mass_casualty_action(self, env: EmergencyResponseEnv, observation: np.ndarray) -> int:
        """Mass casualty response"""
        total_casualties = sum(inc.casualties for inc in env.active_emergencies.values())

        if total_casualties > 20:
            # Activate emergency shelters
            return 32

        if total_casualties > 10:
            # Request mutual aid
            if not env.mutual_aid_requested:
                return 26

        # Prioritize ambulances
        for unit_idx, unit in enumerate(env.response_units):
            if not unit.busy and unit.unit_type == UnitType.AMBULANCE:
                return unit_idx

        # Activate emergency broadcast
        return 45

    def _adaptive_action(self, env: EmergencyResponseEnv, observation: np.ndarray) -> int:
        """Adaptive strategy that combines multiple approaches"""
        # Analyze situation complexity
        complexity_score = (len(env.active_emergencies) * 2 +
                          sum(inc.severity for inc in env.active_emergencies.values()) +
                          env.total_casualties)

        if complexity_score > 50:
            return self._disaster_management_action(env, observation)
        elif complexity_score > 20:
            return self._resource_optimization_action(env, observation)
        else:
            return self._focused_response_action(env, observation)

    def _calculate_test_result(self, scenario_name: str, config: Dict[str, Any],
                             env: EmergencyResponseEnv, info: Dict[str, Any],
                             episode_reward: float, step_count: int) -> TestResult:
        """Calculate comprehensive test result for a single episode"""
        # Success criteria check
        success_criteria = config.get("success_criteria", {})
        success = True

        if "max_casualties" in success_criteria:
            if env.total_casualties > success_criteria["max_casualties"]:
                success = False

        if "max_response_time" in success_criteria and env.response_times:
            if max(env.response_times) > success_criteria["max_response_time"]:
                success = False

        # Calculate resource efficiency
        total_units = len(env.response_units)
        units_used = sum(1 for unit in env.response_units if unit.fatigue > 0)
        resource_efficiency = units_used / total_units if total_units > 0 else 0

        # Calculate coordination score
        unit_types_used = len(set(unit.unit_type for unit in env.response_units if unit.fatigue > 0))
        coordination_score = unit_types_used / len(UnitType) * 100

        return TestResult(
            scenario_name=scenario_name,
            total_casualties=env.total_casualties,
            lives_saved=env.lives_saved,
            emergencies_resolved=env.emergencies_resolved,
            avg_response_time=np.mean(env.response_times) if env.response_times else 0,
            max_response_time=max(env.response_times) if env.response_times else 0,
            episode_duration=step_count,
            final_reward=episode_reward,
            success=success,
            termination_reason=info.get("termination_reason", "completed"),
            resource_efficiency=resource_efficiency,
            coordination_score=coordination_score
        )

    def _display_scenario_result(self, result: TestResult):
        """Display results for a scenario (aggregated over multiple episodes)"""
        status = "✅ SUCCESS" if result.success else "❌ FAILED"
        print(f"Status: {status}")
        print(f"Casualties: {result.total_casualties} | Lives Saved: {result.lives_saved}")
        print(f"Emergencies Resolved: {result.emergencies_resolved}")
        print(f"Avg Response Time: {result.avg_response_time:.1f} min")
        print(f"Resource Efficiency: {result.resource_efficiency:.1%}")
        print(f"Coordination Score: {result.coordination_score:.1f}/100")
        print(f"Final Reward: {result.final_reward:.0f}")
        print(f"Duration: {result.episode_duration} timesteps")
        print(f"Termination: {result.termination_reason}")

    # The rest of the class (report generation, visualization, strategy comparison) is unchanged.

    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report (aggregated over multiple episodes per scenario)"""
        # ... unchanged from original ...
        print("=" * 80)
        print("COMPREHENSIVE TEST RESULTS SUMMARY")
        print("=" * 80)

        # Overall statistics
        total_scenarios = len(self.test_results)
        successful_scenarios = sum(1 for r in self.test_results if r.success)
        success_rate = successful_scenarios / total_scenarios * 100

        total_casualties = sum(r.total_casualties for r in self.test_results)
        total_lives_saved = sum(r.lives_saved for r in self.test_results)
        total_emergencies = sum(r.emergencies_resolved for r in self.test_results)

        avg_response_time = np.mean([r.avg_response_time for r in self.test_results if r.avg_response_time > 0])
        avg_resource_efficiency = np.mean([r.resource_efficiency for r in self.test_results])
        avg_coordination = np.mean([r.coordination_score for r in self.test_results])

        print(f"Overall Success Rate: {success_rate:.1f}% ({successful_scenarios}/{total_scenarios})")
        print(f"Total Casualties: {total_casualties}")
        print(f"Total Lives Saved: {total_lives_saved}")
        print(f"Total Emergencies Resolved: {total_emergencies}")
        print(f"Average Response Time: {avg_response_time:.1f} minutes")
        print(f"Average Resource Efficiency: {avg_resource_efficiency:.1%}")
        print(f"Average Coordination Score: {avg_coordination:.1f}/100")
        print()

        # Scenario breakdown
        print("SCENARIO BREAKDOWN:")
        print("-" * 50)
        for result in self.test_results:
            status_icon = "✅" if result.success else "❌"
            print(f"{status_icon} {result.scenario_name.replace('_', ' ').title()}")
            print(f"   Casualties: {result.total_casualties}, Lives Saved: {result.lives_saved}")
            print(f"   Response Time: {result.avg_response_time:.1f}min, Reward: {result.final_reward:.0f}")
            print()

        # Performance analysis
        print("PERFORMANCE ANALYSIS:")
        print("-" * 50)

        best_scenario = min(self.test_results, key=lambda r: r.total_casualties)
        worst_scenario = max(self.test_results, key=lambda r: r.total_casualties)

        print(f"Best Performance: {best_scenario.scenario_name} ({best_scenario.total_casualties} casualties)")
        print(f"Worst Performance: {worst_scenario.scenario_name} ({worst_scenario.total_casualties} casualties)")

        fastest_response = min(self.test_results, key=lambda r: r.avg_response_time if r.avg_response_time > 0 else float('inf'))
        print(f"Fastest Response: {fastest_response.scenario_name} ({fastest_response.avg_response_time:.1f} min)")

        # Strategy effectiveness
        print()
        print("STRATEGY EFFECTIVENESS ANALYSIS:")
        print("-" * 50)

        strategy_map = {
            "single_incident": "Focused Response",
            "multiple_incidents": "Resource Optimization",
            "natural_disaster": "Disaster Management",
            "technological_disaster": "Hazmat Protocol",
            "mass_casualty_event": "Mass Casualty Response"
        }

        for result in self.test_results:
            strategy = strategy_map.get(result.scenario_name, "Unknown")
            effectiveness = "Highly Effective" if result.success and result.total_casualties < 10 else \
                          "Effective" if result.success else \
                          "Needs Improvement"
            print(f"{strategy}: {effectiveness}")
            print(f"   Key Metrics: {result.total_casualties} casualties, {result.coordination_score:.0f}% coordination")

        print()
        print("RECOMMENDATIONS:")
        print("-" * 50)

        recommendations = []

        if avg_response_time > 20:
            recommendations.append("• Improve unit positioning and dispatch protocols")

        if total_casualties > total_lives_saved:
            recommendations.append("• Focus on life-saving interventions and medical response")

        if avg_resource_efficiency < 0.6:
            recommendations.append("• Optimize resource allocation and reduce unit idle time")

        if avg_coordination < 70:
            recommendations.append("• Enhance inter-agency coordination and communication")

        if success_rate < 80:
            recommendations.append("• Review and improve emergency response strategies")

        if not recommendations:
            recommendations.append("• Excellent performance across all metrics!")

        for rec in recommendations:
            print(rec)

        # Generate report dictionary
        report = {
            "summary": {
                "total_scenarios": total_scenarios,
                "success_rate": success_rate,
                "total_casualties": total_casualties,
                "total_lives_saved": total_lives_saved,
                "avg_response_time": avg_response_time,
                "avg_resource_efficiency": avg_resource_efficiency,
                "avg_coordination": avg_coordination
            },
            "scenarios": [
                {
                    "name": r.scenario_name,
                    "success": r.success,
                    "casualties": r.total_casualties,
                    "lives_saved": r.lives_saved,
                    "response_time": r.avg_response_time,
                    "reward": r.final_reward
                } for r in self.test_results
            ],
            "recommendations": recommendations
        }

        return report

    def _create_visualizations(self):
        """Create performance visualization charts (aggregated over multiple episodes per scenario)"""
        # ... unchanged from original ...

        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

            # Scenario names for x-axis
            scenario_names = [r.scenario_name.replace('_', '\n').title() for r in self.test_results]

            # 1. Casualties vs Lives Saved
            casualties = [r.total_casualties for r in self.test_results]
            lives_saved = [r.lives_saved for r in self.test_results]

            x = np.arange(len(scenario_names))
            width = 0.35

            ax1.bar(x - width/2, casualties, width, label='Casualties', color='red', alpha=0.7)
            ax1.bar(x + width/2, lives_saved, width, label='Lives Saved', color='green', alpha=0.7)
            ax1.set_xlabel('Scenarios')
            ax1.set_ylabel('Count')
            ax1.set_title('Casualties vs Lives Saved by Scenario')
            ax1.set_xticks(x)
            ax1.set_xticklabels(scenario_names, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Response Times
            response_times = [r.avg_response_time for r in self.test_results]
            colors = ['green' if r.success else 'red' for r in self.test_results]

            bars = ax2.bar(scenario_names, response_times, color=colors, alpha=0.7)
            ax2.set_xlabel('Scenarios')
            ax2.set_ylabel('Average Response Time (minutes)')
            ax2.set_title('Response Time Performance')
            ax2.set_xticklabels(scenario_names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)

            # Add target line (20 minutes)
            ax2.axhline(y=20, color='orange', linestyle='--', label='Target (20 min)')
            ax2.legend()

            # 3. Resource Efficiency and Coordination
            efficiency = [r.resource_efficiency * 100 for r in self.test_results]
            coordination = [r.coordination_score for r in self.test_results]

            ax3.scatter(efficiency, coordination, c=casualties, s=100, alpha=0.7, cmap='RdYlGn_r')
            ax3.set_xlabel('Resource Efficiency (%)')
            ax3.set_ylabel('Coordination Score')
            ax3.set_title('Resource Efficiency vs Coordination\n(Color: Casualties)')
            ax3.grid(True, alpha=0.3)

            # Add colorbar
            cbar = plt.colorbar(ax3.collections[0], ax=ax3)
            cbar.set_label('Casualties')

            # 4. Overall Performance Radar
            rewards = [r.final_reward for r in self.test_results]
            success_scores = [100 if r.success else 0 for r in self.test_results]

            ax4.plot(scenario_names, success_scores, 'go-', label='Success Rate', linewidth=2, markersize=8)
            ax4_twin = ax4.twinx()
            ax4_twin.plot(scenario_names, rewards, 'bo-', label='Final Reward', linewidth=2, markersize=8)

            ax4.set_xlabel('Scenarios')
            ax4.set_ylabel('Success Rate', color='green')
            ax4_twin.set_ylabel('Final Reward', color='blue')
            ax4.set_title('Success Rate and Reward by Scenario')
            ax4.set_xticklabels(scenario_names, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)

            # Combine legends
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            plt.tight_layout()
            plt.savefig('emergency_response_test_results.png', dpi=300, bbox_inches='tight')
            print(f"\n📊 Visualization saved as 'emergency_response_test_results.png'")

            if self.render_mode == "human":
                plt.show()

        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")

    def compare_strategies(self) -> Dict[str, Any]:
        """Compare different emergency response strategies (using aggregated scenario results)"""
        # ... unchanged from original ...

        print("\n" + "=" * 80)
        print("STRATEGY COMPARISON ANALYSIS")
        print("=" * 80)

        strategies = {
            "Centralized Command": "Single command center coordinates all responses",
            "Distributed Response": "Local units respond independently to nearest incidents",
            "Priority-Based": "Resources allocated based on incident severity and casualties",
            "Resource-Balanced": "Equal distribution of resources across all incidents",
            "Adaptive": "Strategy changes based on situation complexity"
        }

        print("STRATEGY DESCRIPTIONS:")
        print("-" * 50)
        for strategy, description in strategies.items():
            print(f"• {strategy}: {description}")

        print("\nSTRATEGY EFFECTIVENESS:")
        print("-" * 50)

        # Analyze results by strategy type
        strategy_performance = {}

        for result in self.test_results:
            scenario_type = result.scenario_name

            if scenario_type == "single_incident":
                strategy_type = "Centralized Command"
            elif scenario_type == "multiple_incidents":
                strategy_type = "Resource-Balanced"
            elif scenario_type == "natural_disaster":
                strategy_type = "Adaptive"
            elif scenario_type == "technological_disaster":
                strategy_type = "Priority-Based"
            else:
                strategy_type = "Distributed Response"

            if strategy_type not in strategy_performance:
                strategy_performance[strategy_type] = []

            strategy_performance[strategy_type].append({
                'success': result.success,
                'casualties': result.total_casualties,
                'response_time': result.avg_response_time,
                'efficiency': result.resource_efficiency
            })

        # Calculate strategy metrics
        strategy_metrics = {}
        for strategy, results in strategy_performance.items():
            success_rate = sum(1 for r in results if r['success']) / len(results) * 100
            avg_casualties = np.mean([r['casualties'] for r in results])
            avg_response_time = np.mean([r['response_time'] for r in results])
            avg_efficiency = np.mean([r['efficiency'] for r in results]) * 100

            strategy_metrics[strategy] = {
                'success_rate': success_rate,
                'avg_casualties': avg_casualties,
                'avg_response_time': avg_response_time,
                'avg_efficiency': avg_efficiency
            }

        # Display strategy comparison
        for strategy, metrics in strategy_metrics.items():
            print(f"\n{strategy}:")
            print(f"  Success Rate: {metrics['success_rate']:.1f}%")
            print(f"  Avg Casualties: {metrics['avg_casualties']:.1f}")
            print(f"  Avg Response Time: {metrics['avg_response_time']:.1f} min")
            print(f"  Resource Efficiency: {metrics['avg_efficiency']:.1f}%")

        # Recommendations
        print("\nSTRATEGY RECOMMENDATIONS:")
        print("-" * 50)

        best_overall = max(strategy_metrics.items(),
                          key=lambda x: x[1]['success_rate'] - x[1]['avg_casualties']/10)

        print(f"• Best Overall Strategy: {best_overall[0]}")

        fastest_response = min(strategy_metrics.items(),
                             key=lambda x: x[1]['avg_response_time'])
        print(f"• Fastest Response: {fastest_response[0]} ({fastest_response[1]['avg_response_time']:.1f} min)")

        most_efficient = max(strategy_metrics.items(),
                           key=lambda x: x[1]['avg_efficiency'])
        print(f"• Most Efficient: {most_efficient[0]} ({most_efficient[1]['avg_efficiency']:.1f}%)")

        return strategy_metrics

    def generate_training_report(self) -> str:
        """Generate after-action report for emergency management training (aggregated over multiple episodes)"""
        # ... unchanged from original ...
        report = []
        report.append("EMERGENCY RESPONSE TRAINING - AFTER ACTION REPORT")
        report.append("=" * 60)
        report.append(f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        report.append("EXECUTIVE SUMMARY:")
        report.append("-" * 30)
        total_scenarios = len(self.test_results)
        successful = sum(1 for r in self.test_results if r.success)
        report.append(f"• Scenarios Tested: {total_scenarios}")
        report.append(f"• Success Rate: {successful/total_scenarios*100:.1f}%")
        report.append(f"• Total Casualties: {sum(r.total_casualties for r in self.test_results)}")
        report.append(f"• Lives Saved: {sum(r.lives_saved for r in self.test_results)}")
        report.append("")

        report.append("CRITICAL DECISION POINTS:")
        report.append("-" * 30)

        for result in self.test_results:
            report.append(f"\n{result.scenario_name.replace('_', ' ').title()}:")

            if result.total_casualties > 20:
                report.append("  ⚠️  High casualty count - review triage and medical response")

            if result.avg_response_time > 30:
                report.append("  ⚠️  Slow response time - optimize unit deployment")

            if not result.success:
                report.append("  ❌ Mission failure - analyze root causes")
            else:
                report.append("  ✅ Mission success")

            report.append(f"     Response Time: {result.avg_response_time:.1f} minutes")
            report.append(f"     Resource Usage: {result.resource_efficiency:.1%}")
            report.append(f"     Coordination: {result.coordination_score:.0f}/100")

        report.append("\n\nLESSONS LEARNED:")
        report.append("-" * 30)

        # Analyze patterns
        high_casualty_scenarios = [r for r in self.test_results if r.total_casualties > 15]
        if high_casualty_scenarios:
            report.append("• High-casualty scenarios require immediate medical surge capacity")

        slow_response_scenarios = [r for r in self.test_results if r.avg_response_time > 25]
        if slow_response_scenarios:
            report.append("• Pre-positioning resources in high-risk areas improves response times")

        low_coordination_scenarios = [r for r in self.test_results if r.coordination_score < 60]
        if low_coordination_scenarios:
            report.append("• Multi-agency coordination protocols need strengthening")

        report.append("• Early declaration of emergency status unlocks critical resources")
        report.append("• Proactive evacuation reduces civilian exposure")
        report.append("• Communication redundancy is essential for command and control")

        report.append("\n\nTRAINING RECOMMENDATIONS:")
        report.append("-" * 30)
        report.append("1. Conduct regular multi-agency exercises")
        report.append("2. Practice resource allocation under time pressure")
        report.append("3. Improve incident command system implementation")
        report.append("4. Enhance public warning and evacuation procedures")
        report.append("5. Develop contingency plans for communication failures")

        return "\n".join(report)


def main():
    """Main testing function supporting multiple episodes per scenario"""
    print("Initializing Emergency Response Environment Test Suite...")

    # Create test suite
    test_suite = EmergencyTestSuite(render_mode="human", episodes_per_scenario=5)

    # Run all tests
    results = test_suite.run_all_tests(visualize=True)

    # Compare strategies
    strategy_comparison = test_suite.compare_strategies()

    # Generate training report
    training_report = test_suite.generate_training_report()

    # Save training report
    with open("emergency_response_training_report.txt", "w") as f:
        f.write(training_report)

    print(f"\n📋 Training report saved as 'emergency_response_training_report.txt'")
    print("\n🎯 Testing complete! Review results and recommendations above.")

    return results, strategy_comparison, training_report


if __name__ == "__main__":
    main()
