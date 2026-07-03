"""
Environmental Impact Estimation Module
Calculates energy, water, and carbon footprints for AI systems
"""
import numpy as np

class EnvironmentalCalculator:
    def __init__(self, pue=1.4, carbon_intensity=450, wue=1.0):
        """
        Args:
            pue: Power Usage Effectiveness (1.1-1.8 typical)
            carbon_intensity: gCO2e per kWh (regional grid average)
            wue: Water Usage Effectiveness (L per kWh)
        """
        self.pue = pue
        self.carbon_intensity = carbon_intensity  # gCO2e/kWh
        self.wue = wue  # L/kWh

        # GPU power ratings (Watts)
        self.gpu_power = {
            'A100': 400,
            'H100': 700,
            'V100': 300,
            'RTX4090': 450,
            'T4': 70
        }

    def calculate_inference_energy(self, gpu_type='A100', num_gpus=1, 
                                    inference_time_seconds=1.0, batch_size=1):
        """Calculate energy consumption per inference"""
        gpu_watts = self.gpu_power.get(gpu_type, 400)
        total_gpu_power = gpu_watts * num_gpus  # Watts

        # Energy in kWh
        energy_kwh = (total_gpu_power * inference_time_seconds / 3600) * self.pue

        # Adjust for batch size (amortized across batch)
        energy_per_inference = energy_kwh / batch_size

        return energy_per_inference

    def calculate_training_energy(self, gpu_type='A100', num_gpus=8, 
                                   training_hours=24, num_epochs=3):
        """Calculate total training energy"""
        gpu_watts = self.gpu_power.get(gpu_type, 400)
        total_gpu_power = gpu_watts * num_gpus  # Watts

        total_hours = training_hours * num_epochs
        energy_kwh = (total_gpu_power * total_hours / 1000) * self.pue

        return energy_kwh

    def calculate_water_footprint(self, energy_kwh):
        """Calculate water consumption based on energy usage"""
        # Direct cooling water + indirect water from electricity generation
        water_liters = energy_kwh * self.wue
        return water_liters

    def calculate_carbon_footprint(self, energy_kwh):
        """Calculate carbon emissions based on energy and grid intensity"""
        carbon_g = energy_kwh * self.carbon_intensity
        carbon_kg = carbon_g / 1000
        return carbon_kg

    def calculate_full_footprint(self, gpu_type='A100', num_gpus=1, 
                                  inference_time_seconds=1.0, batch_size=1,
                                  scenario='inference'):
        """Calculate complete environmental footprint"""
        if scenario == 'inference':
            energy = self.calculate_inference_energy(gpu_type, num_gpus, 
                                                    inference_time_seconds, batch_size)
        else:
            energy = self.calculate_training_energy(gpu_type, num_gpus)

        water = self.calculate_water_footprint(energy)
        carbon = self.calculate_carbon_footprint(energy)

        return {
            'energy_kwh': energy,
            'water_liters': water,
            'carbon_kgco2e': carbon,
            'pue': self.pue,
            'carbon_intensity': self.carbon_intensity,
            'wue': self.wue
        }

    def compare_grids(self, energy_kwh, grid_intensities=None):
        """Compare carbon footprint across different grid regions"""
        if grid_intensities is None:
            grid_intensities = {
                'Iceland/Renewable': 50,
                'France/Nuclear': 80,
                'US Average': 450,
                'China/Coal': 800,
                'India/Coal': 750
            }

        comparisons = {}
        for region, intensity in grid_intensities.items():
            carbon = energy_kwh * intensity / 1000
            comparisons[region] = {
                'carbon_intensity': intensity,
                'carbon_kgco2e': carbon
            }

        return comparisons

    def estimate_annual_impact(self, daily_queries=1000000, avg_energy_per_query=0.00014):
        """Estimate annual environmental impact at scale"""
        annual_energy = daily_queries * avg_energy_per_query * 365
        annual_water = self.calculate_water_footprint(annual_energy)
        annual_carbon = self.calculate_carbon_footprint(annual_energy)

        return {
            'annual_energy_mwh': annual_energy / 1000,
            'annual_water_million_liters': annual_water / 1e6,
            'annual_carbon_kilotons': annual_carbon / 1000,
            'equivalent_homes': annual_energy / 10000  # Approximate
        }

class OptimizationRecommender:
    def __init__(self):
        self.techniques = {
            'quantization': {'energy_reduction': 0.42, 'accuracy_loss': 0.008},
            'dynamic_batching': {'energy_reduction': 0.25, 'accuracy_loss': 0.0},
            'knowledge_distillation': {'energy_reduction': 0.50, 'accuracy_loss': 0.02},
            'pruning': {'energy_reduction': 0.30, 'accuracy_loss': 0.015},
            'geographic_shifting': {'energy_reduction': 0.20, 'accuracy_loss': 0.0}
        }

    def recommend(self, current_footprint, priority='balanced'):
        """Generate optimization recommendations"""
        recommendations = []

        for technique, params in self.techniques.items():
            new_energy = current_footprint['energy_kwh'] * (1 - params['energy_reduction'])
            new_water = current_footprint['water_liters'] * (1 - params['energy_reduction'])
            new_carbon = current_footprint['carbon_kgco2e'] * (1 - params['energy_reduction'])

            recommendations.append({
                'technique': technique,
                'energy_reduction_%': params['energy_reduction'] * 100,
                'accuracy_loss_%': params['accuracy_loss'] * 100,
                'projected_energy_kwh': new_energy,
                'projected_water_liters': new_water,
                'projected_carbon_kgco2e': new_carbon
            })

        # Sort by priority
        if priority == 'energy':
            recommendations.sort(key=lambda x: x['energy_reduction_%'], reverse=True)
        elif priority == 'accuracy':
            recommendations.sort(key=lambda x: x['accuracy_loss_%'])

        return recommendations
