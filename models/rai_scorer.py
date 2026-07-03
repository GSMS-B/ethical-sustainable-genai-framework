"""
Responsible AI (RAI) Scoring Model
Aggregates ethical and environmental scores into unified governance metric
"""
import numpy as np

class RAIScorer:
    def __init__(self, w1=0.4, w2=0.35, w3=0.25):
        """
        Args:
            w1: Weight for Ethical Score
            w2: Weight for Environmental Score
            w3: Weight for Transparency Score
        """
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def calculate_environmental_score(self, energy_kwh, water_liters, carbon_kgco2e,
                                       baseline_energy=1.0, baseline_water=2.0, 
                                       baseline_carbon=0.5):
        """
        Calculate environmental score using harmonic mean
        Higher score = better environmental performance (lower impact)
        """
        # Normalize against baselines (lower is better, so invert)
        s_e = max(0.01, 1.0 - (energy_kwh / baseline_energy))
        s_w = max(0.01, 1.0 - (water_liters / baseline_water))
        s_c = max(0.01, 1.0 - (carbon_kgco2e / baseline_carbon))

        # Harmonic mean
        s = 3 / (1/s_e + 1/s_w + 1/s_c)
        return min(s, 1.0) * 100

    def calculate_transparency_score(self, data_disclosure=0, 
                                      architecture_transparency=0,
                                      environmental_reporting=0):
        """
        Calculate transparency score (0-100)
        """
        # Data disclosure (0-40 points)
        data_score = min(data_disclosure, 40)

        # Architecture transparency (0-30 points)
        arch_score = min(architecture_transparency, 30)

        # Environmental reporting compliance (0-30 points)
        env_score = min(environmental_reporting, 30)

        return data_score + arch_score + env_score

    def calculate_rai_score(self, ethical_score, environmental_score, transparency_score):
        """
        Calculate composite RAI score
        RAI = w1*E + w2*S + w3*T
        """
        rai = (self.w1 * ethical_score + 
               self.w2 * environmental_score + 
               self.w3 * transparency_score)

        return rai

    def classify_rai(self, rai_score):
        """Classify RAI score into tiers"""
        if rai_score >= 85:
            return 'Excellent'
        elif rai_score >= 70:
            return 'Good'
        elif rai_score >= 55:
            return 'Moderate'
        elif rai_score >= 40:
            return 'Poor'
        else:
            return 'Unacceptable'

    def full_evaluation(self, ethical_score, energy_kwh, water_liters, carbon_kgco2e,
                        data_disclosure=0, architecture_transparency=0, 
                        environmental_reporting=0):
        """Complete RAI evaluation pipeline"""
        env_score = self.calculate_environmental_score(energy_kwh, water_liters, carbon_kgco2e)
        trans_score = self.calculate_transparency_score(data_disclosure, 
                                                         architecture_transparency,
                                                         environmental_reporting)
        rai = self.calculate_rai_score(ethical_score, env_score, trans_score)
        classification = self.classify_rai(rai)

        return {
            'ethical_score': ethical_score,
            'environmental_score': env_score,
            'transparency_score': trans_score,
            'rai_score': rai,
            'classification': classification
        }

class ScenarioSimulator:
    def __init__(self, rai_scorer):
        self.rai_scorer = rai_scorer
        self.scenarios = {
            'military': {
                'description': 'Strategic military decision-making',
                'base_risk': 0.9,
                'complexity': 'high'
            },
            'crisis': {
                'description': 'Natural disaster resource allocation',
                'base_risk': 0.6,
                'complexity': 'medium'
            },
            'policy': {
                'description': 'Public policy recommendation',
                'base_risk': 0.45,
                'complexity': 'medium'
            },
            'medical': {
                'description': 'Medical diagnosis and treatment',
                'base_risk': 0.85,
                'complexity': 'high'
            },
            'financial': {
                'description': 'Financial trading and lending',
                'base_risk': 0.75,
                'complexity': 'high'
            }
        }

    def simulate_scenario(self, scenario_type, model_params):
        """Simulate a scenario and return RAI evaluation"""
        scenario = self.scenarios.get(scenario_type, self.scenarios['policy'])

        # Adjust ethical score based on scenario risk
        base_ethical = model_params.get('ethical_score', 70)
        risk_penalty = scenario['base_risk'] * 15
        adjusted_ethical = max(0, base_ethical - risk_penalty)

        result = self.rai_scorer.full_evaluation(
            ethical_score=adjusted_ethical,
            energy_kwh=model_params.get('energy_kwh', 0.1),
            water_liters=model_params.get('water_liters', 0.2),
            carbon_kgco2e=model_params.get('carbon_kgco2e', 0.05),
            data_disclosure=model_params.get('data_disclosure', 20),
            architecture_transparency=model_params.get('architecture_transparency', 15),
            environmental_reporting=model_params.get('environmental_reporting', 10)
        )

        result['scenario_type'] = scenario_type
        result['scenario_description'] = scenario['description']
        return result
