"""
Clinical Trials Simulation using Multi-Armed Bandits

This module implements a clinical trial system that uses bandit algorithms
to optimize treatment allocation, patient safety, and trial efficiency.
The system handles multiple treatment arms, patient characteristics, and
ethical constraints.

Key Features:
- Adaptive trial design with multiple treatment arms
- Patient safety monitoring and early stopping
- Ethical considerations and randomization
- Treatment efficacy and safety analysis
- Sample size optimization
- Real-time monitoring and decision making
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

@dataclass
class Treatment:
    """Treatment arm data structure."""
    treatment_id: int
    name: str
    dosage: str
    mechanism: str
    expected_efficacy: float  # Expected success rate
    expected_safety: float    # Expected safety rate (1 - adverse event rate)
    cost: float
    features: np.ndarray = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = np.random.randn(10)  # Random features for demo

@dataclass
class Patient:
    """Patient data structure with demographics and medical history."""
    patient_id: int
    age: int
    gender: str
    medical_history: List[str]
    biomarkers: Dict[str, float]
    risk_factors: List[str]
    feature_vector: np.ndarray = None
    
    def __post_init__(self):
        if self.feature_vector is None:
            self.feature_vector = np.random.randn(10)  # Random features for demo

@dataclass
class TrialArm:
    """Clinical trial arm data structure."""
    arm_id: int
    treatment: Treatment
    patients: List[int] = None
    outcomes: List[Dict] = None
    current_size: int = 0
    success_count: int = 0
    adverse_events: int = 0
    
    def __post_init__(self):
        if self.patients is None:
            self.patients = []
        if self.outcomes is None:
            self.outcomes = []

class ClinicalTrialBandit:
    """
    Clinical trial system using bandit algorithms.
    
    This class implements various bandit-based approaches for adaptive
    clinical trial design, patient allocation, and treatment optimization.
    """
    
    def __init__(self, 
                 num_treatments: int = 5,
                 num_patients: int = 500,
                 feature_dim: int = 10,
                 max_trial_duration: int = 365):
        """
        Initialize the clinical trial bandit.
        
        Args:
            num_treatments: Number of treatment arms
            num_patients: Number of patients in the trial
            feature_dim: Dimension of treatment/patient feature vectors
            max_trial_duration: Maximum trial duration in days
        """
        self.num_treatments = num_treatments
        self.num_patients = num_patients
        self.feature_dim = feature_dim
        self.max_trial_duration = max_trial_duration
        
        # Initialize treatments and patients
        self.treatments = self._generate_treatments()
        self.patients = self._generate_patients()
        self.trial_arms = self._initialize_trial_arms()
        
        # Bandit state
        self.treatment_estimates = defaultdict(lambda: defaultdict(float))
        self.treatment_counts = defaultdict(lambda: defaultdict(int))
        self.patient_treatment_assignments = {}
        
        # Performance tracking
        self.trial_outcomes = []
        self.safety_events = []
        self.efficacy_results = []
        self.ethical_violations = []
        
        # Trial state
        self.current_day = 0
        self.enrolled_patients = set()
        self.completed_patients = set()
        
        # Safety monitoring
        self.safety_thresholds = {
            'adverse_event_rate': 0.15,  # 15% adverse event rate threshold
            'efficacy_threshold': 0.30,   # 30% efficacy threshold
            'sample_size_min': 20,        # Minimum patients per arm
            'early_stopping_threshold': 0.25  # 25% adverse event rate for early stopping
        }
        
    def _generate_treatments(self) -> Dict[int, Treatment]:
        """Generate synthetic treatment data."""
        treatments = {}
        mechanisms = ['inhibitor', 'agonist', 'antibody', 'vaccine', 'gene_therapy']
        dosages = ['low', 'medium', 'high']
        
        for i in range(self.num_treatments):
            treatment_id = i + 1
            name = f"Treatment {treatment_id}"
            mechanism = random.choice(mechanisms)
            dosage = random.choice(dosages)
            
            # Generate realistic efficacy and safety profiles
            if mechanism == 'antibody':
                expected_efficacy = random.uniform(0.4, 0.7)  # Higher efficacy
                expected_safety = random.uniform(0.8, 0.95)   # Good safety
            elif mechanism == 'vaccine':
                expected_efficacy = random.uniform(0.3, 0.6)
                expected_safety = random.uniform(0.85, 0.98)
            elif mechanism == 'inhibitor':
                expected_efficacy = random.uniform(0.2, 0.5)
                expected_safety = random.uniform(0.7, 0.9)
            else:
                expected_efficacy = random.uniform(0.1, 0.4)
                expected_safety = random.uniform(0.6, 0.85)
            
            cost = random.uniform(1000, 10000)
            
            treatments[treatment_id] = Treatment(
                treatment_id=treatment_id,
                name=name,
                dosage=dosage,
                mechanism=mechanism,
                expected_efficacy=expected_efficacy,
                expected_safety=expected_safety,
                cost=cost
            )
        
        return treatments
    
    def _generate_patients(self) -> Dict[int, Patient]:
        """Generate synthetic patient data."""
        patients = {}
        medical_conditions = ['diabetes', 'hypertension', 'obesity', 'smoking', 
                            'heart_disease', 'cancer_history', 'autoimmune']
        risk_factors = ['age_65_plus', 'obese', 'smoker', 'sedentary', 
                       'family_history', 'previous_treatment']
        biomarkers = ['glucose', 'cholesterol', 'blood_pressure', 'bmi', 'inflammation']
        
        for i in range(self.num_patients):
            patient_id = i + 1
            age = random.randint(18, 85)
            gender = random.choice(['male', 'female', 'other'])
            
            # Generate medical history
            num_conditions = random.randint(0, 3)
            medical_history = random.sample(medical_conditions, num_conditions)
            
            # Generate biomarkers
            biomarkers_dict = {}
            for biomarker in biomarkers:
                if biomarker == 'glucose':
                    biomarkers_dict[biomarker] = random.uniform(70, 200)
                elif biomarker == 'cholesterol':
                    biomarkers_dict[biomarker] = random.uniform(120, 300)
                elif biomarker == 'blood_pressure':
                    biomarkers_dict[biomarker] = random.uniform(90, 180)
                elif biomarker == 'bmi':
                    biomarkers_dict[biomarker] = random.uniform(18, 40)
                else:
                    biomarkers_dict[biomarker] = random.uniform(0, 10)
            
            # Generate risk factors
            num_risk_factors = random.randint(0, 4)
            risk_factors_list = random.sample(risk_factors, num_risk_factors)
            
            patients[patient_id] = Patient(
                patient_id=patient_id,
                age=age,
                gender=gender,
                medical_history=medical_history,
                biomarkers=biomarkers_dict,
                risk_factors=risk_factors_list
            )
        
        return patients
    
    def _initialize_trial_arms(self) -> Dict[int, TrialArm]:
        """Initialize trial arms for each treatment."""
        trial_arms = {}
        
        for treatment_id, treatment in self.treatments.items():
            trial_arms[treatment_id] = TrialArm(
                arm_id=treatment_id,
                treatment=treatment
            )
        
        return trial_arms
    
    def calculate_patient_treatment_compatibility(self, 
                                               patient_id: int, 
                                               treatment_id: int) -> float:
        """
        Calculate compatibility between patient and treatment.
        
        Args:
            patient_id: Patient identifier
            treatment_id: Treatment identifier
            
        Returns:
            Compatibility score (higher is better)
        """
        patient = self.patients[patient_id]
        treatment = self.treatments[treatment_id]
        
        # Base compatibility from treatment mechanism
        base_compatibility = 0.5
        
        # Age-based adjustments
        if patient.age > 65:
            if treatment.mechanism in ['antibody', 'vaccine']:
                base_compatibility += 0.2  # Better for elderly
            elif treatment.mechanism == 'gene_therapy':
                base_compatibility -= 0.3  # Riskier for elderly
        
        # Medical history adjustments
        if 'diabetes' in patient.medical_history:
            if treatment.mechanism == 'inhibitor':
                base_compatibility += 0.1  # Good for diabetes
        if 'autoimmune' in patient.medical_history:
            if treatment.mechanism == 'antibody':
                base_compatibility -= 0.2  # Risk for autoimmune
        
        # Biomarker-based adjustments
        if 'inflammation' in patient.biomarkers:
            inflammation_level = patient.biomarkers['inflammation']
            if inflammation_level > 5:
                if treatment.mechanism == 'antibody':
                    base_compatibility += 0.3  # Good for high inflammation
        
        # Risk factor adjustments
        if 'obese' in patient.risk_factors:
            if treatment.mechanism == 'vaccine':
                base_compatibility -= 0.1  # Lower efficacy in obese
        
        return max(0.0, min(1.0, base_compatibility))
    
    def epsilon_greedy_allocation(self, patient_id: int, epsilon: float = 0.2) -> int:
        """
        Allocate patient to treatment using epsilon-greedy strategy.
        
        Args:
            patient_id: Patient identifier
            epsilon: Exploration rate
            
        Returns:
            Allocated treatment ID
        """
        available_treatments = list(self.treatments.keys())
        
        if not available_treatments:
            return None
        
        # Exploration: choose random treatment
        if random.random() < epsilon:
            return random.choice(available_treatments)
        
        # Exploitation: choose treatment with highest estimated efficacy
        best_treatment = available_treatments[0]
        best_score = -float('inf')
        
        for treatment_id in available_treatments:
            # Combine estimated efficacy with patient compatibility
            estimated_efficacy = self.treatment_estimates[patient_id][treatment_id]
            compatibility = self.calculate_patient_treatment_compatibility(patient_id, treatment_id)
            
            # Combined score
            score = estimated_efficacy + compatibility * 0.3
            
            if score > best_score:
                best_score = score
                best_treatment = treatment_id
        
        return best_treatment
    
    def ucb_allocation(self, patient_id: int, alpha: float = 2.0) -> int:
        """
        Allocate patient to treatment using Upper Confidence Bound (UCB).
        
        Args:
            patient_id: Patient identifier
            alpha: Exploration parameter
            
        Returns:
            Allocated treatment ID
        """
        available_treatments = list(self.treatments.keys())
        
        if not available_treatments:
            return None
        
        best_treatment = available_treatments[0]
        best_ucb = -float('inf')
        
        for treatment_id in available_treatments:
            # Get current estimate
            estimate = self.treatment_estimates[patient_id][treatment_id]
            
            # Get number of times this treatment has been allocated to similar patients
            count = self.treatment_counts[patient_id][treatment_id]
            
            # UCB formula
            if count == 0:
                ucb = float('inf')  # Prioritize unexplored treatments
            else:
                ucb = estimate + alpha * np.sqrt(np.log(len(self.trial_outcomes) + 1) / count)
            
            # Add compatibility score
            compatibility = self.calculate_patient_treatment_compatibility(patient_id, treatment_id)
            ucb += compatibility * 0.2
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_treatment = treatment_id
        
        return best_treatment
    
    def thompson_sampling_allocation(self, patient_id: int) -> int:
        """
        Allocate patient to treatment using Thompson sampling.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Allocated treatment ID
        """
        available_treatments = list(self.treatments.keys())
        
        if not available_treatments:
            return None
        
        best_treatment = available_treatments[0]
        best_sample = -float('inf')
        
        for treatment_id in available_treatments:
            # Get current estimate and uncertainty
            estimate = self.treatment_estimates[patient_id][treatment_id]
            count = self.treatment_counts[patient_id][treatment_id]
            
            # Sample from posterior (assuming Beta distribution)
            if count == 0:
                # Uniform prior for unexplored treatments
                sample = random.random()
            else:
                # Beta posterior based on successes and failures
                successes = sum(1 for outcome in self.trial_outcomes 
                              if outcome['treatment_id'] == treatment_id and outcome['success'])
                failures = count - successes
                
                # Sample from Beta distribution
                sample = np.random.beta(successes + 1, failures + 1)
            
            # Add compatibility score
            compatibility = self.calculate_patient_treatment_compatibility(patient_id, treatment_id)
            sample += compatibility * 0.1
            
            if sample > best_sample:
                best_sample = sample
                best_treatment = treatment_id
        
        return best_treatment
    
    def adaptive_randomization_allocation(self, patient_id: int) -> int:
        """
        Allocate patient using adaptive randomization.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Allocated treatment ID
        """
        available_treatments = list(self.treatments.keys())
        
        if not available_treatments:
            return None
        
        # Calculate allocation probabilities based on current performance
        total_patients = sum(arm.current_size for arm in self.trial_arms.values())
        
        if total_patients == 0:
            # Equal allocation for first patients
            return random.choice(available_treatments)
        
        # Calculate allocation probabilities
        probabilities = []
        for treatment_id in available_treatments:
            arm = self.trial_arms[treatment_id]
            
            if arm.current_size == 0:
                # Equal probability for arms with no patients
                probabilities.append(1.0)
            else:
                # Probability based on current success rate
                success_rate = arm.success_count / arm.current_size
                # Add small constant to avoid zero probability
                probabilities.append(success_rate + 0.1)
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]
        
        # Sample treatment based on probabilities
        return np.random.choice(available_treatments, p=probabilities)
    
    def simulate_patient_outcome(self, patient_id: int, treatment_id: int) -> Dict:
        """
        Simulate patient outcome for a given treatment.
        
        Args:
            patient_id: Patient identifier
            treatment_id: Treatment identifier
            
        Returns:
            Outcome dictionary
        """
        patient = self.patients[patient_id]
        treatment = self.treatments[treatment_id]
        
        # Base success probability from treatment efficacy
        base_success_prob = treatment.expected_efficacy
        
        # Patient-specific adjustments
        compatibility = self.calculate_patient_treatment_compatibility(patient_id, treatment_id)
        success_prob = base_success_prob * (0.8 + 0.4 * compatibility)
        
        # Age-based adjustments
        if patient.age > 65:
            success_prob *= 0.9  # Lower efficacy in elderly
        elif patient.age < 30:
            success_prob *= 1.1  # Higher efficacy in young
        
        # Medical history adjustments
        if 'diabetes' in patient.medical_history:
            success_prob *= 0.95
        if 'autoimmune' in patient.medical_history:
            success_prob *= 0.9
        
        # Simulate success/failure
        success = random.random() < success_prob
        
        # Simulate adverse events
        base_adverse_prob = 1 - treatment.expected_safety
        adverse_prob = base_adverse_prob * (1 + 0.2 * len(patient.risk_factors))
        adverse_event = random.random() < adverse_prob
        
        # Determine outcome severity
        if adverse_event:
            severity = random.choice(['mild', 'moderate', 'severe'])
        else:
            severity = 'none'
        
        return {
            'patient_id': patient_id,
            'treatment_id': treatment_id,
            'success': success,
            'adverse_event': adverse_event,
            'severity': severity,
            'day': self.current_day
        }
    
    def enroll_patient(self, patient_id: int, allocation_algorithm: str = 'adaptive') -> int:
        """
        Enroll a patient in the trial.
        
        Args:
            patient_id: Patient identifier
            allocation_algorithm: Algorithm for treatment allocation
            
        Returns:
            Allocated treatment ID
        """
        if patient_id in self.enrolled_patients:
            return None  # Patient already enrolled
        
        # Choose allocation algorithm
        if allocation_algorithm == 'epsilon_greedy':
            treatment_id = self.epsilon_greedy_allocation(patient_id)
        elif allocation_algorithm == 'ucb':
            treatment_id = self.ucb_allocation(patient_id)
        elif allocation_algorithm == 'thompson':
            treatment_id = self.thompson_sampling_allocation(patient_id)
        elif allocation_algorithm == 'adaptive':
            treatment_id = self.adaptive_randomization_allocation(patient_id)
        else:
            treatment_id = self.adaptive_randomization_allocation(patient_id)
        
        if treatment_id is None:
            return None
        
        # Enroll patient
        self.enrolled_patients.add(patient_id)
        self.patient_treatment_assignments[patient_id] = treatment_id
        
        # Add to trial arm
        arm = self.trial_arms[treatment_id]
        arm.patients.append(patient_id)
        arm.current_size += 1
        
        return treatment_id
    
    def simulate_patient_followup(self, patient_id: int) -> Dict:
        """
        Simulate patient follow-up and outcome assessment.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Follow-up outcome
        """
        if patient_id not in self.enrolled_patients:
            return None
        
        treatment_id = self.patient_treatment_assignments[patient_id]
        
        # Simulate outcome
        outcome = self.simulate_patient_outcome(patient_id, treatment_id)
        
        # Update trial arm
        arm = self.trial_arms[treatment_id]
        arm.outcomes.append(outcome)
        
        if outcome['success']:
            arm.success_count += 1
        
        if outcome['adverse_event']:
            arm.adverse_events += 1
        
        # Update bandit estimates
        self._update_treatment_estimates(patient_id, treatment_id, outcome)
        
        # Record outcome
        self.trial_outcomes.append(outcome)
        
        # Mark patient as completed
        self.completed_patients.add(patient_id)
        
        return outcome
    
    def _update_treatment_estimates(self, patient_id: int, treatment_id: int, outcome: Dict):
        """Update bandit estimates after patient outcome."""
        current_estimate = self.treatment_estimates[patient_id][treatment_id]
        current_count = self.treatment_counts[patient_id][treatment_id]
        
        # Incremental update
        new_count = current_count + 1
        reward = 1.0 if outcome['success'] else 0.0
        new_estimate = (current_estimate * current_count + reward) / new_count
        
        self.treatment_estimates[patient_id][treatment_id] = new_estimate
        self.treatment_counts[patient_id][treatment_id] = new_count
    
    def check_safety_monitoring(self) -> Dict:
        """
        Check safety monitoring criteria.
        
        Returns:
            Safety monitoring results
        """
        safety_results = {
            'arms_to_stop': [],
            'overall_safety_ok': True,
            'safety_alerts': []
        }
        
        for arm_id, arm in self.trial_arms.items():
            if arm.current_size < self.safety_thresholds['sample_size_min']:
                continue
            
            # Calculate adverse event rate
            adverse_rate = arm.adverse_events / arm.current_size
            
            # Check early stopping criteria
            if adverse_rate > self.safety_thresholds['early_stopping_threshold']:
                safety_results['arms_to_stop'].append(arm_id)
                safety_results['safety_alerts'].append(
                    f"Arm {arm_id} stopped due to high adverse event rate: {adverse_rate:.3f}"
                )
            
            # Check overall safety
            if adverse_rate > self.safety_thresholds['adverse_event_rate']:
                safety_results['overall_safety_ok'] = False
        
        return safety_results
    
    def check_efficacy_monitoring(self) -> Dict:
        """
        Check efficacy monitoring criteria.
        
        Returns:
            Efficacy monitoring results
        """
        efficacy_results = {
            'best_treatment': None,
            'efficacy_threshold_met': False,
            'efficacy_alerts': []
        }
        
        best_arm = None
        best_efficacy = 0.0
        
        for arm_id, arm in self.trial_arms.items():
            if arm.current_size < self.safety_thresholds['sample_size_min']:
                continue
            
            efficacy_rate = arm.success_count / arm.current_size
            
            if efficacy_rate > best_efficacy:
                best_efficacy = efficacy_rate
                best_arm = arm_id
            
            # Check if efficacy threshold is met
            if efficacy_rate > self.safety_thresholds['efficacy_threshold']:
                efficacy_results['efficacy_threshold_met'] = True
                efficacy_results['efficacy_alerts'].append(
                    f"Arm {arm_id} meets efficacy threshold: {efficacy_rate:.3f}"
                )
        
        efficacy_results['best_treatment'] = best_arm
        
        return efficacy_results
    
    def simulate_trial(self, 
                      allocation_algorithm: str = 'adaptive',
                      max_patients: int = 200) -> Dict:
        """
        Simulate a complete clinical trial.
        
        Args:
            allocation_algorithm: Algorithm for treatment allocation
            max_patients: Maximum number of patients to enroll
            
        Returns:
            Trial results dictionary
        """
        trial_results = {
            'algorithm': allocation_algorithm,
            'total_patients': 0,
            'completed_patients': 0,
            'trial_duration': 0,
            'arm_performance': {},
            'safety_events': [],
            'efficacy_results': [],
            'ethical_violations': []
        }
        
        # Enroll patients over time
        for day in range(self.max_trial_duration):
            self.current_day = day
            
            # Enroll new patients (simulate enrollment rate)
            if len(self.enrolled_patients) < max_patients and day % 7 == 0:  # Weekly enrollment
                new_patients = min(5, max_patients - len(self.enrolled_patients))
                
                for _ in range(new_patients):
                    # Select random patient not yet enrolled
                    available_patients = [pid for pid in range(1, self.num_patients + 1)
                                        if pid not in self.enrolled_patients]
                    
                    if available_patients:
                        patient_id = random.choice(available_patients)
                        treatment_id = self.enroll_patient(patient_id, allocation_algorithm)
                        
                        if treatment_id:
                            trial_results['total_patients'] += 1
            
            # Simulate follow-up for enrolled patients
            for patient_id in list(self.enrolled_patients):
                if patient_id not in self.completed_patients:
                    # Simulate completion after 30 days
                    if day - self.current_day >= 30:
                        outcome = self.simulate_patient_followup(patient_id)
                        if outcome:
                            trial_results['completed_patients'] += 1
            
            # Safety monitoring
            safety_results = self.check_safety_monitoring()
            if safety_results['safety_alerts']:
                trial_results['safety_events'].extend(safety_results['safety_alerts'])
            
            # Efficacy monitoring
            efficacy_results = self.check_efficacy_monitoring()
            if efficacy_results['efficacy_alerts']:
                trial_results['efficacy_results'].extend(efficacy_results['efficacy_alerts'])
            
            # Check for early stopping
            if not safety_results['overall_safety_ok']:
                trial_results['safety_events'].append("Trial stopped due to safety concerns")
                break
            
            if efficacy_results['efficacy_threshold_met']:
                trial_results['efficacy_results'].append("Trial stopped due to efficacy threshold met")
                break
        
        # Calculate final arm performance
        for arm_id, arm in self.trial_arms.items():
            if arm.current_size > 0:
                trial_results['arm_performance'][arm_id] = {
                    'patients': arm.current_size,
                    'success_rate': arm.success_count / arm.current_size,
                    'adverse_event_rate': arm.adverse_events / arm.current_size,
                    'treatment': arm.treatment.name
                }
        
        trial_results['trial_duration'] = self.current_day
        
        return trial_results
    
    def evaluate_allocation_algorithms(self, 
                                     num_trials: int = 3,
                                     patients_per_trial: int = 150) -> Dict:
        """
        Evaluate different allocation algorithms.
        
        Args:
            num_trials: Number of trials per algorithm
            patients_per_trial: Patients per trial
            
        Returns:
            Evaluation results dictionary
        """
        algorithms = ['epsilon_greedy', 'ucb', 'thompson', 'adaptive']
        results = {}
        
        for algorithm in algorithms:
            print(f"Evaluating {algorithm} algorithm...")
            
            algorithm_results = {
                'success_rates': [],
                'adverse_event_rates': [],
                'trial_durations': [],
                'patient_counts': []
            }
            
            for trial in range(num_trials):
                # Reset trial state
                self._reset_trial_state()
                
                # Run trial
                trial_result = self.simulate_trial(algorithm, patients_per_trial)
                
                # Calculate average success rate across arms
                success_rates = [arm['success_rate'] for arm in trial_result['arm_performance'].values()]
                adverse_rates = [arm['adverse_event_rate'] for arm in trial_result['arm_performance'].values()]
                
                if success_rates:
                    algorithm_results['success_rates'].append(np.mean(success_rates))
                if adverse_rates:
                    algorithm_results['adverse_event_rates'].append(np.mean(adverse_rates))
                
                algorithm_results['trial_durations'].append(trial_result['trial_duration'])
                algorithm_results['patient_counts'].append(trial_result['completed_patients'])
            
            results[algorithm] = algorithm_results
        
        return results
    
    def _reset_trial_state(self):
        """Reset trial state for fair comparison."""
        # Reset trial arms
        for arm in self.trial_arms.values():
            arm.patients = []
            arm.outcomes = []
            arm.current_size = 0
            arm.success_count = 0
            arm.adverse_events = 0
        
        # Reset tracking variables
        self.trial_outcomes = []
        self.safety_events = []
        self.efficacy_results = []
        self.ethical_violations = []
        self.enrolled_patients = set()
        self.completed_patients = set()
        self.patient_treatment_assignments = {}
        self.current_day = 0
    
    def plot_evaluation_results(self, results: Dict):
        """
        Plot evaluation results for comparison.
        
        Args:
            results: Evaluation results dictionary
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Clinical Trial Algorithm Evaluation', fontsize=16)
        
        algorithms = list(results.keys())
        
        # Success rates
        success_means = [np.mean(results[alg]['success_rates']) for alg in algorithms]
        success_stds = [np.std(results[alg]['success_rates']) for alg in algorithms]
        
        axes[0, 0].bar(algorithms, success_means, yerr=success_stds, capsize=5)
        axes[0, 0].set_title('Average Success Rate')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Adverse event rates
        adverse_means = [np.mean(results[alg]['adverse_event_rates']) for alg in algorithms]
        adverse_stds = [np.std(results[alg]['adverse_event_rates']) for alg in algorithms]
        
        axes[0, 1].bar(algorithms, adverse_means, yerr=adverse_stds, capsize=5)
        axes[0, 1].set_title('Average Adverse Event Rate')
        axes[0, 1].set_ylabel('Adverse Event Rate')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Trial durations
        duration_means = [np.mean(results[alg]['trial_durations']) for alg in algorithms]
        duration_stds = [np.std(results[alg]['trial_durations']) for alg in algorithms]
        
        axes[1, 0].bar(algorithms, duration_means, yerr=duration_stds, capsize=5)
        axes[1, 0].set_title('Average Trial Duration')
        axes[1, 0].set_ylabel('Duration (Days)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Patient counts
        patient_means = [np.mean(results[alg]['patient_counts']) for alg in algorithms]
        patient_stds = [np.std(results[alg]['patient_counts']) for alg in algorithms]
        
        axes[1, 1].bar(algorithms, patient_means, yerr=patient_stds, capsize=5)
        axes[1, 1].set_title('Average Completed Patients')
        axes[1, 1].set_ylabel('Number of Patients')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_ethical_considerations(self):
        """Demonstrate ethical considerations in clinical trials."""
        print("=== Ethical Considerations Demonstration ===")
        
        # Test different allocation strategies
        strategies = {
            'equal_allocation': 'Equal allocation to all arms',
            'adaptive_randomization': 'Adaptive randomization based on performance',
            'ucb_allocation': 'UCB-based allocation',
            'thompson_sampling': 'Thompson sampling allocation'
        }
        
        print("\nEthical Analysis of Allocation Strategies:")
        for strategy, description in strategies.items():
            print(f"\n{strategy.upper()}:")
            print(f"  - Description: {description}")
            
            if strategy == 'equal_allocation':
                print("  - Ethical Pros: Ensures equal access to all treatments")
                print("  - Ethical Cons: May expose patients to inferior treatments")
            elif strategy == 'adaptive_randomization':
                print("  - Ethical Pros: Balances learning with patient benefit")
                print("  - Ethical Cons: May favor treatments prematurely")
            elif strategy == 'ucb_allocation':
                print("  - Ethical Pros: Optimizes for best treatment discovery")
                print("  - Ethical Cons: May exploit promising treatments early")
            elif strategy == 'thompson_sampling':
                print("  - Ethical Pros: Bayesian approach with uncertainty quantification")
                print("  - Ethical Cons: May be less interpretable")
        
        # Demonstrate safety monitoring
        print("\nSafety Monitoring Demonstration:")
        print("  - Adverse event rate threshold: 15%")
        print("  - Early stopping threshold: 25%")
        print("  - Minimum sample size per arm: 20 patients")
        
        # Run safety demonstration
        self._reset_trial_state()
        trial_result = self.simulate_trial('adaptive', 100)
        
        print(f"\nSafety Monitoring Results:")
        print(f"  - Total patients: {trial_result['total_patients']}")
        print(f"  - Safety events: {len(trial_result['safety_events'])}")
        
        for arm_id, performance in trial_result['arm_performance'].items():
            print(f"  - Arm {arm_id}: {performance['adverse_event_rate']:.3f} adverse event rate")

def main():
    """Main demonstration function."""
    print("Clinical Trials Simulation using Multi-Armed Bandits")
    print("=" * 60)
    
    # Initialize clinical trial system
    trial_system = ClinicalTrialBandit(
        num_treatments=5,
        num_patients=300,
        feature_dim=10,
        max_trial_duration=180
    )
    
    print(f"Initialized with {trial_system.num_treatments} treatments and {trial_system.num_patients} patients")
    
    # Demonstrate ethical considerations
    trial_system.demonstrate_ethical_considerations()
    
    # Evaluate different algorithms
    print("\n=== Algorithm Evaluation ===")
    print("Running evaluation (this may take a moment)...")
    
    results = trial_system.evaluate_allocation_algorithms(
        num_trials=2,
        patients_per_trial=100
    )
    
    # Plot results
    trial_system.plot_evaluation_results(results)
    
    # Print summary statistics
    print("\n=== Evaluation Summary ===")
    for algorithm, result in results.items():
        avg_success = np.mean(result['success_rates'])
        avg_adverse = np.mean(result['adverse_event_rates'])
        avg_duration = np.mean(result['trial_durations'])
        avg_patients = np.mean(result['patient_counts'])
        
        print(f"{algorithm.upper()}:")
        print(f"  - Average Success Rate: {avg_success:.3f}")
        print(f"  - Average Adverse Event Rate: {avg_adverse:.3f}")
        print(f"  - Average Trial Duration: {avg_duration:.1f} days")
        print(f"  - Average Completed Patients: {avg_patients:.1f}")
    
    print("\n=== Key Insights ===")
    print("1. Adaptive randomization balances learning with patient benefit")
    print("2. Safety monitoring is crucial for ethical trial conduct")
    print("3. Patient characteristics significantly affect treatment outcomes")
    print("4. Early stopping criteria protect patient safety")
    print("5. Bayesian methods provide uncertainty quantification")

if __name__ == "__main__":
    main() 