import numpy as np
from contextual_thompson_sampling import ContextualThompsonSampling

class AdaptiveClinicalTrial:
    """
    Adaptive Clinical Trial using contextual bandits.
    
    This class implements a contextual bandit for adaptively allocating patients
    to treatment arms while learning effectiveness.
    """
    
    def __init__(self, n_treatments, patient_feature_dim):
        """
        Initialize Adaptive Clinical Trial.
        
        Args:
            n_treatments: Number of treatment arms
            patient_feature_dim: Dimension of patient feature vectors
        """
        self.bandit = ContextualThompsonSampling(patient_feature_dim)
        self.treatment_features = self._extract_treatment_features(n_treatments)
        
    def assign_treatment(self, patient_features):
        """
        Assign treatment based on patient characteristics.
        
        Args:
            patient_features: Patient feature vector
            
        Returns:
            int: Index of assigned treatment
        """
        contextual_features = self._create_contextual_features(patient_features)
        treatment_idx = self.bandit.select_arm(contextual_features)
        return treatment_idx
    
    def update(self, treatment_idx, patient_features, outcome):
        """
        Update with treatment outcome.
        
        Args:
            treatment_idx: Index of assigned treatment
            patient_features: Patient feature vector
            outcome: Treatment outcome (improvement, survival time)
        """
        contextual_features = self._create_contextual_features(patient_features)
        self.bandit.update(treatment_idx, outcome, contextual_features)
    
    def get_best_treatment(self, patient_features):
        """
        Get the best treatment for a patient.
        
        Args:
            patient_features: Patient feature vector
            
        Returns:
            int: Index of best treatment
        """
        contextual_features = self._create_contextual_features(patient_features)
        return self.bandit.select_arm(contextual_features)
    
    def _extract_treatment_features(self, n_treatments):
        """
        Extract features for each treatment.
        
        Args:
            n_treatments: Number of treatments
            
        Returns:
            np.ndarray: Array of treatment features
        """
        features = []
        for i in range(n_treatments):
            treatment_features = [
                i % 3,  # Treatment type
                np.random.uniform(0.1, 2.0),  # Dosage
                np.random.choice([0, 1]),  # Oral/IV
                np.random.uniform(0, 24),  # Frequency
                np.random.choice([0, 1])   # Experimental/Control
            ]
            features.append(treatment_features)
        return np.array(features)
    
    def _create_contextual_features(self, patient_features):
        """
        Create contextual features for all treatments.
        
        Args:
            patient_features: Patient feature vector
            
        Returns:
            list: List of contextual feature vectors
        """
        contextual_features = []
        for treatment_feat in self.treatment_features:
            # Combine patient features with treatment features
            combined = np.concatenate([patient_features, treatment_feat])
            contextual_features.append(combined)
        return contextual_features

class DrugDiscoveryBandit:
    """
    Drug Discovery Bandit for screening drug candidates.
    
    This class implements a linear contextual bandit for efficiently screening
    multiple drug candidates.
    """
    
    def __init__(self, n_compounds, target_feature_dim):
        """
        Initialize Drug Discovery Bandit.
        
        Args:
            n_compounds: Number of drug compounds
            target_feature_dim: Dimension of target feature vectors
        """
        from linucb import LinUCB
        self.bandit = LinUCB(target_feature_dim)
        self.compound_features = self._extract_compound_features(n_compounds)
        
    def select_compound(self, target_features):
        """
        Select compound to test.
        
        Args:
            target_features: Target feature vector
            
        Returns:
            int: Index of selected compound
        """
        contextual_features = self._create_contextual_features(target_features)
        compound_idx = self.bandit.select_arm(contextual_features)
        return compound_idx
    
    def update(self, compound_idx, target_features, efficacy):
        """
        Update with efficacy results.
        
        Args:
            compound_idx: Index of tested compound
            target_features: Target feature vector
            efficacy: Efficacy score (0-1)
        """
        contextual_features = self._create_contextual_features(target_features)
        self.bandit.update(compound_idx, efficacy, contextual_features)
    
    def _extract_compound_features(self, n_compounds):
        """
        Extract features for each compound.
        
        Args:
            n_compounds: Number of compounds
            
        Returns:
            np.ndarray: Array of compound features
        """
        features = []
        for i in range(n_compounds):
            compound_features = [
                i % 10,  # Chemical class
                np.random.uniform(100, 1000),  # Molecular weight
                np.random.uniform(0, 10),  # LogP
                np.random.uniform(0, 20),  # H-bond donors
                np.random.uniform(0, 20),  # H-bond acceptors
                np.random.uniform(0, 10)   # Rotatable bonds
            ]
            features.append(compound_features)
        return np.array(features)
    
    def _create_contextual_features(self, target_features):
        """
        Create contextual features for all compounds.
        
        Args:
            target_features: Target feature vector
            
        Returns:
            list: List of contextual feature vectors
        """
        contextual_features = []
        for compound_feat in self.compound_features:
            # Combine target features with compound features
            combined = np.concatenate([target_features, compound_feat])
            contextual_features.append(combined)
        return contextual_features
