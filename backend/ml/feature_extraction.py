"""
Feature Extraction for Bitcoin Fraud Detection ML Model
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import json
import logging
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class BitcoinFeatureExtractor:
    """
    Extract features from Bitcoin address analysis for ML model training
    """
    
    def __init__(self):
        self.feature_names = []
        self._initialize_feature_mapping()
    
    def _initialize_feature_mapping(self):
        """Initialize the mapping of feature names"""
        self.feature_names = [
            # Basic transaction metrics
            'transaction_count',
            'total_received_btc',
            'total_sent_btc',
            'current_balance_btc',
            'average_transaction_value',
            'median_transaction_value',
            'transaction_value_std',
            'min_transaction_value',
            'max_transaction_value',
            
            # Activity patterns
            'activity_span_days',
            'transactions_per_day',
            'velocity_btc_per_tx',
            'turnover_ratio',
            'retention_ratio',
            
            # Network topology features
            'unique_input_addresses',
            'unique_output_addresses',
            'input_output_ratio',
            'degree_centrality',
            'betweenness_centrality',
            'closeness_centrality',
            'eigenvector_centrality',
            'clustering_coefficient',
            
            # Temporal patterns
            'average_interval_hours',
            'median_interval_hours',
            'interval_std_hours',
            'burst_count',
            'max_burst_size',
            'regularity_score',
            'coefficient_of_variation',
            
            # Flow concentration
            'input_gini_coefficient',
            'output_gini_coefficient',
            'top_input_concentration',
            'top_output_concentration',
            
            # Suspicious behavior indicators
            'rapid_movement_count',
            'round_amounts_ratio',
            'mixing_suspicious_score',
            'high_fan_out_ratio',
            'cluster_size',
            'community_modularity',
            
            # Advanced features
            'transaction_density',
            'value_flow_variance',
            'address_reuse_frequency',
            'output_script_diversity',
            'time_concentration_score',
            
            # Enhanced network features (NEW)
            'pagerank_score',
            'hits_hub_score',
            'hits_authority_score',
            'local_clustering_coefficient',
            'neighbor_diversity_score',
            
            # Enhanced temporal features (NEW)
            'transaction_sequence_entropy',
            'temporal_motif_count',
            'periodic_transaction_score',
            'time_of_day_variance',
            'weekend_activity_ratio',
            
            # Enhanced behavioral features (NEW)
            'peel_chain_likelihood',
            'self_loop_ratio',
            'dormancy_periods_count',
            'reactivation_pattern_score',
            'transaction_size_entropy'
        ]
    
    def extract_features_from_analysis(self, analysis_data: Dict) -> np.ndarray:
        """
        Extract features from blockchain analysis data
        """
        try:
            features = np.zeros(len(self.feature_names))
            
            # Extract basic metrics
            basic_metrics = analysis_data.get('basic_metrics', {})
            features[0] = basic_metrics.get('transaction_count', 0)
            features[1] = basic_metrics.get('total_received_btc', 0)
            features[2] = basic_metrics.get('total_sent_btc', 0)
            features[3] = basic_metrics.get('balance_btc', 0)
            
            # Extract enhanced network features
            network_data = analysis_data.get('network_analysis', {})
            if network_data:
                self._extract_enhanced_network_features(features, network_data)
            
            # Extract enhanced temporal features
            temporal_data = analysis_data.get('temporal_analysis', {})
            if temporal_data:
                self._extract_enhanced_temporal_features(features, temporal_data)
            
            # Extract enhanced behavioral features
            behavioral_data = analysis_data.get('transaction_patterns', {})
            if behavioral_data:
                self._extract_enhanced_behavioral_features(features, behavioral_data)
            
            # Calculate derived metrics
            tx_count = max(features[0], 1)
            total_volume = features[1] + features[2]
            
            features[4] = total_volume / tx_count  # average_transaction_value
            features[8] = min(features[1], features[2]) if features[1] > 0 and features[2] > 0 else 0  # min_transaction_value
            features[9] = max(features[1], features[2])  # max_transaction_value
            
            # Activity patterns
            activity_patterns = analysis_data.get('basic_metrics', {})
            if 'activity_patterns' in analysis_data:
                patterns = analysis_data['activity_patterns']
                features[11] = patterns.get('velocity', 0)
                features[12] = patterns.get('turnover_ratio', 0)
                features[13] = patterns.get('retention_ratio', 0)
            
            # Transaction patterns
            transaction_patterns = analysis_data.get('transaction_patterns', {})
            
            # Flow concentration
            flow_concentration = transaction_patterns.get('flow_concentration', {})
            features[14] = flow_concentration.get('unique_input_addresses', 0)
            features[15] = flow_concentration.get('unique_output_addresses', 0)
            features[16] = features[15] / max(features[14], 1)  # input_output_ratio
            features[26] = flow_concentration.get('input_gini', 0)
            features[27] = flow_concentration.get('output_gini', 0)
            features[28] = flow_concentration.get('top_input_concentration', 0)
            features[29] = flow_concentration.get('top_output_concentration', 0)
            
            # Network analysis
            network_analysis = analysis_data.get('network_analysis', {})
            centrality_measures = network_analysis.get('centrality_measures', {})
            features[17] = centrality_measures.get('degree_centrality', 0)
            features[18] = centrality_measures.get('betweenness_centrality', 0)
            features[19] = centrality_measures.get('closeness_centrality', 0)
            features[20] = centrality_measures.get('eigenvector_centrality', 0)
            features[21] = network_analysis.get('clustering_coefficient', 0)
            
            # Temporal analysis
            temporal_analysis = analysis_data.get('temporal_analysis', {})
            
            frequency_data = temporal_analysis.get('transaction_frequency', {})
            features[10] = frequency_data.get('time_span_days', 0)  # activity_span_days
            features[22] = frequency_data.get('average_interval_hours', 0)
            features[23] = frequency_data.get('median_interval_hours', 0)
            features[24] = frequency_data.get('std_interval_hours', 0)
            
            if features[10] > 0:
                features[25] = features[0] / features[10]  # transactions_per_day
            
            burst_detection = temporal_analysis.get('burst_detection', {})
            features[26] = burst_detection.get('burst_count', 0)
            features[27] = burst_detection.get('max_burst_size', 0)
            
            regularity = temporal_analysis.get('regularity_analysis', {})
            features[28] = regularity.get('regularity_score', 0)
            features[29] = regularity.get('coefficient_of_variation', 0)
            
            # Rapid movements
            rapid_movements = transaction_patterns.get('rapid_movement_count', 0)
            features[30] = rapid_movements
            
            # Amount statistics
            amount_stats = transaction_patterns.get('amount_statistics', {})
            features[5] = amount_stats.get('median_amount', 0)  # median_transaction_value
            features[6] = amount_stats.get('std_amount', 0)  # transaction_value_std
            features[31] = amount_stats.get('round_amounts', 0) / max(tx_count, 1)  # round_amounts_ratio
            
            # Clustering analysis
            clustering_analysis = analysis_data.get('clustering_analysis', {})
            features[34] = clustering_analysis.get('cluster_size', 1)
            
            community_detection = network_analysis.get('community_detection', {})
            features[35] = community_detection.get('modularity', 0)
            
            # Advanced features
            features[36] = features[0] / max(features[10], 1)  # transaction_density
            features[37] = features[6] / max(features[4], 1)  # value_flow_variance
            
            # High fan-out ratio
            if features[15] > 0:
                features[33] = min(features[15] / max(features[0], 1), 1.0)  # high_fan_out_ratio
            
            # Mixing suspicious score (placeholder - would need more sophisticated calculation)
            fraud_signals = analysis_data.get('fraud_signals', {})
            features[32] = fraud_signals.get('overall_fraud_score', 0)
            
            # Address reuse and script diversity (approximate)
            features[38] = 1.0 / max(features[14] + features[15], 1)  # address_reuse_frequency
            features[39] = min(features[15] / max(features[0], 1), 1.0)  # output_script_diversity
            
            # Time concentration score
            if features[24] > 0 and features[22] > 0:
                features[40] = features[24] / features[22]  # time_concentration_score
            
            # Ensure no NaN or infinite values
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return np.zeros(len(self.feature_names))
    
    def extract_features_batch(self, analysis_data_list: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from multiple address analyses
        """
        features_matrix = []
        addresses = []
        
        for analysis_data in analysis_data_list:
            if 'error' not in analysis_data:
                features = self.extract_features_from_analysis(analysis_data)
                features_matrix.append(features)
                addresses.append(analysis_data.get('address', 'unknown'))
        
        return np.array(features_matrix), addresses
    
    def create_feature_dataframe(self, analysis_data_list: List[Dict]) -> pd.DataFrame:
        """
        Create a pandas DataFrame with extracted features
        """
        features_matrix, addresses = self.extract_features_batch(analysis_data_list)
        
        df = pd.DataFrame(features_matrix, columns=self.feature_names)
        df['address'] = addresses
        
        return df
    
    def get_feature_importance_mapping(self) -> Dict[str, str]:
        """
        Get human-readable descriptions for each feature
        """
        return {
            'transaction_count': 'Total number of transactions',
            'total_received_btc': 'Total BTC received',
            'total_sent_btc': 'Total BTC sent',
            'current_balance_btc': 'Current balance in BTC',
            'average_transaction_value': 'Average transaction value',
            'median_transaction_value': 'Median transaction value',
            'transaction_value_std': 'Standard deviation of transaction values',
            'min_transaction_value': 'Minimum transaction value',
            'max_transaction_value': 'Maximum transaction value',
            'activity_span_days': 'Days between first and last transaction',
            'transactions_per_day': 'Average transactions per day',
            'velocity_btc_per_tx': 'BTC velocity per transaction',
            'turnover_ratio': 'Ratio of sent to received',
            'retention_ratio': 'Ratio of balance to received',
            'unique_input_addresses': 'Number of unique input addresses',
            'unique_output_addresses': 'Number of unique output addresses',
            'input_output_ratio': 'Ratio of output to input addresses',
            'degree_centrality': 'Network degree centrality',
            'betweenness_centrality': 'Network betweenness centrality',
            'closeness_centrality': 'Network closeness centrality',
            'eigenvector_centrality': 'Network eigenvector centrality',
            'clustering_coefficient': 'Network clustering coefficient',
            'average_interval_hours': 'Average time between transactions (hours)',
            'median_interval_hours': 'Median time between transactions (hours)',
            'interval_std_hours': 'Standard deviation of transaction intervals',
            'burst_count': 'Number of transaction bursts',
            'max_burst_size': 'Maximum burst size',
            'regularity_score': 'Transaction timing regularity score',
            'coefficient_of_variation': 'Coefficient of variation for intervals',
            'input_gini_coefficient': 'Gini coefficient for input distribution',
            'output_gini_coefficient': 'Gini coefficient for output distribution',
            'top_input_concentration': 'Concentration in top input addresses',
            'top_output_concentration': 'Concentration in top output addresses',
            'rapid_movement_count': 'Number of rapid fund movements',
            'round_amounts_ratio': 'Ratio of round amount transactions',
            'mixing_suspicious_score': 'Mixing service suspicion score',
            'high_fan_out_ratio': 'High fan-out pattern ratio',
            'cluster_size': 'Address cluster size',
            'community_modularity': 'Community detection modularity',
            'transaction_density': 'Transaction density over time',
            'value_flow_variance': 'Variance in value flows',
            'address_reuse_frequency': 'Address reuse frequency',
            'output_script_diversity': 'Diversity of output scripts',
            'time_concentration_score': 'Time concentration score'
        }
    
    def validate_features(self, features: np.ndarray) -> bool:
        """
        Validate that features are in expected ranges
        """
        if len(features) != len(self.feature_names):
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            return False
        
        # Check for negative values where they shouldn't exist
        non_negative_indices = [0, 1, 2, 3, 4, 5, 8, 9, 10, 14, 15, 25, 26, 27, 34]
        if np.any(features[non_negative_indices] < 0):
            return False
        
        # Check for probability/ratio values in [0, 1]
        probability_indices = [12, 13, 16, 17, 18, 19, 20, 21, 28, 31, 32, 33, 38, 39]
        if np.any(features[probability_indices] < 0) or np.any(features[probability_indices] > 1):
            return False
        
        return True
    
    def _extract_enhanced_network_features(self, features: np.ndarray, network_data: Dict):
        """
        Extract enhanced network features for improved fraud detection
        """
        # PageRank score - measure of address importance in the transaction network
        features[41] = network_data.get('pagerank_score', 0.0)
        
        # HITS algorithm scores
        features[42] = network_data.get('hits_hub_score', 0.0)
        features[43] = network_data.get('hits_authority_score', 0.0)
        
        # Local clustering coefficient - measures how close neighbors are to forming a clique
        features[44] = network_data.get('local_clustering_coefficient', 0.0)
        
        # Neighbor diversity - measures diversity of transaction partners
        features[45] = network_data.get('neighbor_diversity_score', 0.0)
    
    def _extract_enhanced_temporal_features(self, features: np.ndarray, temporal_data: Dict):
        """
        Extract enhanced temporal features for improved fraud detection
        """
        # Transaction sequence entropy - measures randomness in transaction timing
        features[46] = temporal_data.get('transaction_sequence_entropy', 0.0)
        
        # Temporal motif count - recurring patterns in transaction timing
        features[47] = temporal_data.get('temporal_motif_count', 0.0)
        
        # Periodic transaction score - detects regular transaction patterns
        features[48] = temporal_data.get('periodic_transaction_score', 0.0)
        
        # Time of day variance - variance in transaction times
        features[49] = temporal_data.get('time_of_day_variance', 0.0)
        
        # Weekend activity ratio - ratio of weekend to weekday activity
        features[50] = temporal_data.get('weekend_activity_ratio', 0.0)
    
    def _extract_enhanced_behavioral_features(self, features: np.ndarray, behavioral_data: Dict):
        """
        Extract enhanced behavioral features for improved fraud detection
        """
        # Peel chain likelihood - probability this address is part of a peel chain
        features[51] = behavioral_data.get('peel_chain_likelihood', 0.0)
        
        # Self-loop ratio - transactions sent back to the same address
        features[52] = behavioral_data.get('self_loop_ratio', 0.0)
        
        # Dormancy periods count - number of inactive periods
        features[53] = behavioral_data.get('dormancy_periods_count', 0.0)
        
        # Reactivation pattern score - suspicious reactivation after dormancy
        features[54] = behavioral_data.get('reactivation_pattern_score', 0.0)
        
        # Transaction size entropy - randomness in transaction sizes
        features[55] = behavioral_data.get('transaction_size_entropy', 0.0)
    
    def normalize_features(self, features: np.ndarray, feature_stats: Optional[Dict] = None) -> np.ndarray:
        """
        Normalize features for ML model training
        """
        if feature_stats is None:
            # Use default normalization
            normalized = np.copy(features)
            
            # Log transform for highly skewed features
            log_transform_indices = [0, 1, 2, 3, 4, 5, 8, 9, 10, 14, 15, 25, 34]
            for idx in log_transform_indices:
                if features[idx] > 0:
                    normalized[idx] = np.log1p(features[idx])
            
            # Apply min-max scaling to certain features
            minmax_indices = [22, 23, 24, 36, 37, 40]
            for idx in minmax_indices:
                if features[idx] > 0:
                    normalized[idx] = min(features[idx] / 100.0, 1.0)  # Cap at reasonable values
            
            return normalized
        else:
            # Use provided statistics for normalization
            means = feature_stats.get('means', np.zeros(len(features)))
            stds = feature_stats.get('stds', np.ones(len(features)))
            
            return (features - means) / np.maximum(stds, 1e-8)

# Testing and example usage
def test_feature_extraction():
    """Test feature extraction with sample data"""
    extractor = BitcoinFeatureExtractor()
    
    # Sample analysis data
    sample_analysis = {
        'address': 'test_address',
        'basic_metrics': {
            'transaction_count': 100,
            'total_received_btc': 10.5,
            'total_sent_btc': 9.5,
            'balance_btc': 1.0
        },
        'transaction_patterns': {
            'flow_concentration': {
                'unique_input_addresses': 50,
                'unique_output_addresses': 80,
                'input_gini': 0.6,
                'output_gini': 0.7
            },
            'amount_statistics': {
                'median_amount': 0.1,
                'std_amount': 0.05,
                'round_amounts': 10
            },
            'rapid_movement_count': 5
        },
        'network_analysis': {
            'centrality_measures': {
                'degree_centrality': 0.1,
                'betweenness_centrality': 0.05,
                'closeness_centrality': 0.2,
                'eigenvector_centrality': 0.15
            },
            'clustering_coefficient': 0.3
        },
        'temporal_analysis': {
            'transaction_frequency': {
                'time_span_days': 365,
                'average_interval_hours': 72,
                'median_interval_hours': 48,
                'std_interval_hours': 24
            },
            'burst_detection': {
                'burst_count': 3,
                'max_burst_size': 5
            },
            'regularity_analysis': {
                'regularity_score': 0.8,
                'coefficient_of_variation': 0.5
            }
        },
        'clustering_analysis': {
            'cluster_size': 15
        },
        'fraud_signals': {
            'overall_fraud_score': 0.3
        }
    }
    
    features = extractor.extract_features_from_analysis(sample_analysis)
    print(f"Extracted {len(features)} features")
    print(f"Feature validation: {extractor.validate_features(features)}")
    
    # Create DataFrame
    df = extractor.create_feature_dataframe([sample_analysis])
    print(f"DataFrame shape: {df.shape}")
    print(f"Sample features:\n{df.head()}")

if __name__ == "__main__":
    test_feature_extraction()