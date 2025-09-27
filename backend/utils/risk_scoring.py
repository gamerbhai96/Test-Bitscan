"""
Advanced Risk Scoring Engine for Bitcoin Address Assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level enumeration"""
    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class RiskFactor:
    """Individual risk factor"""
    name: str
    value: float
    weight: float
    description: str
    evidence: List[str]

@dataclass
class RiskScore:
    """Comprehensive risk score result"""
    address: str
    overall_score: float
    risk_level: RiskLevel
    confidence: float
    risk_factors: List[RiskFactor]
    timestamp: str
    explanation: str

class RiskScoringEngine:
    """
    Advanced risk scoring engine that combines multiple risk factors
    """
    
    def __init__(self):
        self.weights = self._initialize_weights()
        self.thresholds = self._initialize_thresholds()
        
    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize risk factor weights"""
        return {
            # Blockchain behavior weights
            'transaction_velocity': 0.15,
            'mixing_indicators': 0.20,
            'cluster_analysis': 0.12,
            'temporal_patterns': 0.10,
            'network_centrality': 0.15,
            
            # ML model weights
            'ml_prediction': 0.25,
            
            # External data weights
            'known_scam_database': 0.35,
            'user_reports': 0.15,
            'exchange_blacklist': 0.20,
            
            # Behavioral pattern weights
            'round_amounts': 0.08,
            'rapid_movements': 0.12,
            'high_fan_out': 0.10,
            'low_retention': 0.08,
            
            # Network topology weights
            'betweenness_centrality': 0.12,
            'community_isolation': 0.08,
            'graph_density': 0.06
        }
    
    def _initialize_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize risk level thresholds"""
        return {
            'overall': {
                'minimal': 0.0,
                'low': 0.2,
                'medium': 0.4,
                'high': 0.7,
                'critical': 0.9
            },
            'ml_confidence': {
                'minimum': 0.6  # Minimum ML confidence to trust prediction
            },
            'known_scam': {
                'definitive': 1.0  # If in known scam database
            }
        }
    
    def calculate_comprehensive_risk_score(
        self,
        blockchain_analysis: Dict,
        ml_prediction: Dict,
        enrichment_data: Dict
    ) -> RiskScore:
        """
        Calculate comprehensive risk score from all data sources
        """
        risk_factors = []
        
        # Extract individual risk factors
        risk_factors.extend(self._analyze_blockchain_risk_factors(blockchain_analysis))
        risk_factors.extend(self._analyze_ml_risk_factors(ml_prediction))
        risk_factors.extend(self._analyze_enrichment_risk_factors(enrichment_data))
        
        # Calculate weighted risk score
        overall_score = self._calculate_weighted_score(risk_factors)
        
        # Determine risk level
        risk_level = self._determine_risk_level(overall_score)
        
        # Calculate confidence
        confidence = self._calculate_confidence(risk_factors, ml_prediction)
        
        # Generate explanation
        explanation = self._generate_explanation(risk_factors, overall_score, risk_level)
        
        return RiskScore(
            address=blockchain_analysis.get('address', 'unknown'),
            overall_score=overall_score,
            risk_level=risk_level,
            confidence=confidence,
            risk_factors=risk_factors,
            timestamp=datetime.now().isoformat(),
            explanation=explanation
        )
    
    def _analyze_blockchain_risk_factors(self, analysis: Dict) -> List[RiskFactor]:
        """Extract risk factors from blockchain analysis"""
        factors = []
        
        # Transaction velocity risk
        basic_metrics = analysis.get('basic_metrics', {})
        tx_count = basic_metrics.get('transaction_count', 0)
        activity_span = analysis.get('temporal_analysis', {}).get('transaction_frequency', {}).get('time_span_days', 1)
        
        if activity_span > 0:
            velocity = tx_count / activity_span
            velocity_risk = min(velocity / 10.0, 1.0)  # Normalize high velocity
            
            factors.append(RiskFactor(
                name='transaction_velocity',
                value=velocity_risk,
                weight=self.weights['transaction_velocity'],
                description=f'High transaction velocity: {velocity:.2f} transactions/day',
                evidence=[f'{tx_count} transactions over {activity_span} days']
            ))
        
        # Mixing indicators
        fraud_signals = analysis.get('fraud_signals', {})
        mixing_score = fraud_signals.get('mixing_service_usage', 0) or 0
        if isinstance(mixing_score, bool):
            mixing_score = 1.0 if mixing_score else 0.0
        
        factors.append(RiskFactor(
            name='mixing_indicators',
            value=float(mixing_score),
            weight=self.weights['mixing_indicators'],
            description='Potential mixing service usage detected',
            evidence=['Multiple small outputs', 'Round number amounts', 'High input/output ratio']
        ))
        
        # Cluster analysis
        clustering = analysis.get('clustering_analysis', {})
        cluster_size = clustering.get('cluster_size', 1)
        cluster_risk = min((cluster_size - 1) / 50.0, 1.0)  # Risk increases with cluster size
        
        factors.append(RiskFactor(
            name='cluster_analysis',
            value=cluster_risk,
            weight=self.weights['cluster_analysis'],
            description=f'Large address cluster detected: {cluster_size} addresses',
            evidence=[f'Cluster contains {cluster_size} related addresses']
        ))
        
        # Temporal patterns
        temporal = analysis.get('temporal_analysis', {})
        burst_count = temporal.get('burst_detection', {}).get('burst_count', 0)
        burst_risk = min(burst_count / 5.0, 1.0)  # Risk increases with burst activity
        
        factors.append(RiskFactor(
            name='temporal_patterns',
            value=burst_risk,
            weight=self.weights['temporal_patterns'],
            description=f'Burst activity detected: {burst_count} bursts',
            evidence=[f'{burst_count} periods of rapid transaction activity']
        ))
        
        # Network centrality
        network = analysis.get('network_analysis', {})
        centrality = network.get('centrality_measures', {})
        betweenness = centrality.get('betweenness_centrality', 0)
        
        factors.append(RiskFactor(
            name='betweenness_centrality',
            value=float(betweenness),
            weight=self.weights['betweenness_centrality'],
            description=f'High network centrality: {betweenness:.3f}',
            evidence=[f'Betweenness centrality of {betweenness:.3f}']
        ))
        
        # Rapid movements
        tx_patterns = analysis.get('transaction_patterns', {})
        rapid_movements = tx_patterns.get('rapid_movement_count', 0)
        rapid_risk = min(rapid_movements / 10.0, 1.0)
        
        factors.append(RiskFactor(
            name='rapid_movements',
            value=rapid_risk,
            weight=self.weights['rapid_movements'],
            description=f'Rapid fund movements: {rapid_movements} instances',
            evidence=[f'{rapid_movements} transactions within 10-minute windows']
        ))
        
        return factors
    
    def _analyze_ml_risk_factors(self, ml_prediction: Dict) -> List[RiskFactor]:
        """Extract risk factors from ML prediction"""
        factors = []
        
        fraud_probability = ml_prediction.get('fraud_probability', 0.5)
        ml_confidence = ml_prediction.get('confidence', 0.0)
        
        # Only trust ML prediction if confidence is high enough
        if ml_confidence >= self.thresholds['ml_confidence']['minimum']:
            factors.append(RiskFactor(
                name='ml_prediction',
                value=float(fraud_probability),
                weight=self.weights['ml_prediction'],
                description=f'ML fraud probability: {fraud_probability:.2f}',
                evidence=[
                    f'Model confidence: {ml_confidence:.2f}',
                    f'Model used: {ml_prediction.get("model_used", "unknown")}'
                ]
            ))
        
        return factors
    
    def _analyze_enrichment_risk_factors(self, enrichment: Dict) -> List[RiskFactor]:
        """Extract risk factors from enrichment data"""
        factors = []
        
        # Known scam database check
        scam_check = enrichment.get('scam_database_check', {})
        if scam_check.get('is_scam', False):
            factors.append(RiskFactor(
                name='known_scam_database',
                value=1.0,
                weight=self.weights['known_scam_database'],
                description='Address found in known scam database',
                evidence=[
                    f'Scam type: {scam_check.get("scam_type", "unknown")}',
                    f'Source: {scam_check.get("source", "unknown")}'
                ]
            ))
        
        # User reports
        user_reports = enrichment.get('user_reports', [])
        if user_reports:
            report_risk = min(len(user_reports) / 5.0, 1.0)
            factors.append(RiskFactor(
                name='user_reports',
                value=report_risk,
                weight=self.weights['user_reports'],
                description=f'Multiple user reports: {len(user_reports)} reports',
                evidence=[f'{len(user_reports)} user reports filed']
            ))
        
        # External sources
        external = enrichment.get('external_sources', {})
        for source, data in external.items():
            if data.get('is_suspicious', False):
                factors.append(RiskFactor(
                    name=f'external_{source}',
                    value=0.8,
                    weight=0.1,  # Lower weight for external sources
                    description=f'Flagged by external source: {source}',
                    evidence=[f'Reports from {source}']
                ))
        
        return factors
    
    def _calculate_weighted_score(self, risk_factors: List[RiskFactor]) -> float:
        """Calculate weighted risk score"""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for factor in risk_factors:
            weighted_contribution = factor.value * factor.weight
            total_weighted_score += weighted_contribution
            total_weight += factor.weight
        
        # Normalize by total weight to get score between 0 and 1
        if total_weight > 0:
            normalized_score = total_weighted_score / total_weight
        else:
            normalized_score = 0.0
        
        return min(max(normalized_score, 0.0), 1.0)
    
    def _determine_risk_level(self, score: float) -> RiskLevel:
        """Determine risk level based on score"""
        thresholds = self.thresholds['overall']
        
        if score >= thresholds['critical']:
            return RiskLevel.CRITICAL
        elif score >= thresholds['high']:
            return RiskLevel.HIGH
        elif score >= thresholds['medium']:
            return RiskLevel.MEDIUM
        elif score >= thresholds['low']:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL
    
    def _calculate_confidence(self, risk_factors: List[RiskFactor], ml_prediction: Dict) -> float:
        """Calculate confidence in the risk assessment"""
        confidence_factors = []
        
        # ML model confidence
        ml_confidence = ml_prediction.get('confidence', 0.0)
        confidence_factors.append(ml_confidence * 0.4)
        
        # Number of risk factors (more factors = higher confidence)
        factor_confidence = min(len(risk_factors) / 10.0, 1.0)
        confidence_factors.append(factor_confidence * 0.3)
        
        # Presence of definitive indicators (known scam database)
        has_definitive = any(f.name == 'known_scam_database' for f in risk_factors)
        if has_definitive:
            confidence_factors.append(0.3)
        
        # Weight distribution (more balanced weights = higher confidence)
        if risk_factors:
            weights = [f.weight for f in risk_factors]
            weight_variance = np.var(weights)
            weight_confidence = max(0, 1.0 - weight_variance)
            confidence_factors.append(weight_confidence * 0.2)
        
        return min(sum(confidence_factors), 1.0)
    
    def _generate_explanation(self, risk_factors: List[RiskFactor], score: float, risk_level: RiskLevel) -> str:
        """Generate human-readable explanation"""
        explanation_parts = []
        
        # Overall assessment
        explanation_parts.append(f"Overall Risk Assessment: {risk_level.value} ({score:.2f}/1.00)")
        
        # Top risk factors
        sorted_factors = sorted(risk_factors, key=lambda x: x.value * x.weight, reverse=True)
        top_factors = sorted_factors[:3]
        
        if top_factors:
            explanation_parts.append("\nPrimary Risk Factors:")
            for i, factor in enumerate(top_factors, 1):
                contribution = factor.value * factor.weight
                explanation_parts.append(f"{i}. {factor.description} (Impact: {contribution:.3f})")
        
        # Risk mitigation or escalation advice
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            explanation_parts.append("\n⚠️ HIGH RISK: Strongly recommend avoiding transactions with this address.")
        elif risk_level == RiskLevel.MEDIUM:
            explanation_parts.append("\n⚡ MEDIUM RISK: Exercise caution and conduct additional due diligence.")
        elif risk_level == RiskLevel.LOW:
            explanation_parts.append("\n✓ LOW RISK: Address appears relatively safe but continue monitoring.")
        else:
            explanation_parts.append("\n✅ MINIMAL RISK: Address shows no significant red flags.")
        
        return "\n".join(explanation_parts)
    
    def batch_score_addresses(
        self,
        address_data: List[Tuple[Dict, Dict, Dict]]
    ) -> List[RiskScore]:
        """
        Score multiple addresses in batch
        Each tuple contains: (blockchain_analysis, ml_prediction, enrichment_data)
        """
        scores = []
        
        for blockchain_analysis, ml_prediction, enrichment_data in address_data:
            try:
                score = self.calculate_comprehensive_risk_score(
                    blockchain_analysis, ml_prediction, enrichment_data
                )
                scores.append(score)
            except Exception as e:
                logger.error(f"Error scoring address {blockchain_analysis.get('address', 'unknown')}: {e}")
                # Create fallback score
                fallback_score = RiskScore(
                    address=blockchain_analysis.get('address', 'unknown'),
                    overall_score=0.5,
                    risk_level=RiskLevel.MEDIUM,
                    confidence=0.0,
                    risk_factors=[],
                    timestamp=datetime.now().isoformat(),
                    explanation=f"Error in risk calculation: {str(e)}"
                )
                scores.append(fallback_score)
        
        return scores
    
    def export_risk_factors_for_ml(self, risk_factors: List[RiskFactor]) -> np.ndarray:
        """
        Export risk factors as feature vector for ML model training
        """
        feature_vector = np.zeros(len(self.weights))
        
        for i, (factor_name, _) in enumerate(self.weights.items()):
            # Find matching risk factor
            matching_factor = next((f for f in risk_factors if f.name == factor_name), None)
            if matching_factor:
                feature_vector[i] = matching_factor.value
        
        return feature_vector
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update risk factor weights (for model tuning)"""
        self.weights.update(new_weights)
        logger.info(f"Updated {len(new_weights)} risk factor weights")
    
    def get_risk_distribution_stats(self, risk_scores: List[RiskScore]) -> Dict:
        """Get statistical distribution of risk scores"""
        if not risk_scores:
            return {}
        
        scores = [rs.overall_score for rs in risk_scores]
        risk_levels = [rs.risk_level.value for rs in risk_scores]
        
        from collections import Counter
        level_counts = Counter(risk_levels)
        
        return {
            'total_addresses': len(risk_scores),
            'mean_score': np.mean(scores),
            'median_score': np.median(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'risk_level_distribution': dict(level_counts),
            'high_risk_percentage': (level_counts.get('HIGH', 0) + level_counts.get('CRITICAL', 0)) / len(risk_scores) * 100
        }

# Testing function
def test_risk_scoring_engine():
    """Test the risk scoring engine"""
    engine = RiskScoringEngine()
    
    # Sample test data
    blockchain_analysis = {
        'address': 'test_address',
        'basic_metrics': {'transaction_count': 500},
        'temporal_analysis': {
            'transaction_frequency': {'time_span_days': 30},
            'burst_detection': {'burst_count': 5}
        },
        'fraud_signals': {'mixing_service_usage': True},
        'clustering_analysis': {'cluster_size': 25},
        'network_analysis': {
            'centrality_measures': {'betweenness_centrality': 0.8}
        },
        'transaction_patterns': {'rapid_movement_count': 8}
    }
    
    ml_prediction = {
        'fraud_probability': 0.85,
        'confidence': 0.9,
        'model_used': 'ensemble'
    }
    
    enrichment_data = {
        'scam_database_check': {
            'is_scam': True,
            'scam_type': 'ponzi_scheme',
            'source': 'community_reports'
        },
        'user_reports': [{'type': 'fraud'}, {'type': 'scam'}],
        'external_sources': {
            'bitcoinabuse': {'is_suspicious': True}
        }
    }
    
    # Calculate risk score
    risk_score = engine.calculate_comprehensive_risk_score(
        blockchain_analysis, ml_prediction, enrichment_data
    )
    
    print(f"Risk Score Test Results:")
    print(f"Address: {risk_score.address}")
    print(f"Overall Score: {risk_score.overall_score:.3f}")
    print(f"Risk Level: {risk_score.risk_level.value}")
    print(f"Confidence: {risk_score.confidence:.3f}")
    print(f"Number of Risk Factors: {len(risk_score.risk_factors)}")
    print(f"\nExplanation:\n{risk_score.explanation}")

if __name__ == "__main__":
    test_risk_scoring_engine()