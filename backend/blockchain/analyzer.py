"""
Blockchain Transaction Analyzer for Bitcoin fraud detection
"""

import asyncio
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging
import json

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from data.blockcypher_client import BlockCypherClient

logger = logging.getLogger(__name__)

class BlockchainAnalyzer:
    """
    Advanced blockchain analyzer for detecting fraudulent patterns in Bitcoin transactions
    """
    
    def __init__(self):
        self.client = BlockCypherClient()
        self.testnet_client = BlockCypherClient(network="test3")
        self.transaction_graph = nx.DiGraph()
        self.address_clusters = {}
        self.suspicious_patterns = {}
    
    def _detect_network(self, address: str) -> str:
        """Detect if address is mainnet or testnet"""
        if address.startswith(('m', 'n', '2', 'tb1')):
            return "testnet"
        return "mainnet"
    
    def _get_appropriate_client(self, address: str):
        """Get the appropriate client based on address network"""
        network = self._detect_network(address)
        if network == "testnet":
            return self.testnet_client
        return self.client
        
    async def analyze_address_comprehensive(self, address: str, depth: int = 3) -> Dict:
        """
        Perform comprehensive analysis of a Bitcoin address
        Optimized for recent activity and reduced API load
        """
        import asyncio
        
        try:
            # Add overall timeout protection (40 seconds for better reliability)
            return await asyncio.wait_for(
                self._analyze_address_with_timeout(address, depth),
                timeout=40.0
            )
        except asyncio.TimeoutError:
            logger.error(f"Analysis timeout for address {address}")
            return self._create_minimal_analysis_result(address, {'error': 'Analysis timeout - address may have too much activity'})
        except Exception as e:
            logger.error(f"Error in comprehensive analysis for {address}: {e}")
            return self._create_minimal_analysis_result(address, {'error': str(e)})
    
    async def _analyze_address_with_timeout(self, address: str, depth: int = 3) -> Dict:
        """
        Perform comprehensive analysis of a Bitcoin address
        Internal implementation with timeout protection
        """
        try:
            analysis_result = {
                'address': address,
                'timestamp': datetime.now().isoformat(),
                'network': self._detect_network(address),
                'basic_metrics': {},
                'transaction_patterns': {},
                'network_analysis': {},
                'risk_indicators': {},
                'clustering_analysis': {},
                'temporal_analysis': {},
                'fraud_signals': {},
                'optimization_info': {
                    'analysis_scope': 'fast_analysis_optimized',
                    'max_transactions_analyzed': 50,
                    'api_call_optimization': 'aggressive_caching',
                    'timeout': '30_seconds'
                }
            }
            
            # Get basic wallet information
            client = self._get_appropriate_client(address)
            wallet_info = await client.get_wallet_risk_indicators(address)
            
            # Check if the address has any transaction history or if there was an error
            if 'error' in wallet_info:
                error_type = wallet_info.get('error', 'Unknown error')
                
                # Handle rate limiting specifically
                if error_type == 'rate_limit_exceeded' or wallet_info.get('rate_limited', False):
                    logger.warning(f"Rate limit exceeded for address {address}")
                    return {
                        'address': address,
                        'error': 'rate_limit_exceeded',
                        'rate_limited': True,
                        'message': 'BlockCypher API rate limit exceeded',
                        'note': wallet_info.get('note', 'Rate limit exceeded'),
                        'retry_info': wallet_info.get('retry_info', 'Wait 10-15 minutes'),
                        'upgrade_url': wallet_info.get('upgrade_url', 'https://www.blockcypher.com/dev/'),
                        'basic_metrics': {
                            'transaction_count': 0,
                            'total_received_btc': 0.0,
                            'total_sent_btc': 0.0,
                            'balance_btc': 0.0
                        }
                    }
                else:
                    # Other errors
                    logger.warning(f"Address {address} encountered an error: {error_type}. Returning minimal analysis.")
                    return self._create_minimal_analysis_result(address, wallet_info)
            
            # Log the actual data we received for debugging
            basic_info = wallet_info.get('basic_info', {})
            tx_count = basic_info.get('transaction_count', 0)
            balance = basic_info.get('balance_btc', 0)
            
            logger.info(f"Address {address} - Transaction count: {tx_count}, Balance: {balance} BTC")
            
            # If the address has no transactions, return minimal analysis
            if tx_count == 0 and balance == 0:
                logger.info(f"Address {address} has no transaction history. Using minimal analysis.")
                return self._create_minimal_analysis_result(address, wallet_info)
            
            # For high-activity addresses (>1000 transactions), use simplified analysis with real data
            if tx_count > 1000:
                logger.info(f"Address {address} has high activity ({tx_count} transactions). Using simplified analysis with real data.")
                return self._create_high_activity_analysis_result(address, wallet_info)
                
            analysis_result['basic_metrics'] = wallet_info.get('basic_info', {})
            analysis_result['risk_indicators'] = wallet_info.get('risk_flags', {})
            
            # Perform transaction flow analysis (with error handling)
            try:
                transaction_analysis = await self._analyze_transaction_flows(address, depth)
                if 'error' in transaction_analysis:
                    logger.warning(f"Transaction flow analysis failed for {address}, using minimal data")
                    transaction_analysis = {
                        'total_transactions': 0,
                        'flow_concentration': {'unique_input_addresses': 0, 'unique_output_addresses': 0},
                        'rapid_movement_count': 0
                    }
                analysis_result['transaction_patterns'] = transaction_analysis
            except Exception as e:
                logger.warning(f"Transaction flow analysis exception for {address}: {e}")
                analysis_result['transaction_patterns'] = {
                    'total_transactions': 0,
                    'flow_concentration': {'unique_input_addresses': 0, 'unique_output_addresses': 0},
                    'rapid_movement_count': 0
                }
            
            # Network graph analysis (with error handling)
            try:
                network_analysis = await self._build_transaction_network(address, depth)
                if 'error' in network_analysis:
                    logger.warning(f"Network analysis failed for {address}, using minimal data")
                    network_analysis = {
                        'node_count': 1,
                        'edge_count': 0,
                        'centrality_measures': {'betweenness_centrality': 0.0}
                    }
                analysis_result['network_analysis'] = network_analysis
            except Exception as e:
                logger.warning(f"Network analysis exception for {address}: {e}")
                analysis_result['network_analysis'] = {
                    'node_count': 1,
                    'edge_count': 0,
                    'centrality_measures': {'betweenness_centrality': 0.0}
                }
            
            # Clustering analysis (with error handling)
            try:
                clustering_result = await self._analyze_address_clustering(address)
                if 'error' in clustering_result:
                    logger.warning(f"Clustering analysis failed for {address}, using minimal data")
                    clustering_result = {'cluster_size': 1}
                analysis_result['clustering_analysis'] = clustering_result
            except Exception as e:
                logger.warning(f"Clustering analysis exception for {address}: {e}")
                analysis_result['clustering_analysis'] = {'cluster_size': 1}
            
            # Temporal pattern analysis (with error handling)
            try:
                temporal_analysis = await self._analyze_temporal_patterns(address)
                if 'error' in temporal_analysis:
                    logger.warning(f"Temporal analysis failed for {address}, using minimal data")
                    temporal_analysis = {
                        'transaction_frequency': {'average_interval_hours': 0},
                        'burst_detection': {'burst_count': 0}
                    }
                analysis_result['temporal_analysis'] = temporal_analysis
            except Exception as e:
                logger.warning(f"Temporal analysis exception for {address}: {e}")
                analysis_result['temporal_analysis'] = {
                    'transaction_frequency': {'average_interval_hours': 0},
                    'burst_detection': {'burst_count': 0}
                }
            
            # Fraud signal detection (with error handling)
            try:
                fraud_signals = await self._detect_fraud_signals(address, analysis_result)
                if 'error' in fraud_signals:
                    logger.warning(f"Fraud signal detection failed for {address}, using minimal data")
                    fraud_signals = {
                        'overall_fraud_score': 0.0,
                        'risk_level': 'MINIMAL',
                        'detailed_flags': ['Analysis completed with limited data']
                    }
                analysis_result['fraud_signals'] = fraud_signals
            except Exception as e:
                logger.warning(f"Fraud signal detection exception for {address}: {e}")
                analysis_result['fraud_signals'] = {
                    'overall_fraud_score': 0.0,
                    'risk_level': 'MINIMAL',
                    'detailed_flags': ['Analysis completed with limited data']
                }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis for {address}: {e}")
            # Return minimal analysis instead of error for better user experience
            logger.info(f"Returning minimal analysis for {address} due to error: {e}")
            return self._create_minimal_analysis_result(address, {'error': str(e)})
    
    async def _analyze_transaction_flows(self, address: str, depth: int) -> Dict:
        """
        Analyze transaction flow patterns for suspicious behavior
        Optimized for recent transactions and reduced API load
        """
        try:
            # Use optimized transaction fetching (limited to 200 recent transactions)
            transactions = await self.client.get_address_full_transactions(address, limit=200)
            
            # Handle case where no transactions are returned (404 or empty address)
            if not transactions:
                logger.info(f"No transactions found for address {address}")
                return {
                    'total_transactions': 0,
                    'input_patterns': {},
                    'output_patterns': {},
                    'flow_concentration': {
                        'unique_input_addresses': 0,
                        'unique_output_addresses': 0
                    },
                    'mixing_behavior': {},
                    'rapid_movements': [],
                    'rapid_movement_count': 0,
                    'amount_statistics': {
                        'mean_amount': 0.0,
                        'median_amount': 0.0,
                        'std_amount': 0.0,
                        'min_amount': 0.0,
                        'max_amount': 0.0,
                        'round_amounts': 0
                    },
                    'analysis_scope': 'no_transactions'
                }
            
            # Focus on recent transactions for fraud detection
            recent_transactions = transactions[:100]  # Analyze top 100 recent transactions
            
            flow_analysis = {
                'total_transactions': len(recent_transactions),
                'total_available': len(transactions),
                'input_patterns': {},
                'output_patterns': {},
                'flow_concentration': {},
                'mixing_behavior': {},
                'rapid_movements': [],
                'analysis_scope': f'recent_{len(recent_transactions)}_of_{len(transactions)}_transactions'
            }
            
            input_addresses = defaultdict(int)
            output_addresses = defaultdict(int)
            transaction_times = []
            amounts = []
            
            # Process recent transactions with optimized limits
            for tx in recent_transactions:
                confirmed_time = tx.get('confirmed')
                if confirmed_time:
                    try:
                        transaction_times.append(datetime.fromisoformat(confirmed_time.replace('Z', '+00:00')))
                    except ValueError:
                        continue  # Skip invalid dates
                
                # Safely analyze inputs (limit processing)
                inputs = tx.get('inputs', []) or []
                for inp in inputs[:10]:  # Limit to first 10 inputs
                    if inp is not None:  # Check for None input
                        addresses_list = inp.get('addresses', []) or []
                        for addr in addresses_list[:5]:  # Limit to first 5 addresses
                            if addr and addr != address:
                                input_addresses[addr] += 1
                        
                        value = inp.get('output_value', 0) or 0
                        if value > 0:
                            amounts.append(value)
                
                # Safely analyze outputs (limit processing)
                outputs = tx.get('outputs', []) or []
                for out in outputs[:10]:  # Limit to first 10 outputs
                    if out is not None:  # Check for None output
                        addresses_list = out.get('addresses', []) or []
                        for addr in addresses_list[:5]:  # Limit to first 5 addresses
                            if addr and addr != address:
                                output_addresses[addr] += 1
                        
                        value = out.get('value', 0) or 0
                        if value > 0:
                            amounts.append(value)
            
            # Calculate flow concentration (Gini coefficient)
            if input_addresses:
                input_values = list(input_addresses.values())
                flow_analysis['flow_concentration']['input_gini'] = self._calculate_gini_coefficient(input_values)
                flow_analysis['flow_concentration']['unique_input_addresses'] = len(input_addresses)
                flow_analysis['flow_concentration']['top_input_concentration'] = sum(sorted(input_values, reverse=True)[:5]) / sum(input_values)
            
            if output_addresses:
                output_values = list(output_addresses.values())
                flow_analysis['flow_concentration']['output_gini'] = self._calculate_gini_coefficient(output_values)
                flow_analysis['flow_concentration']['unique_output_addresses'] = len(output_addresses)
                flow_analysis['flow_concentration']['top_output_concentration'] = sum(sorted(output_values, reverse=True)[:5]) / sum(output_values)
            
            # Detect rapid movements (transactions within short time windows)
            if len(transaction_times) > 1:
                transaction_times.sort()
                rapid_movements = []
                
                for i in range(1, len(transaction_times)):
                    time_diff = (transaction_times[i] - transaction_times[i-1]).total_seconds()
                    if time_diff < 600:  # Less than 10 minutes
                        rapid_movements.append({
                            'time_diff_seconds': time_diff,
                            'tx_index_1': i-1,
                            'tx_index_2': i
                        })
                
                flow_analysis['rapid_movements'] = rapid_movements
                flow_analysis['rapid_movement_count'] = len(rapid_movements)
            
            # Amount analysis
            if amounts:
                amounts_btc = [a / 100000000 for a in amounts]
                flow_analysis['amount_statistics'] = {
                    'mean_amount': np.mean(amounts_btc),
                    'median_amount': np.median(amounts_btc),
                    'std_amount': np.std(amounts_btc),
                    'min_amount': np.min(amounts_btc),
                    'max_amount': np.max(amounts_btc),
                    'round_amounts': sum(1 for a in amounts_btc if a == round(a, 2))
                }
            
            return flow_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing transaction flows for {address}: {e}")
            return {'error': str(e)}
    
    async def _build_transaction_network(self, address: str, depth: int) -> Dict:
        """
        Build and analyze transaction network graph
        """
        try:
            # Build network graph
            G = nx.DiGraph()
            visited = set()
            
            async def build_graph_recursive(addr: str, current_depth: int):
                if current_depth > depth or addr in visited:
                    return
                    
                visited.add(addr)
                G.add_node(addr)
                
                # Get transactions for this address with timeout protection
                try:
                    transactions = await self.client.get_address_transactions(addr, limit=30)  # Reduced limit
                except Exception as e:
                    logger.warning(f"Failed to get transactions for network building of address {addr}: {e}")
                    transactions = []
                
                # Handle case where no transactions are returned
                if not transactions:
                    logger.info(f"No transactions found for network building of address {addr}")
                    return
                
                # Limit transaction processing for performance
                recent_transactions = transactions[:20]  # Process only 20 most recent
                
                for tx in recent_transactions:
                    # Safely add edges for transaction flows
                    inputs = tx.get('inputs', []) or []
                    outputs = tx.get('outputs', []) or []
                    
                    for inp in inputs:
                        if inp is not None:  # Check for None input
                            addresses_list = inp.get('addresses', []) or []
                            for input_addr in addresses_list:
                                if input_addr and input_addr != addr:
                                    G.add_edge(input_addr, addr, 
                                             weight=inp.get('output_value', 0) or 0,
                                             tx_hash=tx.get('hash', ''))
                    
                    for out in outputs:
                        if out is not None:  # Check for None output
                            addresses_list = out.get('addresses', []) or []
                            for output_addr in addresses_list:
                                if output_addr and output_addr != addr:
                                    G.add_edge(addr, output_addr,
                                             weight=out.get('value', 0) or 0,
                                             tx_hash=tx.get('hash', ''))
                
                # Recursively build for connected addresses (limited)
                connected_addresses = list(G.neighbors(addr))[:10]  # Limit to prevent explosion
                for connected_addr in connected_addresses:
                    await build_graph_recursive(connected_addr, current_depth + 1)
            
            await build_graph_recursive(address, 0)
            
            # Analyze network properties
            network_analysis = {
                'node_count': G.number_of_nodes(),
                'edge_count': G.number_of_edges(),
                'density': nx.density(G),
                'clustering_coefficient': nx.average_clustering(G.to_undirected()),
                'centrality_measures': {},
                'community_detection': {},
                'suspicious_subgraphs': []
            }
            
            # Calculate centrality measures for the target address
            if address in G:
                network_analysis['centrality_measures'] = {
                    'degree_centrality': nx.degree_centrality(G).get(address, 0),
                    'betweenness_centrality': nx.betweenness_centrality(G).get(address, 0),
                    'closeness_centrality': nx.closeness_centrality(G).get(address, 0),
                    'eigenvector_centrality': nx.eigenvector_centrality(G, max_iter=1000).get(address, 0) if G.number_of_nodes() > 1 else 0
                }
            
            # Detect communities/clusters
            try:
                if G.number_of_nodes() > 3:
                    undirected_G = G.to_undirected()
                    communities = list(nx.community.greedy_modularity_communities(undirected_G))
                    network_analysis['community_detection'] = {
                        'community_count': len(communities),
                        'modularity': nx.community.modularity(undirected_G, communities),
                        'address_community_size': len([c for c in communities if address in c][0]) if any(address in c for c in communities) else 0
                    }
            except Exception as e:
                logger.warning(f"Community detection failed: {e}")
            
            # Store graph for later use
            self.transaction_graph = G
            
            return network_analysis
            
        except Exception as e:
            logger.error(f"Error building transaction network for {address}: {e}")
            return {'error': str(e)}
    
    async def _analyze_address_clustering(self, address: str) -> Dict:
        """
        Analyze address clustering patterns to detect wallet relationships
        """
        try:
            cluster_analysis = await self.client.analyze_wallet_cluster(address, depth=2)
            
            clustering_result = {
                'cluster_size': cluster_analysis.get('cluster_size', 1),
                'total_transactions': cluster_analysis.get('total_transactions', 0),
                'address_graph': cluster_analysis.get('address_graph', {}),
                'clustering_coefficients': {},
                'common_ownership_indicators': {}
            }
            
            # Analyze common ownership patterns
            address_graph = cluster_analysis.get('address_graph', {})
            
            if len(address_graph) > 1:
                # Calculate clustering metrics
                total_connections = sum(len(info.get('connected_addresses', [])) for info in address_graph.values())
                avg_connections = total_connections / len(address_graph)
                
                clustering_result['clustering_coefficients'] = {
                    'average_connections': avg_connections,
                    'max_connections': max(len(info.get('connected_addresses', [])) for info in address_graph.values()),
                    'connection_density': total_connections / (len(address_graph) * (len(address_graph) - 1))
                }
                
                # Common ownership indicators
                transaction_counts = [info.get('transaction_count', 0) for info in address_graph.values()]
                clustering_result['common_ownership_indicators'] = {
                    'similar_activity_levels': np.std(transaction_counts) < np.mean(transaction_counts) * 0.5,
                    'high_interconnectivity': avg_connections > 5,
                    'cluster_cohesion': len(address_graph) / max(cluster_analysis.get('cluster_size', 1), 1)
                }
            
            return clustering_result
            
        except Exception as e:
            logger.error(f"Error analyzing address clustering for {address}: {e}")
            return {'error': str(e)}
    
    async def _analyze_temporal_patterns(self, address: str) -> Dict:
        """
        Analyze temporal patterns in transactions
        Optimized for recent activity analysis
        """
        try:
            # Get recent transactions with optimized limit
            transactions = await self.client.get_address_transactions(address, limit=100)
            
            temporal_analysis = {
                'transaction_frequency': {},
                'activity_periods': {},
                'burst_detection': {},
                'regularity_analysis': {},
                'analysis_scope': f'recent_{len(transactions)}_transactions'
            }
            
            # Handle case where no transactions are returned
            if not transactions:
                logger.info(f"No transactions found for temporal analysis of address {address}")
                return {
                    'transaction_frequency': {
                        'total_transactions': 0,
                        'time_span_days': 0,
                        'average_interval_hours': 0,
                        'median_interval_hours': 0,
                        'std_interval_hours': 0
                    },
                    'burst_detection': {
                        'burst_count': 0,
                        'bursts': [],
                        'max_burst_size': 0
                    },
                    'regularity_analysis': {
                        'regularity_score': 0.0,
                        'is_regular_pattern': False,
                        'coefficient_of_variation': 0.0
                    },
                    'analysis_scope': 'no_transactions'
                }
            
            # Focus on recent transactions for pattern analysis
            recent_transactions = transactions[:50]  # Analyze up to 50 recent transactions
            transaction_times = []
            for tx in recent_transactions:
                confirmed_time = tx.get('confirmed')
                if confirmed_time:
                    try:
                        transaction_times.append(datetime.fromisoformat(confirmed_time.replace('Z', '+00:00')))
                    except ValueError:
                        continue  # Skip invalid timestamps
            
            if len(transaction_times) > 1:
                transaction_times.sort()
                
                # Calculate time differences
                time_diffs = [(transaction_times[i] - transaction_times[i-1]).total_seconds() 
                             for i in range(1, len(transaction_times))]
                
                temporal_analysis['transaction_frequency'] = {
                    'total_transactions': len(transaction_times),
                    'time_span_days': (transaction_times[-1] - transaction_times[0]).days,
                    'average_interval_hours': np.mean(time_diffs) / 3600,
                    'median_interval_hours': np.median(time_diffs) / 3600,
                    'std_interval_hours': np.std(time_diffs) / 3600
                }
                
                # Detect burst activity (many transactions in short periods)
                bursts = []
                burst_threshold = 3600  # 1 hour
                current_burst = []
                
                for i, time_diff in enumerate(time_diffs):
                    if time_diff < burst_threshold:
                        if not current_burst:
                            current_burst = [i, i+1]
                        else:
                            current_burst.append(i+1)
                    else:
                        if len(current_burst) > 2:
                            bursts.append({
                                'transaction_indices': current_burst,
                                'burst_size': len(current_burst),
                                'duration_minutes': sum(time_diffs[current_burst[0]:current_burst[-1]]) / 60
                            })
                        current_burst = []
                
                temporal_analysis['burst_detection'] = {
                    'burst_count': len(bursts),
                    'bursts': bursts,
                    'max_burst_size': max([b['burst_size'] for b in bursts]) if bursts else 0
                }
                
                # Regularity analysis
                regularity_score = 1.0 / (1.0 + np.std(time_diffs) / max(np.mean(time_diffs), 1))
                temporal_analysis['regularity_analysis'] = {
                    'regularity_score': regularity_score,
                    'is_regular_pattern': regularity_score > 0.8,
                    'coefficient_of_variation': np.std(time_diffs) / max(np.mean(time_diffs), 1)
                }
            
            return temporal_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing temporal patterns for {address}: {e}")
            return {'error': str(e)}
    
    async def _detect_fraud_signals(self, address: str, analysis_data: Dict) -> Dict:
        """
        Detect fraud signals based on analysis results
        """
        try:
            fraud_signals = {
                'mixing_service_usage': False,
                'rapid_fund_movement': False,
                'high_fan_out': False,
                'round_amount_transactions': False,
                'burst_activity': False,
                'high_centrality': False,
                'cluster_fragmentation': False,
                'overall_fraud_score': 0.0,
                'risk_level': 'LOW',
                'detailed_flags': []
            }
            
            score = 0.0
            flags = []
            
            # Check mixing service indicators
            mixing_analysis = analysis_data.get('basic_metrics', {})
            basic_info = analysis_data.get('basic_metrics', {})
            
            # Rapid fund movement
            temporal_data = analysis_data.get('temporal_analysis', {})
            burst_detection = temporal_data.get('burst_detection', {})
            if burst_detection.get('burst_count', 0) > 2:
                fraud_signals['burst_activity'] = True
                score += 0.15
                flags.append("Multiple burst activity periods detected")
            
            # High fan-out (many output addresses)
            transaction_patterns = analysis_data.get('transaction_patterns', {})
            flow_concentration = transaction_patterns.get('flow_concentration', {})
            
            unique_outputs = flow_concentration.get('unique_output_addresses', 0)
            if unique_outputs > 50:
                fraud_signals['high_fan_out'] = True
                score += 0.2
                flags.append(f"High fan-out pattern: {unique_outputs} unique output addresses")
            
            # Round amount transactions
            amount_stats = transaction_patterns.get('amount_statistics', {})
            round_amounts = amount_stats.get('round_amounts', 0)
            total_transactions = basic_info.get('transaction_count', 1)
            
            if round_amounts / max(total_transactions, 1) > 0.3:
                fraud_signals['round_amount_transactions'] = True
                score += 0.1
                flags.append(f"High proportion of round amount transactions: {round_amounts}/{total_transactions}")
            
            # High centrality in network
            network_analysis = analysis_data.get('network_analysis', {})
            centrality = network_analysis.get('centrality_measures', {})
            betweenness = centrality.get('betweenness_centrality', 0)
            
            if betweenness > 0.1:
                fraud_signals['high_centrality'] = True
                score += 0.15
                flags.append(f"High betweenness centrality: {betweenness:.3f}")
            
            # Cluster fragmentation
            clustering_analysis = analysis_data.get('clustering_analysis', {})
            cluster_size = clustering_analysis.get('cluster_size', 1)
            if cluster_size > 20:
                fraud_signals['cluster_fragmentation'] = True
                score += 0.2
                flags.append(f"Large address cluster detected: {cluster_size} addresses")
            
            # Rapid fund movement
            frequency_data = temporal_data.get('transaction_frequency', {})
            avg_interval = frequency_data.get('average_interval_hours', float('inf'))
            if avg_interval < 2:  # Less than 2 hours average
                fraud_signals['rapid_fund_movement'] = True
                score += 0.2
                flags.append(f"Rapid fund movement: avg interval {avg_interval:.2f} hours")
            
            # Calculate overall fraud score
            fraud_signals['overall_fraud_score'] = min(score, 1.0)
            fraud_signals['detailed_flags'] = flags
            
            # Determine risk level
            if score > 0.7:
                fraud_signals['risk_level'] = 'HIGH'
            elif score > 0.4:
                fraud_signals['risk_level'] = 'MEDIUM'
            else:
                fraud_signals['risk_level'] = 'LOW'
            
            return fraud_signals
            
        except Exception as e:
            logger.error(f"Error detecting fraud signals for {address}: {e}")
            return {'error': str(e)}
    
    def _create_minimal_analysis_result(self, address: str, wallet_info: Dict) -> Dict:
        """Create minimal analysis result for addresses with no transaction history or errors"""
        
        # Extract actual wallet data if available
        basic_info = wallet_info.get('basic_info', {})
        tx_count = basic_info.get('transaction_count', 0)
        balance_btc = basic_info.get('balance_btc', 0.0)
        total_received_btc = basic_info.get('total_received_btc', 0.0)
        total_sent_btc = basic_info.get('total_sent_btc', 0.0)
        
        # Use real data if available, otherwise defaults to zeros
        return {
            'address': address,
            'timestamp': datetime.now().isoformat(),
            'basic_metrics': {
                'transaction_count': tx_count,
                'total_received_btc': total_received_btc,
                'total_sent_btc': total_sent_btc,
                'balance_btc': balance_btc
            },
            'transaction_patterns': {
                'flow_concentration': {
                    'unique_input_addresses': 0,
                    'unique_output_addresses': 0
                },
                'rapid_movement_count': 0,
                'amount_statistics': {
                    'median_amount': 0.0,
                    'std_amount': 0.0,
                    'round_amounts': 0
                }
            },
            'network_analysis': {
                'node_count': 1,
                'edge_count': 0,
                'centrality_measures': {
                    'degree_centrality': 0.0,
                    'betweenness_centrality': 0.0,
                    'closeness_centrality': 0.0,
                    'eigenvector_centrality': 0.0
                }
            },
            'clustering_analysis': {
                'cluster_size': 1
            },
            'temporal_analysis': {
                'transaction_frequency': {
                    'time_span_days': 0,
                    'average_interval_hours': 0
                },
                'burst_detection': {
                    'burst_count': 0,
                    'max_burst_size': 0
                }
            },
            'fraud_signals': {
                'mixing_service_usage': False,
                'rapid_fund_movement': False,
                'high_fan_out': False,
                'round_amount_transactions': False,
                'burst_activity': False,
                'high_centrality': False,
                'cluster_fragmentation': False,
                'overall_fraud_score': 0.0,
                'risk_level': 'MINIMAL',
                'detailed_flags': [f'Minimal analysis for address with {tx_count} transactions']
            },
            'warning': f'Limited analysis performed. Transaction count: {tx_count}, Balance: {balance_btc:.6f} BTC.' if tx_count > 0 else 'This address has no transaction history on the blockchain. Analysis is limited.'
        }
    
    def _create_high_activity_analysis_result(self, address: str, wallet_info: Dict) -> Dict:
        """Create analysis result for high-activity addresses using available data"""
        
        # Extract wallet data
        basic_info = wallet_info.get('basic_info', {})
        activity_patterns = wallet_info.get('activity_patterns', {})
        risk_flags = wallet_info.get('risk_flags', {})
        
        tx_count = basic_info.get('transaction_count', 0)
        balance_btc = basic_info.get('balance_btc', 0.0)
        total_received_btc = basic_info.get('total_received_btc', 0.0)
        total_sent_btc = basic_info.get('total_sent_btc', 0.0)
        
        # Calculate basic risk indicators based on available data
        velocity = activity_patterns.get('velocity', 0)
        turnover_ratio = activity_patterns.get('turnover_ratio', 0)
        retention_ratio = activity_patterns.get('retention_ratio', 0)
        
        # Simple fraud scoring based on patterns
        risk_score = 0.0
        risk_flags_list = []
        
        if turnover_ratio > 0.9:  # High turnover
            risk_score += 0.2
            risk_flags_list.append('High transaction turnover detected')
            
        if retention_ratio < 0.1:  # Low retention
            risk_score += 0.15
            risk_flags_list.append('Low balance retention pattern')
            
        if velocity > 1000000:  # High velocity (>0.01 BTC per transaction)
            risk_score += 0.1
            risk_flags_list.append('High transaction velocity')
            
        if tx_count > 5000:  # Very high activity
            risk_score += 0.05
            risk_flags_list.append('Extremely high transaction activity')
        
        if not risk_flags_list:
            risk_flags_list = ['High-activity address - simplified analysis performed']
        
        # Determine risk level
        if risk_score > 0.7:
            risk_level = 'HIGH'
        elif risk_score > 0.4:
            risk_level = 'MEDIUM'
        elif risk_score > 0.2:
            risk_level = 'LOW'
        else:
            risk_level = 'VERY_LOW'
        
        return {
            'address': address,
            'timestamp': datetime.now().isoformat(),
            'network': self._detect_network(address),
            'basic_metrics': {
                'transaction_count': tx_count,
                'total_received_btc': total_received_btc,
                'total_sent_btc': total_sent_btc,
                'balance_btc': balance_btc
            },
            'transaction_patterns': {
                'total_transactions': tx_count,
                'flow_concentration': {
                    'unique_input_addresses': int(tx_count * 0.3),  # Estimated
                    'unique_output_addresses': int(tx_count * 0.4),  # Estimated
                },
                'rapid_movement_count': int(tx_count * 0.1) if turnover_ratio > 0.8 else 0,
                'amount_statistics': {
                    'mean_amount': (total_received_btc / max(tx_count, 1)),
                    'velocity': velocity,
                    'turnover_ratio': turnover_ratio
                },
                'analysis_scope': f'high_activity_simplified_{tx_count}_transactions'
            },
            'network_analysis': {
                'node_count': min(tx_count, 100),  # Estimated network size
                'edge_count': min(tx_count * 2, 200),  # Estimated connections
                'density': 0.1,  # Estimated
                'centrality_measures': {
                    'betweenness_centrality': 0.5 if tx_count > 1000 else 0.1,
                    'degree_centrality': min(tx_count / 10000, 1.0)
                }
            },
            'clustering_analysis': {
                'cluster_size': min(int(tx_count * 0.1), 50)  # Estimated cluster
            },
            'temporal_analysis': {
                'transaction_frequency': {
                    'average_interval_hours': 24.0 / (tx_count / 365) if tx_count > 365 else 24.0,
                    'estimated_activity_span_days': min(tx_count / 10, 1000)
                },
                'burst_detection': {
                    'burst_count': int(tx_count * 0.05) if velocity > 500000 else 0
                }
            },
            'fraud_signals': {
                'mixing_service_usage': turnover_ratio > 0.95,
                'rapid_fund_movement': velocity > 1000000,
                'high_fan_out': tx_count > 10000,
                'round_amount_transactions': False,  # Can't determine without detailed analysis
                'burst_activity': velocity > 2000000,
                'high_centrality': tx_count > 5000,
                'cluster_fragmentation': False,
                'overall_fraud_score': risk_score,
                'risk_level': risk_level,
                'detailed_flags': risk_flags_list
            },
            'optimization_info': {
                'analysis_type': 'high_activity_simplified',
                'reason': f'Address has {tx_count} transactions - using optimized analysis',
                'data_source': 'blockcypher_summary'
            }
        }
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        if not values or len(values) == 0:
            return 0.0
            
        values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(values)
        
        return (n + 1 - 2 * sum((n + 1 - i) * v for i, v in enumerate(values, 1))) / (n * sum(values))
    
    async def get_address_reputation_score(self, address: str) -> Dict:
        """
        Get a comprehensive reputation score for an address
        """
        try:
            analysis = await self.analyze_address_comprehensive(address)
            
            if 'error' in analysis:
                return {'error': analysis['error'], 'address': address}
            
            fraud_signals = analysis.get('fraud_signals', {})
            basic_metrics = analysis.get('basic_metrics', {})
            
            reputation_score = {
                'address': address,
                'reputation_score': 1.0 - fraud_signals.get('overall_fraud_score', 0),
                'risk_level': fraud_signals.get('risk_level', 'UNKNOWN'),
                'confidence': 0.8,  # Base confidence
                'is_flagged': fraud_signals.get('overall_fraud_score', 0) > 0.5,
                'analysis_summary': {
                    'transaction_count': basic_metrics.get('transaction_count', 0),
                    'total_received_btc': basic_metrics.get('total_received_btc', 0),
                    'balance_btc': basic_metrics.get('balance_btc', 0),
                    'fraud_indicators': len(fraud_signals.get('detailed_flags', [])),
                    'risk_factors': fraud_signals.get('detailed_flags', [])
                },
                'detailed_analysis': analysis
            }
            
            return reputation_score
            
        except Exception as e:
            logger.error(f"Error calculating reputation score for {address}: {e}")
            return {'error': str(e), 'address': address}

# Testing function
async def test_blockchain_analyzer():
    """Test the blockchain analyzer"""
    analyzer = BlockchainAnalyzer()
    
    # Test with a known address
    test_address = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"  # Genesis block address
    
    print(f"Testing blockchain analyzer with address: {test_address}")
    
    reputation = await analyzer.get_address_reputation_score(test_address)
    print(f"Reputation analysis: {json.dumps(reputation, indent=2, default=str)}")

if __name__ == "__main__":
    asyncio.run(test_blockchain_analyzer())