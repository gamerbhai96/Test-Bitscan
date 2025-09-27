"""
Blockchain Graph Visualization for BitScan
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from datetime import datetime
import colorsys

logger = logging.getLogger(__name__)

class BlockchainGraphVisualizer:
    """
    Visualize Bitcoin transaction graphs and suspicious patterns
    """
    
    def __init__(self):
        self.color_scheme = {
            'minimal': '#28a745',    # Green
            'low': '#17a2b8',        # Blue
            'medium': '#ffc107',     # Yellow
            'high': '#fd7e14',       # Orange
            'critical': '#dc3545',   # Red
            'unknown': '#6c757d',    # Gray
            'center': '#007bff'      # Blue for center node
        }
        
    def create_transaction_network_graph(
        self,
        graph_data: Dict,
        center_address: str,
        risk_scores: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Create interactive transaction network visualization
        """
        try:
            nodes = graph_data.get('nodes', [])
            edges = graph_data.get('edges', [])
            
            if not nodes:
                return self._create_empty_graph_message()
            
            # Create NetworkX graph for layout calculation
            G = nx.Graph()
            
            # Add nodes
            for node in nodes:
                G.add_node(node['id'])
            
            # Add edges
            for edge in edges:
                G.add_edge(edge['source'], edge['target'], weight=edge.get('weight', 1))
            
            # Calculate layout
            if len(G.nodes()) > 1:
                pos = nx.spring_layout(G, k=3, iterations=50)
            else:
                pos = {list(G.nodes())[0]: (0, 0)}
            
            # Prepare node traces
            node_trace = self._create_node_trace(nodes, pos, center_address, risk_scores)
            
            # Prepare edge traces
            edge_trace = self._create_edge_trace(edges, pos)
            
            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace])
            
            fig.update_layout(
                title=f"Bitcoin Transaction Network - Center: {center_address[:8]}...{center_address[-8:]}",
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Hover over nodes for details. Larger nodes = higher transaction volume",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor="left", yanchor="bottom",
                    font=dict(color="#666", size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            return {
                'type': 'plotly',
                'figure': fig.to_dict(),
                'summary': {
                    'node_count': len(nodes),
                    'edge_count': len(edges),
                    'center_node': center_address
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating transaction network graph: {e}")
            return self._create_error_graph(str(e))
    
    def _create_node_trace(
        self,
        nodes: List[Dict],
        pos: Dict,
        center_address: str,
        risk_scores: Optional[Dict[str, float]] = None
    ):
        """Create node trace for plotly"""
        
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in nodes:
            node_id = node['id']
            if node_id in pos:
                x, y = pos[node_id]
                node_x.append(x)
                node_y.append(y)
                
                # Node size based on degree or transaction count
                degree = node.get('degree', 1)
                size = max(20, min(60, 10 + degree * 3))
                node_size.append(size)
                
                # Color based on risk level or node type
                if node_id == center_address:
                    color = self.color_scheme['center']
                    risk_level = node.get('risk_level', 'unknown').lower()
                    if risk_level in self.color_scheme:
                        color = self.color_scheme[risk_level]
                elif risk_scores and node_id in risk_scores:
                    risk_score = risk_scores[node_id]
                    color = self._score_to_color(risk_score)
                else:
                    color = self.color_scheme['unknown']
                
                node_color.append(color)
                
                # Hover text
                hover_text = f"Address: {node_id[:8]}...{node_id[-8:]}<br>"
                hover_text += f"Degree: {degree}<br>"
                
                if 'risk_level' in node:
                    hover_text += f"Risk Level: {node['risk_level']}<br>"
                
                if 'betweenness' in node:
                    hover_text += f"Centrality: {node['betweenness']:.3f}<br>"
                
                if risk_scores and node_id in risk_scores:
                    hover_text += f"Risk Score: {risk_scores[node_id]:.3f}<br>"
                
                node_text.append(hover_text)
        
        return go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='rgba(50,50,50,0.5)')
            )
        )
    
    def _create_edge_trace(self, edges: List[Dict], pos: Dict):
        """Create edge trace for plotly"""
        
        edge_x = []
        edge_y = []
        
        for edge in edges:
            source = edge['source']
            target = edge['target']
            
            if source in pos and target in pos:
                x0, y0 = pos[source]
                x1, y1 = pos[target]
                
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
        
        return go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='rgba(125,125,125,0.5)'),
            hoverinfo='none',
            mode='lines'
        )
    
    def create_risk_heatmap(self, risk_data: List[Dict]) -> Dict:
        """
        Create risk score heatmap visualization
        """
        try:
            if not risk_data:
                return self._create_empty_graph_message("No risk data available")
            
            # Prepare data
            addresses = [item['address'][:10] + '...' for item in risk_data]
            risk_scores = [item.get('risk_score', 0) for item in risk_data]
            risk_levels = [item.get('risk_level', 'UNKNOWN') for item in risk_data]
            
            # Create color scale
            colors = []
            for score in risk_scores:
                if score >= 0.9:
                    colors.append('#dc3545')  # Critical - Red
                elif score >= 0.7:
                    colors.append('#fd7e14')  # High - Orange
                elif score >= 0.4:
                    colors.append('#ffc107')  # Medium - Yellow
                elif score >= 0.2:
                    colors.append('#17a2b8')  # Low - Blue
                else:
                    colors.append('#28a745')  # Minimal - Green
            
            fig = go.Figure(data=go.Bar(
                x=addresses,
                y=risk_scores,
                marker_color=colors,
                text=[f"{score:.2f}" for score in risk_scores],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' +
                             'Risk Score: %{y:.3f}<br>' +
                             '<extra></extra>'
            ))
            
            fig.update_layout(
                title="Bitcoin Address Risk Score Distribution",
                xaxis_title="Bitcoin Addresses",
                yaxis_title="Risk Score",
                yaxis=dict(range=[0, 1]),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            return {
                'type': 'plotly',
                'figure': fig.to_dict(),
                'summary': {
                    'total_addresses': len(risk_data),
                    'avg_risk_score': np.mean(risk_scores),
                    'high_risk_count': sum(1 for score in risk_scores if score >= 0.7)
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating risk heatmap: {e}")
            return self._create_error_graph(str(e))
    
    def create_temporal_analysis_chart(self, temporal_data: Dict) -> Dict:
        """
        Create temporal pattern analysis visualization
        """
        try:
            burst_data = temporal_data.get('burst_detection', {})
            frequency_data = temporal_data.get('transaction_frequency', {})
            
            # Create subplot figure
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=['Transaction Burst Pattern', 'Transaction Frequency Analysis'],
                vertical_spacing=0.12
            )
            
            # Burst pattern chart
            bursts = burst_data.get('bursts', [])
            if bursts:
                burst_sizes = [b.get('burst_size', 0) for b in bursts]
                burst_durations = [b.get('duration_minutes', 0) for b in bursts]
                
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(burst_sizes) + 1)),
                    y=burst_sizes,
                    mode='markers+lines',
                    name='Burst Size',
                    marker=dict(size=10, color='red'),
                    hovertemplate='Burst %{x}<br>Size: %{y} transactions<br><extra></extra>'
                ), row=1, col=1)
            
            # Frequency metrics
            metrics = {
                'Avg Interval (hours)': frequency_data.get('average_interval_hours', 0),
                'Median Interval (hours)': frequency_data.get('median_interval_hours', 0),
                'Std Interval (hours)': frequency_data.get('std_interval_hours', 0)
            }
            
            fig.add_trace(go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                name='Frequency Metrics',
                marker_color=['blue', 'green', 'orange']
            ), row=2, col=1)
            
            fig.update_layout(
                title="Temporal Transaction Pattern Analysis",
                height=600,
                showlegend=False
            )
            
            fig.update_xaxes(title_text="Burst Number", row=1, col=1)
            fig.update_yaxes(title_text="Transactions", row=1, col=1)
            fig.update_xaxes(title_text="Metric", row=2, col=1)
            fig.update_yaxes(title_text="Hours", row=2, col=1)
            
            return {
                'type': 'plotly',
                'figure': fig.to_dict(),
                'summary': {
                    'burst_count': len(bursts),
                    'avg_interval_hours': frequency_data.get('average_interval_hours', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating temporal analysis chart: {e}")
            return self._create_error_graph(str(e))
    
    def create_feature_importance_chart(self, feature_importance: Dict) -> Dict:
        """
        Create ML model feature importance visualization
        """
        try:
            if not feature_importance:
                return self._create_empty_graph_message("No feature importance data available")
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:15]  # Top 15 features
            
            feature_names = [name.replace('_', ' ').title() for name, _ in top_features]
            importance_values = [importance for _, importance in top_features]
            
            fig = go.Figure(data=go.Bar(
                x=importance_values,
                y=feature_names,
                orientation='h',
                marker_color=px.colors.sequential.Viridis[:len(top_features)],
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<br><extra></extra>'
            ))
            
            fig.update_layout(
                title="ML Model Feature Importance (Top 15)",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=600,
                margin=dict(l=200)
            )
            
            return {
                'type': 'plotly',
                'figure': fig.to_dict(),
                'summary': {
                    'total_features': len(feature_importance),
                    'top_feature': sorted_features[0][0] if sorted_features else None,
                    'top_importance': sorted_features[0][1] if sorted_features else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating feature importance chart: {e}")
            return self._create_error_graph(str(e))
    
    def create_cluster_analysis_visualization(self, cluster_data: Dict) -> Dict:
        """
        Create address cluster analysis visualization
        """
        try:
            address_graph = cluster_data.get('address_graph', {})
            
            if not address_graph:
                return self._create_empty_graph_message("No cluster data available")
            
            # Prepare data for visualization
            addresses = list(address_graph.keys())
            connection_counts = [
                len(info.get('connected_addresses', []))
                for info in address_graph.values()
            ]
            transaction_counts = [
                info.get('transaction_count', 0)
                for info in address_graph.values()
            ]
            depths = [
                info.get('depth', 0)
                for info in address_graph.values()
            ]
            
            # Create bubble chart
            fig = go.Figure(data=go.Scatter(
                x=connection_counts,
                y=transaction_counts,
                mode='markers',
                marker=dict(
                    size=[max(10, min(50, count/10)) for count in transaction_counts],
                    color=depths,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Depth Level")
                ),
                text=[f"{addr[:8]}...{addr[-6:]}" for addr in addresses],
                hovertemplate='<b>%{text}</b><br>' +
                             'Connections: %{x}<br>' +
                             'Transactions: %{y}<br>' +
                             'Depth: %{marker.color}<br>' +
                             '<extra></extra>'
            ))
            
            fig.update_layout(
                title="Address Cluster Analysis",
                xaxis_title="Number of Connected Addresses",
                yaxis_title="Transaction Count",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            return {
                'type': 'plotly',
                'figure': fig.to_dict(),
                'summary': {
                    'cluster_size': len(addresses),
                    'avg_connections': np.mean(connection_counts) if connection_counts else 0,
                    'total_transactions': sum(transaction_counts)
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating cluster visualization: {e}")
            return self._create_error_graph(str(e))
    
    def _score_to_color(self, score: float) -> str:
        """Convert risk score to color"""
        if score >= 0.9:
            return self.color_scheme['critical']
        elif score >= 0.7:
            return self.color_scheme['high']
        elif score >= 0.4:
            return self.color_scheme['medium']
        elif score >= 0.2:
            return self.color_scheme['low']
        else:
            return self.color_scheme['minimal']
    
    def _create_empty_graph_message(self, message: str = "No data available") -> Dict:
        """Create empty graph with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor="center", yanchor="middle",
            font=dict(size=16, color="gray"),
            showarrow=False
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return {
            'type': 'plotly',
            'figure': fig.to_dict(),
            'summary': {'message': message}
        }
    
    def _create_error_graph(self, error_message: str) -> Dict:
        """Create error graph"""
        return self._create_empty_graph_message(f"Error: {error_message}")
    
    def export_visualization_data(self, visualization_data: Dict) -> str:
        """Export visualization data as JSON"""
        try:
            return json.dumps(visualization_data, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error exporting visualization data: {e}")
            return "{\"error\": \"Failed to export data\"}"

# Testing function
def test_blockchain_visualizer():
    """Test the blockchain visualizer"""
    visualizer = BlockchainGraphVisualizer()
    
    # Sample graph data
    sample_graph_data = {
        'nodes': [
            {'id': 'addr1', 'degree': 5, 'risk_level': 'HIGH'},
            {'id': 'addr2', 'degree': 3, 'risk_level': 'LOW'},
            {'id': 'addr3', 'degree': 8, 'risk_level': 'MEDIUM'}
        ],
        'edges': [
            {'source': 'addr1', 'target': 'addr2', 'weight': 1.5},
            {'source': 'addr2', 'target': 'addr3', 'weight': 2.0}
        ]
    }
    
    # Test network graph
    network_viz = visualizer.create_transaction_network_graph(
        sample_graph_data, 'addr1'
    )
    print("Network visualization created:", network_viz['summary'])
    
    # Sample risk data
    sample_risk_data = [
        {'address': 'addr1', 'risk_score': 0.8, 'risk_level': 'HIGH'},
        {'address': 'addr2', 'risk_score': 0.3, 'risk_level': 'LOW'},
        {'address': 'addr3', 'risk_score': 0.6, 'risk_level': 'MEDIUM'}
    ]
    
    # Test risk heatmap
    heatmap_viz = visualizer.create_risk_heatmap(sample_risk_data)
    print("Risk heatmap created:", heatmap_viz['summary'])

if __name__ == "__main__":
    test_blockchain_visualizer()