# BitScan - Bitcoin Scam Pattern Analyzer

![BitScan Logo](https://img.shields.io/badge/BitScan-Bitcoin%20Fraud%20Detection-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🔍 Overview

BitScan is an advanced Bitcoin scam pattern analyzer designed to detect fraudulent investment schemes through comprehensive blockchain analytics, machine learning, and pattern recognition. The system helps protect users from crypto investment fraud by analyzing transaction flows, identifying suspicious wallet behaviors, and providing real-time risk assessments.

## 🚀 Key Features

- **🧠 AI-Powered Fraud Detection**: Machine learning models trained on known scam patterns
- **📊 Blockchain Analytics**: Deep transaction flow analysis and graph visualization
- **⚡ Real-time Risk Scoring**: Instant risk assessment for Bitcoin addresses
- **🌐 Public API**: RESTful API for integration with exchanges and DeFi platforms
- **🔗 Data Enrichment**: Integration with scam databases and community reports
- **📈 Network Analysis**: Graph analytics to visualize suspicious wallet activity
- **🛡️ Pattern Recognition**: Detection of mixing services, rapid fund movement, and cluster analysis

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.8+
- **Machine Learning**: scikit-learn, XGBoost, imbalanced-learn
- **Blockchain Data**: BlockCypher API integration
- **Graph Analytics**: NetworkX, igraph
- **Database**: SQLite (easily upgradeable to PostgreSQL)
- **Visualization**: Plotly, Dash, Cytoscape
- **Deployment**: Docker, Uvicorn

## 📁 Project Structure

```
bitscan/
├── src/bitscan/
│   ├── api/                 # FastAPI routes and endpoints
│   ├── blockchain/          # Blockchain analysis engine
│   ├── data/               # Data collection and enrichment
│   │   ├── blockcypher_client.py    # BlockCypher API client
│   │   └── elliptic_dataset.py      # EllipticPlusPlus dataset loader
│   ├── ml/                 # Machine learning models
│   │   ├── fraud_detector.py        # Enhanced fraud detection
│   │   └── feature_extraction.py   # Feature engineering
│   ├── utils/              # Utility functions
│   └── visualization/      # Graph visualization components
├── web/static/             # Frontend assets (CSS, JS)
├── data/                   # Data storage
│   ├── models/             # Trained ML models
│   └── elliptic_dataset/   # EllipticPlusPlus dataset (after download)
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── .env.example           # Environment configuration template
└── README.md              # This file
```

## 🚦 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git
- Optional: BlockCypher API token for enhanced data collection

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/bitscan/bitscan.git
   cd bitscan
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

## 🔧 Configuration

Create a `.env` file with the following configuration:

```env
# BlockCypher API (optional but recommended)
BLOCKCYPHER_API_TOKEN=your_api_token_here

# Application settings
FASTAPI_SECRET_KEY=your_secret_key_here
DATABASE_URL=sqlite:///bitscan.db

# ML Model settings
MODEL_THRESHOLD=0.5
```

## 📖 API Usage

### Analyze a Bitcoin Address

```bash
curl -X GET "http://localhost:8000/api/v1/analyze/1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
```

Response:
```json
{
  "address": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
  "risk_score": 0.1,
  "risk_level": "LOW",
  "is_flagged": false,
  "confidence": 0.95,
  "analysis_summary": {
    "transaction_count": 1234,
    "total_received_btc": 68.12,
    "risk_indicators": 2
  }
}
```

### Batch Analysis

```bash
curl -X POST "http://localhost:8000/api/v1/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "addresses": ["address1", "address2", "address3"],
    "depth": 2
  }'
```

### Transaction Graph

```bash
curl -X GET "http://localhost:8000/api/v1/graph/1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa?depth=2"
```

## 🤖 Machine Learning Models

BitScan employs multiple ML models for fraud detection:

1. **Random Forest**: Feature importance and interpretability
2. **XGBoost**: High-performance gradient boosting
3. **Logistic Regression**: Fast and interpretable baseline
4. **Isolation Forest**: Unsupervised anomaly detection
5. **Ensemble Model**: Combines predictions from multiple models

### Feature Engineering

The system extracts 40+ features including:

- Transaction patterns (velocity, frequency, amounts)
- Network topology (centrality measures, clustering)
- Temporal patterns (burst detection, regularity)
- Flow concentration (Gini coefficients, fan-out ratios)
- Behavioral indicators (mixing patterns, rapid movements)

## 🔍 Fraud Detection Techniques

### Pattern Recognition

- **Mixing Service Detection**: Identifies use of Bitcoin mixers
- **Rapid Fund Movement**: Detects quick consecutive transactions
- **High Fan-out**: Flags addresses with many output destinations
- **Cluster Analysis**: Groups related addresses under common control
- **Round Amount Detection**: Identifies suspiciously round transaction amounts

### Risk Indicators

- **High Velocity**: Frequent large transactions
- **Low Retention**: Minimal balance retention
- **Network Centrality**: High betweenness in transaction graphs
- **Temporal Clustering**: Burst activity patterns
- **Address Reuse**: Patterns suggesting automation

## 📊 Visualization

BitScan provides interactive visualizations:

- **Transaction Graphs**: Network visualization of address relationships
- **Risk Heat Maps**: Geographic and temporal risk distribution
- **Feature Importance**: ML model interpretability charts
- **Timeline Analysis**: Transaction patterns over time

## 🛡️ Security Considerations

- **Rate Limiting**: API endpoints are rate-limited
- **Input Validation**: All inputs are validated and sanitized
- **Privacy**: No personal data is stored or transmitted
- **Caching**: Analysis results are cached to improve performance

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Comprehensive system test
python test_bitscan.py

# Test EllipticPlusPlus integration
python integrate_elliptic_dataset.py

# Test individual components
python -c "from test_bitscan import BitScanTester; import asyncio; asyncio.run(BitScanTester().test_model_training())"
```

The test suite includes:
- ✅ API connection testing
- ✅ ML model training validation
- ✅ Fraud detection accuracy
- ✅ EllipticPlusPlus dataset integration
- ✅ Comprehensive system health checks

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- BlockCypher for blockchain data API
- scikit-learn and XGBoost teams for ML frameworks
- FastAPI team for the excellent web framework
- Bitcoin community for open blockchain data

## 📞 Support

- **Documentation**: [https://bitscan.readthedocs.io](https://bitscan.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/bitscan/bitscan/issues)
- **Email**: support@bitscan.com
- **Discord**: [BitScan Community](https://discord.gg/bitscan)

## 🔄 Roadmap

- [ ] Ethereum and other cryptocurrency support
- [ ] Browser extension for real-time warnings
- [ ] Mobile app for on-the-go analysis
- [ ] Advanced graph neural networks
- [ ] Integration with major exchanges
- [ ] Multi-language support

---

**⚠️ Disclaimer**: BitScan is a tool for analysis and education. Always conduct your own research and due diligence before making any cryptocurrency investments or transactions.