"""
Data Enrichment Module for BitScan
Integrates with scam databases and known fraud patterns
"""

import sqlite3
import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import csv
import hashlib

logger = logging.getLogger(__name__)

class ScamDatabase:
    """
    Local database for storing and managing scam reports and known fraud patterns
    """
    
    def __init__(self, db_path: str = "data/bitscan.db"):
        self.db_path = db_path
        self.ensure_database_exists()
    
    def ensure_database_exists(self):
        """Create database and tables if they don't exist"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Known scam addresses table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scam_addresses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    address TEXT UNIQUE NOT NULL,
                    scam_type TEXT,
                    description TEXT,
                    amount_stolen_btc REAL,
                    first_reported DATE,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source TEXT,
                    confidence_score REAL DEFAULT 0.8,
                    is_verified BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Analysis results cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    address TEXT NOT NULL,
                    analysis_data TEXT,
                    risk_score REAL,
                    is_flagged BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    analysis_hash TEXT
                )
            """)
            
            # User reports table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    address TEXT NOT NULL,
                    report_type TEXT,
                    description TEXT,
                    evidence_urls TEXT,
                    reporter_ip TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'pending',
                    verified_by TEXT
                )
            """)
            
            # Fraud patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fraud_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_name TEXT NOT NULL,
                    pattern_data TEXT,
                    description TEXT,
                    effectiveness_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def add_scam_address(self, address: str, scam_type: str, description: str = "", 
                        amount_stolen: float = 0.0, source: str = "manual") -> bool:
        """Add a known scam address to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO scam_addresses 
                    (address, scam_type, description, amount_stolen_btc, first_reported, source)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (address, scam_type, description, amount_stolen, datetime.now().date(), source))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error adding scam address {address}: {e}")
            return False
    
    def is_known_scam(self, address: str) -> Optional[Dict]:
        """Check if address is a known scam"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT scam_type, description, amount_stolen_btc, confidence_score, source
                    FROM scam_addresses WHERE address = ?
                """, (address,))
                
                result = cursor.fetchone()
                if result:
                    return {
                        'is_scam': True,
                        'scam_type': result[0],
                        'description': result[1],
                        'amount_stolen_btc': result[2],
                        'confidence_score': result[3],
                        'source': result[4]
                    }
                return {'is_scam': False}
        except Exception as e:
            logger.error(f"Error checking scam status for {address}: {e}")
            return {'is_scam': False, 'error': str(e)}
    
    def cache_analysis_result(self, address: str, analysis_data: Dict, 
                            expiry_hours: int = 24) -> bool:
        """Cache analysis result for faster future lookups"""
        try:
            # Create hash of analysis data for integrity
            analysis_json = json.dumps(analysis_data, sort_keys=True)
            analysis_hash = hashlib.md5(analysis_json.encode()).hexdigest()
            
            expires_at = datetime.now() + timedelta(hours=expiry_hours)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO analysis_cache 
                    (address, analysis_data, risk_score, is_flagged, expires_at, analysis_hash)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    address, 
                    analysis_json,
                    analysis_data.get('fraud_signals', {}).get('overall_fraud_score', 0.5),
                    analysis_data.get('fraud_signals', {}).get('overall_fraud_score', 0) > 0.5,
                    expires_at,
                    analysis_hash
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error caching analysis for {address}: {e}")
            return False
    
    def get_cached_analysis(self, address: str) -> Optional[Dict]:
        """Get cached analysis result if still valid"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT analysis_data, created_at FROM analysis_cache 
                    WHERE address = ? AND expires_at > ? 
                    ORDER BY created_at DESC LIMIT 1
                """, (address, datetime.now()))
                
                result = cursor.fetchone()
                if result:
                    analysis_data = json.loads(result[0])
                    analysis_data['cached'] = True
                    analysis_data['cached_at'] = result[1]
                    return analysis_data
                return None
        except Exception as e:
            logger.error(f"Error retrieving cached analysis for {address}: {e}")
            return None
    
    def add_user_report(self, address: str, report_type: str, description: str,
                       evidence_urls: List[str] = None, reporter_ip: str = "") -> bool:
        """Add user report about suspicious address"""
        try:
            evidence_json = json.dumps(evidence_urls or [])
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO user_reports 
                    (address, report_type, description, evidence_urls, reporter_ip)
                    VALUES (?, ?, ?, ?, ?)
                """, (address, report_type, description, evidence_json, reporter_ip))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error adding user report for {address}: {e}")
            return False
    
    def get_address_reports(self, address: str) -> List[Dict]:
        """Get all user reports for an address"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT report_type, description, evidence_urls, created_at, status
                    FROM user_reports WHERE address = ? ORDER BY created_at DESC
                """, (address,))
                
                reports = []
                for row in cursor.fetchall():
                    reports.append({
                        'report_type': row[0],
                        'description': row[1],
                        'evidence_urls': json.loads(row[2]),
                        'created_at': row[3],
                        'status': row[4]
                    })
                return reports
        except Exception as e:
            logger.error(f"Error getting reports for {address}: {e}")
            return []

class DataEnrichment:
    """
    Main data enrichment service that integrates multiple data sources
    """
    
    def __init__(self):
        self.scam_db = ScamDatabase()
        self.external_sources = {
            'bitcoinabuse': 'https://www.bitcoinabuse.com/api/address',
            'scammer_info': 'https://scammer.info/api/search',
            # Add more sources as needed
        }
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def initialize_known_scams(self):
        """Initialize database with known scam addresses"""
        # Sample known scam addresses (these would be from real sources)
        known_scams = [
            {
                'address': '1Ay8vMC7R1UbyCCZRVULMV7iQpHSAbguJP',
                'scam_type': 'ponzi_scheme',
                'description': 'BitConnect exit scam wallet',
                'amount_stolen': 0.0,
                'source': 'public_report'
            },
            {
                'address': '1FeexV6bAHb8ybZjqQMjJrcCrHGW9sb6uF',
                'scam_type': 'exchange_hack',
                'description': 'Mt. Gox exchange hack',
                'amount_stolen': 0.0,
                'source': 'public_report'
            },
            # Add more known scam addresses
        ]
        
        for scam in known_scams:
            self.scam_db.add_scam_address(
                scam['address'],
                scam['scam_type'],
                scam['description'],
                scam['amount_stolen'],
                scam['source']
            )
        
        logger.info(f"Initialized {len(known_scams)} known scam addresses")
    
    async def enrich_address_data(self, address: str) -> Dict:
        """
        Enrich address data with information from multiple sources
        """
        enrichment_data = {
            'address': address,
            'scam_database_check': {},
            'external_sources': {},
            'user_reports': [],
            'reputation_score': 1.0,  # Start with good reputation
            'enrichment_timestamp': datetime.now().isoformat()
        }
        
        # Check local scam database
        scam_check = self.scam_db.is_known_scam(address)
        enrichment_data['scam_database_check'] = scam_check
        
        if scam_check.get('is_scam'):
            enrichment_data['reputation_score'] = 0.0
        
        # Get user reports
        user_reports = self.scam_db.get_address_reports(address)
        enrichment_data['user_reports'] = user_reports
        
        if user_reports:
            # Reduce reputation based on number of reports
            report_penalty = min(len(user_reports) * 0.2, 0.8)
            enrichment_data['reputation_score'] -= report_penalty
        
        # Check external sources (if available)
        external_data = await self._check_external_sources(address)
        enrichment_data['external_sources'] = external_data
        
        # Adjust reputation based on external sources
        for source, data in external_data.items():
            if data.get('is_suspicious', False):
                enrichment_data['reputation_score'] *= 0.5
        
        # Ensure reputation score stays in valid range
        enrichment_data['reputation_score'] = max(0.0, min(1.0, enrichment_data['reputation_score']))
        
        return enrichment_data
    
    async def _check_external_sources(self, address: str) -> Dict:
        """
        Check external scam databases and sources
        """
        external_results = {}
        
        if not self.session:
            return external_results
        
        # Check Bitcoin Abuse database (if API key available)
        try:
            # This would require an API key in real implementation
            # For now, we'll simulate the check
            external_results['bitcoinabuse'] = {
                'checked': True,
                'is_suspicious': False,
                'reports_count': 0,
                'last_seen': None
            }
        except Exception as e:
            logger.warning(f"Error checking bitcoinabuse for {address}: {e}")
            external_results['bitcoinabuse'] = {'error': str(e)}
        
        # Add more external source checks here
        # For demonstration, we'll add some placeholder checks
        
        return external_results
    
    def load_scam_addresses_from_csv(self, csv_file_path: str) -> int:
        """
        Load scam addresses from a CSV file
        Expected format: address,scam_type,description,amount_stolen,source
        """
        loaded_count = 0
        
        try:
            with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    address = row.get('address', '').strip()
                    scam_type = row.get('scam_type', 'unknown').strip()
                    description = row.get('description', '').strip()
                    amount_stolen = float(row.get('amount_stolen', 0) or 0)
                    source = row.get('source', 'csv_import').strip()
                    
                    if address:
                        if self.scam_db.add_scam_address(address, scam_type, description, amount_stolen, source):
                            loaded_count += 1
            
            logger.info(f"Loaded {loaded_count} scam addresses from {csv_file_path}")
            
        except Exception as e:
            logger.error(f"Error loading scam addresses from CSV: {e}")
        
        return loaded_count
    
    async def batch_enrich_addresses(self, addresses: List[str]) -> Dict[str, Dict]:
        """
        Enrich multiple addresses in batch
        """
        results = {}
        
        # Process in batches to avoid overwhelming external APIs
        batch_size = 10
        for i in range(0, len(addresses), batch_size):
            batch = addresses[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [self.enrich_address_data(addr) for addr in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Store results
            for addr, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    results[addr] = {'error': str(result)}
                else:
                    results[addr] = result
            
            # Small delay between batches to be respectful to external APIs
            await asyncio.sleep(1)
        
        return results
    
    def get_enrichment_statistics(self) -> Dict:
        """
        Get statistics about the enrichment database
        """
        try:
            with sqlite3.connect(self.scam_db.db_path) as conn:
                cursor = conn.cursor()
                
                # Count scam addresses
                cursor.execute("SELECT COUNT(*) FROM scam_addresses")
                scam_count = cursor.fetchone()[0]
                
                # Count by scam type
                cursor.execute("""
                    SELECT scam_type, COUNT(*) FROM scam_addresses 
                    GROUP BY scam_type ORDER BY COUNT(*) DESC
                """)
                scam_types = dict(cursor.fetchall())
                
                # Count user reports
                cursor.execute("SELECT COUNT(*) FROM user_reports")
                report_count = cursor.fetchone()[0]
                
                # Count cached analyses
                cursor.execute("SELECT COUNT(*) FROM analysis_cache WHERE expires_at > ?", 
                             (datetime.now(),))
                cache_count = cursor.fetchone()[0]
                
                return {
                    'total_scam_addresses': scam_count,
                    'scam_types_distribution': scam_types,
                    'total_user_reports': report_count,
                    'cached_analyses': cache_count,
                    'database_path': self.scam_db.db_path
                }
                
        except Exception as e:
            logger.error(f"Error getting enrichment statistics: {e}")
            return {'error': str(e)}

# Testing and initialization
async def test_data_enrichment():
    """Test the data enrichment functionality"""
    async with DataEnrichment() as enricher:
        # Initialize with known scams
        enricher.initialize_known_scams()
        
        # Test address enrichment
        test_address = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"  # Genesis block
        enrichment_data = await enricher.enrich_address_data(test_address)
        
        print(f"Enrichment data for {test_address}:")
        print(json.dumps(enrichment_data, indent=2))
        
        # Get statistics
        stats = enricher.get_enrichment_statistics()
        print(f"Enrichment statistics: {stats}")

if __name__ == "__main__":
    asyncio.run(test_data_enrichment())