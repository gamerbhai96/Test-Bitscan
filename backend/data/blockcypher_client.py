"""
BlockCypher API Data Collector for Bitcoin blockchain data
"""

import asyncio
import httpx
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class BlockCypherClient:
    """
    Async client for BlockCypher API to fetch Bitcoin blockchain data
    """
    
    def __init__(self, api_token: Optional[str] = None, network: str = "main"):
        self.api_token = api_token or os.getenv('BLOCKCYPHER_API_TOKEN')
        self.network = network
        
        # Log API token status for debugging (only once)
        if self.api_token:
            logger.debug(f"BlockCypher API token configured (ending: ...{self.api_token[-4:] if len(self.api_token) >= 4 else 'short'})")
        else:
            logger.debug("No BlockCypher API token configured - using public rate limits")
        
        # Set base URL based on network
        if network == "test3" or network == "testnet":
            self.base_url = "https://api.blockcypher.com/v1/btc/test3"
            logger.debug("Using BlockCypher testnet API")
        else:
            self.base_url = "https://api.blockcypher.com/v1/btc/main"
            logger.debug("Using BlockCypher mainnet API")
            
        self.rate_limit_delay = 1.0  # Reduced delay for better responsiveness
        self.last_request_time = 0
        self.request_count = 0
        self.max_requests_per_minute = 15  # More conservative rate limiting
        self.rate_limit_reset_time = None  # Track when rate limits reset
        self.hourly_request_count = 0
        self.hour_start_time = time.time()
        self.cache = {}  # Simple caching to reduce API calls
        self.cache_ttl = 900  # 15 minutes cache TTL for better data freshness
        self.cache_stats = {'hits': 0, 'misses': 0}  # Cache performance tracking
        
    async def _make_request_with_retry(self, endpoint: str, params: Dict = None, max_retries: int = 3) -> Dict:
        """Make request with exponential backoff retry mechanism"""
        for attempt in range(max_retries):
            try:
                return await self._make_request(endpoint, params)
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    logger.error(f"Final attempt failed for {endpoint}: {e}")
                    raise

                # Exponential backoff: wait 2^attempt seconds
                wait_time = 2 ** attempt
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)

        # This should never be reached, but just in case
        raise Exception(f"Failed after {max_retries} attempts")

    async def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        
        # Check cache first for faster responses
        cache_key = f"{endpoint}_{str(params)}"
        if cache_key in self.cache:
            cache_time, cached_data = self.cache[cache_key]
            # Use cache for configured TTL to improve data freshness
            if time.time() - cache_time < self.cache_ttl:
                logger.debug(f"Using cached data for {endpoint} (age: {time.time() - cache_time:.1f}s)")
                self.cache_stats['hits'] += 1
                return cached_data
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
                logger.debug(f"Cache expired for {endpoint}, fetching fresh data")
        else:
            self.cache_stats['misses'] += 1
        
        # Enhanced rate limiting with exponential backoff
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # Reset hourly counter if an hour has passed
        if current_time - self.hour_start_time > 3600:  # 1 hour
            self.hourly_request_count = 0
            self.hour_start_time = current_time
            logger.debug("Hourly rate limit counter reset")
        
        # Check if we're approaching hourly limit (180 out of 200)
        if self.hourly_request_count >= 180:
            logger.warning(f"⚠️  Approaching hourly limit ({self.hourly_request_count}/200 requests used). Being more conservative.")
            # Still allow requests but with longer delays
            await asyncio.sleep(5.0)  # 5 second delay when approaching limit
        
        # If we hit rate limits recently, wait longer
        if self.rate_limit_reset_time and current_time < self.rate_limit_reset_time:
            wait_time = self.rate_limit_reset_time - current_time
            logger.warning(f"Still in rate limit cooldown, waiting {wait_time:.1f} seconds")
            await asyncio.sleep(wait_time)
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.request_count += 1
        self.hourly_request_count += 1
        
        logger.debug(f"API Request #{self.request_count} (hourly: {self.hourly_request_count}/200)")
        
        # Reduced aggressive delays - only add extra delay every 10 requests
        if self.request_count % 10 == 0 and self.request_count > 0:  # Every 10 requests
            await asyncio.sleep(1.0)  # 1 second delay instead of 2
            logger.debug(f"Added extra 1s delay after {self.request_count} requests")
        
        url = f"{self.base_url}{endpoint}"
        
        # Add API token to params if available
        if params is None:
            params = {}
        
        if self.api_token:
            params['token'] = self.api_token
            logger.debug(f"Making API request to {url} with token")
        else:
            logger.debug(f"Making API request to {url} without token")
            
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                self.last_request_time = time.time()
                result = response.json()
                
                # Cache successful responses for faster future access
                if result and not ('error' in result):
                    self.cache[cache_key] = (time.time(), result)
                    # Limit cache size to prevent memory issues
                    if len(self.cache) > 100:
                        # Remove oldest entries
                        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][0])
                        del self.cache[oldest_key]
                
                return result
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Too Many Requests
                    logger.error(f"⚠️  API Rate limit exceeded! Status: 429 Too Many Requests")
                    logger.error(f"🔄 You've hit BlockCypher's rate limits. Please wait before making more requests.")
                    logger.error(f"💡 Consider upgrading your API plan at: https://www.blockcypher.com/dev/")
                    
                    # Set rate limit reset time (wait 5 minutes)
                    self.rate_limit_reset_time = time.time() + 300  # 5 minutes
                    
                    # Wait longer for rate limits
                    await asyncio.sleep(10.0)  # Wait 10 seconds before returning
                    
                    # Return structured response indicating rate limit
                    return {
                        'error': 'rate_limit_exceeded',
                        'message': 'BlockCypher API rate limit exceeded - please wait 5 minutes',
                        'retry_after': '5 minutes',
                        'upgrade_info': 'https://www.blockcypher.com/dev/',
                        'status_code': 429
                    }
                elif e.response.status_code == 404:
                    logger.info(f"Address {endpoint} has no transaction history (404 - not found)")
                    return {}
                else:
                    logger.error(f"HTTP error fetching {endpoint}: {e}")
                    raise
            except httpx.TimeoutException as e:
                logger.error(f"Timeout fetching {endpoint}: {e}")
                return {'error': 'timeout', 'message': 'Request timed out'}
            except httpx.HTTPError as e:
                logger.error(f"HTTP error fetching {endpoint}: {e}")
                return {'error': 'http_error', 'message': str(e)}
            except Exception as e:
                logger.error(f"Error fetching {endpoint}: {e}")
                return {'error': 'unknown_error', 'message': str(e)}
    
    async def get_address_info(self, address: str) -> Dict:
        """Get comprehensive address information with enhanced rate limit handling"""
        try:
            response = await self._make_request_with_retry(f"/addrs/{address}")
            
            # Handle rate limit response
            if isinstance(response, dict) and response.get('error') == 'rate_limit_exceeded':
                logger.warning(f"⚠️  Rate limit hit for address {address}")
                return {
                    'address': address,
                    'balance': 0,
                    'total_received': 0,
                    'total_sent': 0,
                    'n_tx': 0,
                    'unconfirmed_balance': 0,
                    'final_balance': 0,
                    'error': 'rate_limit_exceeded',
                    'rate_limited': True,
                    'note': 'BlockCypher API rate limit exceeded - try again in 5-10 minutes',
                    'retry_info': 'Free tier: 3 req/sec, 200 req/hour. Upgrade for unlimited access.',
                    'upgrade_url': 'https://www.blockcypher.com/dev/'
                }
            
            if response and isinstance(response, dict) and 'address' in response:
                return response
            elif response == {}:
                # Handle empty response (404)
                logger.info(f"Address {address} not found (404) - creating minimal response")
                return {
                    'address': address,
                    'balance': 0,
                    'total_received': 0,
                    'total_sent': 0,
                    'n_tx': 0,
                    'unconfirmed_balance': 0,
                    'final_balance': 0,
                    'note': 'Address has no transaction history or is not found'
                }
            else:
                logger.warning(f"Unexpected response for address {address}: {response}")
                return response
                
        except Exception as e:
            logger.error(f"Error fetching address info for {address}: {e}")
            # Return a basic structure for addresses with errors
            return {
                'address': address,
                'balance': 0,
                'total_received': 0,
                'total_sent': 0,
                'n_tx': 0,
                'unconfirmed_balance': 0,
                'final_balance': 0,
                'error': str(e),
                'note': 'Error occurred while fetching address information'
            }
    
    async def get_address_balance(self, address: str) -> Dict:
        """Get address balance"""
        try:
            return await self._make_request(f"/addrs/{address}/balance")
        except Exception as e:
            logger.error(f"Error fetching balance for {address}: {e}")
            return {}
    
    async def get_address_transactions(self, address: str, limit: int = 50, before: Optional[str] = None) -> List[Dict]:
        """Get transactions for an address with enhanced fallback methods and 404 handling"""
        try:
            # Increased limit to 10000 as requested
            effective_limit = min(limit, 10000)
            params = {'limit': effective_limit}
            if before:
                params['before'] = before
            
            # Try primary endpoint first
            try:
                response = await self._make_request_with_retry(f"/addrs/{address}/txs", params)
                
                if response and 'txs' in response:
                    logger.debug(f"Primary endpoint successful for {address}: {len(response.get('txs', []))} transactions")
                    return response.get('txs', [])
                elif response == {}:  # 404 response converted to empty dict
                    logger.info(f"Address {address} not found on primary endpoint (404)")
                    # Continue to fallback
                else:
                    logger.warning(f"Unexpected response from primary endpoint for {address}: {response}")
                    # Continue to fallback
            except Exception as primary_error:
                logger.info(f"Primary endpoint failed for {address}: {primary_error}")
                # Continue to fallback
            
            # Fallback method: try full endpoint
            logger.info(f"Trying fallback /full endpoint for {address}")
            try:
                response = await self._make_request_with_retry(f"/addrs/{address}/full", params)
                
                if response and 'txs' in response:
                    logger.info(f"Fallback successful for {address}: {len(response.get('txs', []))} transactions")
                    return response.get('txs', [])
                elif response == {}:  # 404 response
                    logger.info(f"Address {address} not found on fallback endpoint (404) - likely no transaction history")
                    return []  # Return empty list for addresses with no transactions
                else:
                    logger.warning(f"Unexpected fallback response for {address}: {response}")
                    return []
            except Exception as fallback_error:
                logger.warning(f"Fallback endpoint failed for {address}: {fallback_error}")
                return []
            
        except Exception as e:
            logger.error(f"Error fetching transactions for {address}: {e}")
            return []
    
    async def get_transaction_details(self, tx_hash: str) -> Dict:
        """Get detailed transaction information"""
        try:
            return await self._make_request(f"/txs/{tx_hash}")
        except Exception as e:
            logger.error(f"Error fetching transaction {tx_hash}: {e}")
            return {}
    
    async def get_address_full_transactions(self, address: str, limit: int = 200) -> List[Dict]:
        """Get full transaction details for an address with enhanced 404 handling and time-based filtering"""
        try:
            # Increased limit to 10000 transactions and optimize for recent activity
            effective_limit = min(limit, 10000)
            params = {'limit': effective_limit, 'includeHex': 'false'}
            
            response = await self._make_request_with_retry(f"/addrs/{address}/full", params)
            
            # Handle empty response (404 case)
            if response == {}:
                logger.info(f"Address {address} has no transaction history (404 from /full endpoint)")
                return []
            
            transactions = response.get('txs', [])
            
            # If no transactions, return empty list
            if not transactions:
                logger.info(f"Address {address} returned no transactions from /full endpoint")
                return []
            
            # Filter to last 30 days for fraud detection focus
            if transactions:
                from datetime import datetime, timedelta, timezone
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
                
                recent_transactions = []
                for tx in transactions:
                    confirmed_time = tx.get('confirmed')
                    if confirmed_time:
                        try:
                            # Parse the datetime and make it timezone-aware if needed
                            if confirmed_time.endswith('Z'):
                                tx_date = datetime.fromisoformat(confirmed_time.replace('Z', '+00:00'))
                            else:
                                tx_date = datetime.fromisoformat(confirmed_time)
                                # Make timezone-aware if naive
                                if tx_date.tzinfo is None:
                                    tx_date = tx_date.replace(tzinfo=timezone.utc)
                            
                            if tx_date >= cutoff_date:
                                recent_transactions.append(tx)
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Date parsing failed for transaction: {e}")
                            # Include transaction if date parsing fails
                            recent_transactions.append(tx)
                    else:
                        # Include unconfirmed transactions
                        recent_transactions.append(tx)
                
                logger.info(f"Filtered {len(transactions)} transactions to {len(recent_transactions)} recent transactions for {address}")
                return recent_transactions[:10000]  # Increased safety limit
            
            return transactions
        except Exception as e:
            logger.error(f"Error fetching full transactions for {address}: {e}")
            return []
    
    async def analyze_wallet_cluster(self, address: str, depth: int = 2) -> Dict:
        """
        Analyze wallet cluster by following transaction flows
        Optimized for recent activity and reduced API calls
        """
        visited_addresses = set()
        address_graph = {}
        transactions = {}
        
        # Strict limits to prevent API overload
        max_addresses = 10  # Reduced from 20
        max_depth = min(depth, 2)  # Limit depth to 2
        
        async def traverse_address(addr: str, current_depth: int):
            if (current_depth > max_depth or 
                addr in visited_addresses or 
                len(visited_addresses) >= max_addresses):
                return
                
            visited_addresses.add(addr)
            
            # Get limited recent transactions with error handling
            try:
                txs = await self.get_address_transactions(addr, limit=30)  # Reduced from 50
            except Exception as e:
                logger.warning(f"Failed to get transactions for {addr} in cluster analysis: {e}")
                txs = []
            
            connected_addresses = set()
            
            # Only analyze recent transactions to reduce load
            recent_txs = txs[:20]  # Limit to 20 most recent transactions
            
            for tx in recent_txs:
                tx_hash = tx.get('hash')
                if tx_hash:  # Only store if valid hash
                    transactions[tx_hash] = tx
                
                # Safely analyze inputs and outputs with None checks
                inputs = tx.get('inputs', []) or []
                outputs = tx.get('outputs', []) or []
                
                # Look for common ownership patterns (limit processing)
                for inp in inputs[:5]:  # Limit to first 5 inputs
                    if inp is not None:  # Check for None input
                        addresses_list = inp.get('addresses', []) or []
                        for addr_info in addresses_list[:3]:  # Limit to first 3 addresses
                            if addr_info and addr_info != addr and len(connected_addresses) < 10:
                                connected_addresses.add(addr_info)
                
                for out in outputs[:5]:  # Limit to first 5 outputs
                    if out is not None:  # Check for None output
                        addresses_list = out.get('addresses', []) or []
                        for addr_info in addresses_list[:3]:  # Limit to first 3 addresses
                            if addr_info and addr_info != addr and len(connected_addresses) < 10:
                                connected_addresses.add(addr_info)
            
            address_graph[addr] = {
                'connected_addresses': list(connected_addresses)[:5],  # Limit stored connections
                'transaction_count': len(recent_txs),
                'depth': current_depth
            }
            
            # Recursively analyze fewer connected addresses
            for connected_addr in list(connected_addresses)[:3]:  # Reduced from 5
                if len(visited_addresses) < max_addresses:
                    try:
                        await traverse_address(connected_addr, current_depth + 1)
                    except Exception as e:
                        logger.warning(f"Error in recursive cluster analysis for {connected_addr}: {e}")
                        continue
        
        try:
            await traverse_address(address, 0)
        except Exception as e:
            logger.error(f"Error in cluster analysis for {address}: {e}")
        
        return {
            'address_graph': address_graph,
            'transactions': transactions,
            'cluster_size': len(visited_addresses),
            'total_transactions': len(transactions),
            'analysis_scope': 'last_30_days_optimized'
        }
    
    async def detect_mixing_patterns(self, address: str) -> Dict:
        """
        Detect potential mixing service usage patterns
        Optimized for recent transactions to reduce API load
        """
        # Get recent transactions only to reduce API calls
        txs = await self.get_address_transactions(address, limit=50)
        
        mixing_indicators = {
            'multiple_small_outputs': 0,
            'round_number_amounts': 0,
            'high_input_output_ratio': 0,
            'timing_patterns': [],
            'common_amounts': {},
            'suspicious_score': 0.0,
            'analysis_scope': f'last_{len(txs)}_transactions'
        }
        
        # Focus on most recent transactions for fraud detection
        recent_txs = txs[:30]  # Analyze only 30 most recent transactions
        
        for tx in recent_txs:
            outputs = tx.get('outputs', [])
            inputs = tx.get('inputs', [])
            
            # Check for multiple small outputs (typical of mixers)
            if len(outputs) > 10:
                mixing_indicators['multiple_small_outputs'] += 1
            
            # Check for round number amounts (limit processing)
            for output in outputs[:20]:  # Limit to first 20 outputs
                value = output.get('value', 0)
                if value > 0:
                    # Check if amount is suspiciously round
                    if value % 100000000 == 0:  # Round Bitcoin amounts
                        mixing_indicators['round_number_amounts'] += 1
                    
                    # Track common amounts (limit storage)
                    amount_btc = value / 100000000
                    amount_key = f"{amount_btc:.8f}"
                    if len(mixing_indicators['common_amounts']) < 50:  # Limit tracked amounts
                        mixing_indicators['common_amounts'][amount_key] = mixing_indicators['common_amounts'].get(amount_key, 0) + 1
            
            # High input/output ratio
            if len(inputs) > 1 and len(outputs) > len(inputs) * 2:
                mixing_indicators['high_input_output_ratio'] += 1
            
            # Timing analysis (limited)
            confirmed_time = tx.get('confirmed')
            if confirmed_time and len(mixing_indicators['timing_patterns']) < 100:
                mixing_indicators['timing_patterns'].append(confirmed_time)
        
        # Calculate suspicious score
        total_txs = len(recent_txs)
        if total_txs > 0:
            score = (
                (mixing_indicators['multiple_small_outputs'] / total_txs) * 0.4 +
                (mixing_indicators['round_number_amounts'] / total_txs) * 0.3 +
                (mixing_indicators['high_input_output_ratio'] / total_txs) * 0.3
            )
            mixing_indicators['suspicious_score'] = min(score, 1.0)
        
        return mixing_indicators
    
    async def get_wallet_risk_indicators(self, address: str) -> Dict:
        """
        Get comprehensive risk indicators for a wallet address
        """
        try:
            # Get basic address info
            addr_info = await self.get_address_info(address)
            
            # Handle addresses with no transaction history
            if 'error' in addr_info or addr_info.get('n_tx', 0) == 0:
                logger.info(f"Address {address} has no transaction history or doesn't exist")
                return {
                    'address': address,
                    'basic_info': {
                        'balance': 0,
                        'total_received': 0,
                        'total_sent': 0,
                        'transaction_count': 0,
                        'balance_btc': 0,
                        'total_received_btc': 0,
                        'total_sent_btc': 0
                    },
                    'activity_patterns': {
                        'velocity': 0,
                        'turnover_ratio': 0,
                        'retention_ratio': 0
                    },
                    'mixing_analysis': {
                        'suspicious_score': 0,
                        'multiple_small_outputs': 0,
                        'round_number_amounts': 0,
                        'high_input_output_ratio': 0
                    },
                    'cluster_analysis': {
                        'cluster_size': 1,
                        'connected_addresses': 0
                    },
                    'risk_flags': {
                        'high_velocity': False,
                        'mixing_suspected': False,
                        'large_cluster': False,
                        'low_retention': False
                    },
                    'note': 'Address has no transaction history'
                }
            
            # Get mixing patterns
            mixing_analysis = await self.detect_mixing_patterns(address)
            
            # Get wallet cluster
            cluster_analysis = await self.analyze_wallet_cluster(address, depth=1)
            
            # Calculate age and activity patterns
            first_tx = addr_info.get('tx_url', '')
            total_received = addr_info.get('total_received', 0)
            total_sent = addr_info.get('total_sent', 0)
            balance = addr_info.get('balance', 0)
            tx_count = addr_info.get('n_tx', 0)
            
            # Risk indicators
            risk_indicators = {
                'address': address,
                'basic_info': {
                    'balance': balance,
                    'total_received': total_received,
                    'total_sent': total_sent,
                    'transaction_count': tx_count,
                    'balance_btc': balance / 100000000,
                    'total_received_btc': total_received / 100000000,
                    'total_sent_btc': total_sent / 100000000
                },
                'activity_patterns': {
                    'velocity': (total_sent + total_received) / max(tx_count, 1),
                    'turnover_ratio': total_sent / max(total_received, 1) if total_received > 0 else 0,
                    'retention_ratio': balance / max(total_received, 1) if total_received > 0 else 0
                },
                'mixing_analysis': mixing_analysis,
                'cluster_analysis': {
                    'cluster_size': cluster_analysis.get('cluster_size', 1),
                    'connected_addresses': len(cluster_analysis.get('address_graph', {}))
                },
                'risk_flags': {
                    'high_velocity': (total_sent + total_received) / max(tx_count, 1) > 10000000,  # > 0.1 BTC per tx
                    'mixing_suspected': mixing_analysis.get('suspicious_score', 0) > 0.5,
                    'large_cluster': cluster_analysis.get('cluster_size', 1) > 20,
                    'low_retention': balance / max(total_received, 1) < 0.1 if total_received > 0 else False
                }
            }
            
            return risk_indicators
            
        except Exception as e:
            logger.error(f"Error analyzing wallet {address}: {e}")
            return {'error': str(e), 'address': address}

# Example usage and testing
async def test_blockcypher_client():
    """Test the BlockCypher client with known addresses"""
    client = BlockCypherClient()
    
    # Test with a known Bitcoin address (this is Satoshi's genesis block address)
    test_address = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
    
    print(f"Testing BlockCypher client with address: {test_address}")
    
    # Get basic info
    info = await client.get_address_info(test_address)
    print(f"Address info: {json.dumps(info, indent=2)}")
    
    # Get risk indicators
    risk_indicators = await client.get_wallet_risk_indicators(test_address)
    print(f"Risk indicators: {json.dumps(risk_indicators, indent=2)}")

if __name__ == "__main__":
    asyncio.run(test_blockcypher_client())