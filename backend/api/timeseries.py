"""
Wallet time-series endpoints for BitScan
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import logging

from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import BaseModel, Field

# Reuse existing clients via lazy init like routes.py
import sys
from pathlib import Path as PathLib
backend_path = PathLib(__file__).parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from data.blockcypher_client import BlockCypherClient
from blockchain.analyzer import BlockchainAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter()

# Globals
blockcypher_client: Optional[BlockCypherClient] = None
blockchain_analyzer: Optional[BlockchainAnalyzer] = None


def initialize_services():
    global blockcypher_client, blockchain_analyzer
    if blockcypher_client is None:
        blockcypher_client = BlockCypherClient()
    if blockchain_analyzer is None:
        blockchain_analyzer = BlockchainAnalyzer()


class WalletTimeSeriesPoint(BaseModel):
    date: str
    received_btc: float
    sent_btc: float
    net_btc: float
    cumulative_balance_btc: float
    tx_count: int


class WalletTimeSeriesResponse(BaseModel):
    address: str
    timeframe_days: int
    points: List[WalletTimeSeriesPoint]
    summary: Dict[str, Any]


@router.get("/wallet/{address}/timeseries", response_model=WalletTimeSeriesResponse, tags=["Analysis"])
async def get_wallet_timeseries(
    address: str = Path(..., description="Bitcoin address to build time series for"),
    days: int = Query(default=90, ge=1, le=3650, description="Timeframe in days"),
    granularity: str = Query(default="day", regex="^(day|week|month|year)$", description="Aggregation granularity: day, week, month, or year")
):
    """
    Build per-day time series for a wallet address including received, sent, net and cumulative balance.
    Data is aggregated from recent transactions using BlockCypher client.
    """
    try:
        initialize_services()

        # Basic address validation similar to routes.py (allow testnet for dev)
        def _is_valid_bitcoin_address(addr: str) -> bool:
            return addr and (addr.startswith(("1", "3", "bc1", "bc1q", "bc1p", "tb1", "2", "m", "n")))

        if not _is_valid_bitcoin_address(address):
            raise HTTPException(status_code=400, detail="Invalid Bitcoin address format")

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        # Fetch recent transactions (full provides confirmed timestamp and I/O)
        txs = await blockcypher_client.get_address_full_transactions(address, limit=10000)

        def parse_time(t: Optional[str]):
            if not t:
                return None
            try:
                return datetime.fromisoformat(t.replace('Z', '+00:00'))
            except Exception:
                return None

        # Sort ascending by time
        txs_sorted = sorted(
            [tx for tx in txs if tx.get('confirmed')],
            key=lambda tx: parse_time(tx.get('confirmed')) or datetime.min.replace(tzinfo=timezone.utc)
        )

        # Key builder based on granularity
        def make_key(ts: datetime) -> str:
            if granularity == "year":
                return ts.strftime("%Y")
            elif granularity == "week":
                iso_year, iso_week, _ = ts.isocalendar()
                return f"{iso_year}-W{iso_week:02d}"
            elif granularity == "month":
                return ts.strftime("%Y-%m")
            else:
                return ts.date().isoformat()

        buckets = defaultdict(lambda: {"received": 0.0, "sent": 0.0, "tx_count": 0})
        counterparties = set()

        for tx in txs_sorted:
            ts = parse_time(tx.get('confirmed'))
            if ts is None or ts < cutoff:
                continue

            day_key = make_key(ts)

            # Sum outputs to this address (received)
            outputs = tx.get('outputs', []) or []
            out_sum_sats = 0
            for out in outputs:
                addrs = (out or {}).get('addresses', []) or []
                if address in addrs:
                    val = (out or {}).get('value', 0) or 0
                    out_sum_sats += int(val)
                else:
                    # If this address is in inputs, outputs are counterparties; captured below
                    pass

            # Sum inputs from this address (sent)
            inputs = tx.get('inputs', []) or []
            in_sum_sats = 0
            for inp in inputs:
                addrs = (inp or {}).get('addresses', []) or []
                if address in addrs:
                    val = (inp or {}).get('output_value')
                    if val is None:
                        prev_out = (inp or {}).get('prev_out') or {}
                        val = prev_out.get('value', 0)
                    in_sum_sats += int(val or 0)
                else:
                    # If this address is in outputs, inputs are counterparties; captured below
                    pass

            # Track counterparties: if we received in this tx, counterparties are input addresses; if we sent, counterparties are output addresses
            tx_input_addrs = set()
            for inp in inputs:
                for a in ((inp or {}).get('addresses', []) or []):
                    tx_input_addrs.add(a)

            tx_output_addrs = set()
            for out in outputs:
                for a in ((out or {}).get('addresses', []) or []):
                    tx_output_addrs.add(a)

            if out_sum_sats > 0:  # we received -> inputs are counterparties
                for a in tx_input_addrs:
                    if a and a != address:
                        counterparties.add(a)
            if in_sum_sats > 0:  # we sent -> outputs are counterparties
                for a in tx_output_addrs:
                    if a and a != address:
                        counterparties.add(a)

            received_btc = out_sum_sats / 1e8
            sent_btc = in_sum_sats / 1e8

            buckets[day_key]["received"] += received_btc
            buckets[day_key]["sent"] += sent_btc
            buckets[day_key]["tx_count"] += 1

        # Build ordered series based on granularity
        points: List[WalletTimeSeriesPoint] = []
        cumulative = 0.0

        def iter_keys():
            now = datetime.now(timezone.utc)
            if granularity == "year":
                # iterate last N years
                start_year = (now - timedelta(days=days)).year
                end_year = now.year
                for y in range(start_year, end_year + 1):
                    yield f"{y}"
            elif granularity == "week":
                # iterate ISO weeks between start and end
                start = now - timedelta(days=days)
                # Move start to the Monday of its week
                start_monday = start - timedelta(days=(start.weekday()))
                cur = start_monday
                end = now
                seen = set()
                while cur <= end:
                    iso_year, iso_week, _ = cur.isocalendar()
                    key = f"{iso_year}-W{iso_week:02d}"
                    if key not in seen:
                        seen.add(key)
                        yield key
                    cur += timedelta(days=7)
            elif granularity == "month":
                # iterate months between start and end
                start = now - timedelta(days=days)
                y, m = start.year, start.month
                end_y, end_m = now.year, now.month
                while (y < end_y) or (y == end_y and m <= end_m):
                    yield f"{y:04d}-{m:02d}"
                    m += 1
                    if m > 12:
                        m = 1
                        y += 1
            else:
                # daily over last N days
                for i in range(days, -1, -1):
                    yield (datetime.now(timezone.utc) - timedelta(days=i)).date().isoformat()

        for key in iter_keys():
            rec = float(buckets[key]["received"]) if key in buckets else 0.0
            snt = float(buckets[key]["sent"]) if key in buckets else 0.0
            net = rec - snt
            cumulative += net
            points.append(WalletTimeSeriesPoint(
                date=key,
                received_btc=round(rec, 8),
                sent_btc=round(snt, 8),
                net_btc=round(net, 8),
                cumulative_balance_btc=round(cumulative, 8),
                tx_count=int(buckets[key]["tx_count"]) if key in buckets else 0
            ))

        summary = {
            "total_received_btc": round(sum(p.received_btc for p in points), 8),
            "total_sent_btc": round(sum(p.sent_btc for p in points), 8),
            "net_change_btc": round(sum(p.net_btc for p in points), 8),
            "days": days,
            "first_day": points[0].date if points else None,
            "last_day": points[-1].date if points else None,
            "nonzero_days": sum(1 for p in points if p.tx_count > 0),
            "unique_counterparties": len(counterparties)
        }

        return WalletTimeSeriesResponse(
            address=address,
            timeframe_days=days,
            points=points,
            summary=summary
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building wallet timeseries for {address}: {e}")
        raise HTTPException(status_code=500, detail=f"Time series generation error: {str(e)}")
