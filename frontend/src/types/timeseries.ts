// Wallet time-series types for BitScan charts
export interface WalletTimeSeriesPoint {
  date: string;
  received_btc: number;
  sent_btc: number;
  net_btc: number;
  cumulative_balance_btc: number;
  tx_count: number;
}

export interface WalletTimeSeriesResponse {
  address: string;
  timeframe_days: number;
  points: WalletTimeSeriesPoint[];
  summary: {
    total_received_btc: number;
    total_sent_btc: number;
    net_change_btc: number;
    days: number;
    first_day?: string | null;
    last_day?: string | null;
    nonzero_days: number;
    unique_counterparties?: number;
  };
}
