// API Response Types for BitScan
export interface BitcoinAddress {
  address: string;
  risk_score: number;
  risk_level: 'MINIMAL' | 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' | 'UNKNOWN';
  is_flagged: boolean;
  confidence: number;
  fraud_probability?: number;
  risk_factors?: string[];
  positive_indicators?: string[];
}

export interface AnalysisSummary {
  transaction_count: number;
  total_received_btc: number;
  total_sent_btc: number;
  current_balance_btc: number;
  risk_indicators: number;
  network_centrality?: number;
  cluster_size?: number;
}

export interface ModelPerformance {
  ensemble_confidence: number;
  model_count: number;
  agreement_score: number;
}

export interface DataLimitations {
  rate_limit_detected: boolean;
  real_time_data: boolean;
  api_status: string;
  note?: string;
  description?: string;
  accuracy_note?: string;
  recommendation?: string;
}

export interface DetailedAnalysis {
  blockchain_analysis: any;
  ml_prediction: any;
}

export interface AnalysisResponse {
  address: string;
  risk_score: number;
  risk_level: string;
  is_flagged: boolean;
  confidence: number;
  fraud_probability?: number;
  risk_factors?: string[];
  positive_indicators?: string[];
  analysis_summary: AnalysisSummary;
  model_performance?: ModelPerformance;
  data_limitations?: DataLimitations;
  detailed_analysis?: DetailedAnalysis;
  timestamp: string;
}

export interface SystemStats {
  total_analyses_performed: string | number;
  unique_addresses_analyzed: string | number;
  fraud_detection_rate: number;
  average_analysis_time: string | number;
  system_status: string;
  api_version?: string;
  last_updated?: string;
}

export interface ApiError {
  error: string;
  message: string;
  detail?: string;
  status_code?: number;
}

// Form Types
export interface AddressAnalysisForm {
  address: string;
  include_detailed: boolean;
}

// UI State Types
export interface LoadingState {
  analyzing: boolean;
  loadingStats: boolean;
}

export interface UIAlert {
  type: 'success' | 'error' | 'warning' | 'info';
  message: string;
  id: string;
}