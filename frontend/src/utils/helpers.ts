/**
 * Utility functions for Bitcoin address validation and formatting
 */

/**
 * Validate Bitcoin address format
 */
export const isValidBitcoinAddress = (address: string): boolean => {
  if (!address || typeof address !== 'string') return false;
  
  // Length check
  if (address.length < 26 || address.length > 62) return false;
  
  // Character set check for different address types
  const legacyPattern = /^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$/;
  const segwitPattern = /^bc1[a-z0-9]{39,59}$/;
  const testnetPattern = /^[mn2][a-km-zA-HJ-NP-Z1-9]{25,34}$/;
  const testnetSegwitPattern = /^tb1[a-z0-9]{39,59}$/;
  
  return legacyPattern.test(address) || 
         segwitPattern.test(address) || 
         testnetPattern.test(address) ||
         testnetSegwitPattern.test(address);
};

/**
 * Get risk color based on risk level
 */
export const getRiskColor = (riskLevel: string): string => {
  const colors: Record<string, string> = {
    'MINIMAL': '#4caf50',     // Green
    'VERY_LOW': '#8bc34a',    // Light Green
    'LOW': '#2196f3',         // Blue
    'MEDIUM': '#ff9800',      // Orange
    'ELEVATED': '#f44336',    // Red
    'HIGH': '#d32f2f',        // Dark Red
    'CRITICAL': '#b71c1c',    // Very Dark Red
    'UNKNOWN': '#757575'      // Grey
  };
  return colors[riskLevel] || colors['UNKNOWN'];
};

/**
 * Format BTC amounts with appropriate decimal places
 */
export const formatBTC = (amount: number): string => {
  if (amount === 0) return '0.0000';
  if (amount < 0.0001) return amount.toFixed(8);
  if (amount < 0.01) return amount.toFixed(6);
  if (amount < 1) return amount.toFixed(4);
  return amount.toFixed(4);
};

/**
 * Format large numbers with appropriate suffixes
 */
export const formatNumber = (num: number | string): string => {
  if (typeof num === 'string') return num;
  
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + 'M';
  } else if (num >= 1000) {
    return (num / 1000).toFixed(1) + 'K';
  }
  return num.toString();
};

/**
 * Generate a unique ID for UI elements
 */
export const generateId = (): string => {
  return Math.random().toString(36).substr(2, 9);
};

/**
 * Truncate address for display
 */
export const truncateAddress = (address: string, startChars: number = 8, endChars: number = 8): string => {
  if (address.length <= startChars + endChars + 3) {
    return address;
  }
  return `${address.slice(0, startChars)}...${address.slice(-endChars)}`;
};

/**
 * Copy text to clipboard
 */
export const copyToClipboard = async (text: string): Promise<boolean> => {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (error) {
    console.error('Failed to copy to clipboard:', error);
    return false;
  }
};

/**
 * Format timestamp for display
 */
export const formatTimestamp = (timestamp: string): string => {
  try {
    const date = new Date(timestamp);
    return date.toLocaleString();
  } catch (error) {
    return timestamp;
  }
};

/**
 * Debounce function for performance optimization
 */
export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  delay: number
): ((...args: Parameters<T>) => void) => {
  let timeoutId: number;
  
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = window.setTimeout(() => func(...args), delay);
  };
};

/**
 * Get risk level badge configuration
 */
export const getRiskBadgeConfig = (riskLevel: string) => {
  const configs: Record<string, { color: string; icon: string; severity: 'success' | 'info' | 'warning' | 'error' }> = {
    'MINIMAL': { color: 'success', icon: '✓', severity: 'success' as const },
    'VERY_LOW': { color: 'success', icon: '✓', severity: 'success' as const },
    'LOW': { color: 'info', icon: 'ℹ', severity: 'info' as const },
    'MEDIUM': { color: 'warning', icon: '⚠', severity: 'warning' as const },
    'ELEVATED': { color: 'warning', icon: '⚠', severity: 'warning' as const },
    'HIGH': { color: 'error', icon: '⚠', severity: 'error' as const },
    'CRITICAL': { color: 'error', icon: '🚨', severity: 'error' as const },
    'UNKNOWN': { color: 'default', icon: '?', severity: 'info' as const }
  };
  
  return configs[riskLevel] || configs['UNKNOWN'];
};