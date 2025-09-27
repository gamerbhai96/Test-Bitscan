import { useEffect, useMemo, useState } from 'react';
import type { FC } from 'react';
import { Box, Typography, CircularProgress, Paper } from '@mui/material';
import { 
  ResponsiveContainer, 
  XAxis, 
  YAxis, 
  Tooltip as RechartsTooltip, 
  CartesianGrid, 
  Area, 
  Legend, 
  AreaChart
} from 'recharts';
import { BitScanAPI } from '../services/api';

type Granularity = 'day' | 'week' | 'month' | 'year';

interface WalletTimeSeriesPoint {
  date: string;
  received_btc: number;
  sent_btc: number;
  cumulative_balance_btc: number;
  formattedDate?: string;
}

interface WalletTimeSeriesResponse {
  points: WalletTimeSeriesPoint[];
  summary?: {
    total_received_btc: number;
    total_sent_btc: number;
    net_change_btc: number;
  };
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{
    name: string;
    value: number | string;
    color: string;
    payload: any;
  }>;
  label?: string;
  dark?: boolean;
}

interface Props {
  address: string;
  defaultDays?: number;
  defaultGranularity?: Granularity;
  dark?: boolean;
}

type TooltipItem = {
  name: string;
  value: number | string;
  color: string;
  payload: any;
};

const CustomTooltip: FC<CustomTooltipProps> = ({
  active,
  payload,
  label,
  dark = false
}) => {
  if (active && payload && payload.length) {
    return (
      <Box sx={{ 
        p: 1, 
        backgroundColor: dark ? '#1e1e1e' : '#fff',
        border: `1px solid ${dark ? '#333' : '#ddd'}`,
        boxShadow: '0px 2px 4px rgba(0,0,0,0.1)'
      }}>
        <Typography variant="subtitle2" sx={{ mb: 1, color: dark ? '#fff' : '#333' }}>
          {String(label)}
        </Typography>
        {payload.map((item: TooltipItem, index: number) => (
          <Box key={`tooltip-item-${index}`} sx={{ display: 'flex', justifyContent: 'space-between', gap: 2 }}>
            <Typography variant="body2" sx={{ color: item.color }}>
              {item.name}:
            </Typography>
            <Typography variant="body2" sx={{ color: dark ? '#fff' : '#333' }}>
              {Number(item.value).toFixed(8)} BTC
            </Typography>
          </Box>
        ))}
      </Box>
    );
  }
  return null;
};

const WalletCredibilityChart: FC<Props> = ({ 
  address, 
  defaultDays = 90, 
  defaultGranularity = 'week', 
  dark = false 
}) => {
  // State management
  const [timeframe, setTimeframe] = useState<{
    days: number;
    granularity: Granularity;
  }>({ 
    days: defaultDays, 
    granularity: defaultGranularity 
  });

  const [data, setData] = useState<WalletTimeSeriesResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [activePreset, setActivePreset] = useState<string>('90d');

  // Calculate optimal granularity based on time range
  const getOptimalGranularity = (days: number): Granularity => {
    if (days <= 30) return 'day';
    if (days <= 90) return 'week';
    if (days <= 180) return 'week';
    return 'month';
  };

  // Apply preset with optimal granularity
  const applyPreset = (key: string) => {
    let days: number;
    switch (key) {
      case '30d': days = 30; break;
      case '90d': days = 90; break;
      case '6m': days = 180; break;
      case '1y': days = 365; break;
      default: days = 90;
    }
    
    const granularity = getOptimalGranularity(days);
    setTimeframe({ days, granularity });
    setActivePreset(key);
  };

  // Memoize the processed chart data
  const chartData = useMemo(() => {
    if (!data?.points?.length) return [];
    
    // Ensure data is sorted by date
    const sorted = [...data.points].sort((a, b) => 
      new Date(a.date).getTime() - new Date(b.date).getTime()
    );

    // Process data points for consistent display
    return sorted.map(point => ({
      ...point,
      // Ensure numeric values
      received_btc: Number(point.received_btc) || 0,
      sent_btc: Number(point.sent_btc) || 0,
      cumulative_balance_btc: Number(point.cumulative_balance_btc) || 0,
      // Format date based on granularity
      formattedDate: new Date(point.date).toLocaleDateString('en-US', {
        year: 'numeric',
        month: timeframe.granularity === 'month' ? 'short' : 'numeric',
        day: timeframe.granularity === 'day' ? 'numeric' : undefined,
      })
    }));
  }, [data, timeframe.granularity]);
  
  // Calculate chart width based on number of data points
  const chartWidth = useMemo(() => {
    if (!chartData.length) return '100%';
    // Base width for each data point in pixels
    const baseWidth = timeframe.granularity === 'day' ? 60 : 100;
    const minWidth = 1200; // Minimum width to ensure readability
    return `${Math.max(chartData.length * baseWidth, minWidth)}px`;
  }, [chartData.length, timeframe.granularity]);

  // Fetch data when address or timeframe changes
  useEffect(() => {
    let mounted = true;
    const controller = new AbortController();

    const loadData = async () => {
      if (!address) return;
      
      setLoading(true);
      setError(null);
      
      try {
        const res = await BitScanAPI.getWalletTimeSeries(
          address, 
          timeframe.days, 
          timeframe.granularity
        );
        
        if (mounted) {
          setData(res);
        }
      } catch (error: any) {
        if (error.name !== 'AbortError' && mounted) {
          console.error('Error loading time series:', error);
          setError(error?.message || 'Failed to load time series data');
        }
      } finally {
        if (mounted) setLoading(false);
      }
    };

    loadData();
    
    // Cleanup function to cancel pending requests
    return () => {
      mounted = false;
      controller.abort();
    };
  }, [address, timeframe.days, timeframe.granularity]);

  // Render loading and error states
  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3, textAlign: 'center', color: 'error.main' }}>
        <Typography>Error loading chart data: {error}</Typography>
      </Box>
    );
  }

  if (!chartData.length) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography>No data available for the selected time range</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%' }}>
      {/* Timeframe selector */}
      <Box sx={{ display: 'flex', gap: 1, mb: 2, overflowX: 'auto', pb: 1 }}>
        {['30d', '90d', '6m', '1y'].map((preset) => (
          <Paper
            key={preset}
            elevation={0}
            onClick={() => applyPreset(preset)}
            sx={{
              px: 2,
              py: 1,
              borderRadius: 1,
              cursor: 'pointer',
              backgroundColor: activePreset === preset 
                ? (dark ? 'rgba(99, 102, 241, 0.2)' : 'rgba(99, 102, 241, 0.1)')
                : (dark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)'),
              color: activePreset === preset 
                ? (dark ? '#818cf8' : '#4f46e5')
                : (dark ? 'rgba(255, 255, 255, 0.7)' : 'rgba(0, 0, 0, 0.7)'),
              border: `1px solid ${activePreset === preset 
                ? (dark ? 'rgba(99, 102, 241, 0.5)' : 'rgba(99, 102, 241, 0.3)') 
                : (dark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)')}`,
              '&:hover': {
                backgroundColor: dark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)',
              },
              whiteSpace: 'nowrap',
              fontSize: '0.875rem',
              fontWeight: 500,
              transition: 'all 0.2s',
            }}
          >
            {preset}
          </Paper>
        ))}
      </Box>

      {/* Stats summary */}
      {data?.summary && (
        <Box sx={{ display: 'flex', gap: 2, mb: 3, flexWrap: 'wrap' }}>
          <Paper elevation={0} sx={{ 
            p: 2, 
            flex: 1, 
            minWidth: 150, 
            borderRadius: 2, 
            border: '1px solid', 
            borderColor: dark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.08)' 
          }}>
            <Typography variant="caption" sx={{ opacity: 0.7 }}>Total Received (BTC)</Typography>
            <Typography variant="h6" sx={{ fontWeight: 700, color: '#10b981' }}>{(data.summary.total_received_btc || 0).toFixed(8)}</Typography>
          </Paper>
          <Paper elevation={0} sx={{ 
            p: 2, 
            flex: 1, 
            minWidth: 150, 
            borderRadius: 2, 
            border: '1px solid', 
            borderColor: dark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.08)' 
          }}>
            <Typography variant="caption" sx={{ opacity: 0.7 }}>Total Sent (BTC)</Typography>
            <Typography variant="h6" sx={{ fontWeight: 700, color: '#ef4444' }}>{(data.summary.total_sent_btc || 0).toFixed(8)}</Typography>
          </Paper>
          <Paper elevation={0} sx={{ 
            p: 2, 
            flex: 1, 
            minWidth: 150, 
            borderRadius: 2, 
            border: '1px solid', 
            borderColor: dark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.08)' 
          }}>
            <Typography variant="caption" sx={{ opacity: 0.7 }}>Net Change (BTC)</Typography>
            <Typography 
              variant="h6" 
              sx={{ 
                fontWeight: 700, 
                color: (data.summary.net_change_btc || 0) >= 0 ? '#10b981' : '#ef4444' 
              }}
            >
              {(data.summary.net_change_btc || 0) >= 0 ? '+' : ''}{(data.summary.net_change_btc || 0).toFixed(8)}
            </Typography>
          </Paper>
        </Box>
      )}

      {/* Chart container */}
      <Box sx={{ 
        width: '100%', 
        overflowX: 'auto', 
        '&::-webkit-scrollbar': { 
          height: 8,
          backgroundColor: dark ? 'rgba(0,0,0,0.2)' : 'rgba(0,0,0,0.05)' 
        }, 
        '&::-webkit-scrollbar-thumb': { 
          backgroundColor: dark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)', 
          borderRadius: 4,
          '&:hover': {
            backgroundColor: dark ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.3)'
          }
        } 
      }}>
        <Box sx={{ 
          height: 400, 
          minWidth: chartWidth, 
          pr: 2
        }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart 
              data={chartData} 
              margin={{ top: 10, right: 20, bottom: 10, left: 0 }}
            >
              <defs>
                <linearGradient id="colorReceived" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0.1} />
                </linearGradient>
                <linearGradient id="colorSent" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0.1} />
                </linearGradient>
                <linearGradient id="colorBalance" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.1} />
                </linearGradient>
              </defs>
              <XAxis 
                dataKey="formattedDate" 
                tick={{ 
                  fontSize: 12,
                  fill: dark ? 'rgba(255, 255, 255, 0.7)' : 'rgba(0, 0, 0, 0.7)'
                }}
                tickCount={Math.min(chartData.length, 12)}
                minTickGap={20}
                axisLine={{ stroke: dark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)' }}
                tickLine={{ stroke: dark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)' }}
              />
              <YAxis 
                tickFormatter={(value) => `${value} BTC`}
                tick={{ 
                  fontSize: 12,
                  fill: dark ? 'rgba(255, 255, 255, 0.7)' : 'rgba(0, 0, 0, 0.7)'
                }}
                width={80}
                axisLine={{ stroke: dark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)' }}
                tickLine={{ stroke: dark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)' }}
              />
              <CartesianGrid 
                strokeDasharray="3 3" 
                vertical={false} 
                stroke={dark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'} 
              />
              <RechartsTooltip 
                content={({ active, payload, label }) => (
                  <CustomTooltip 
                    active={active}
                    payload={payload}
                    label={String(label || '')}
                    dark={dark}
                  />
                )}
                cursor={{ 
                  stroke: dark ? 'rgba(255, 255, 255, 0.2)' : 'rgba(0, 0, 0, 0.1)',
                  strokeWidth: 1
                }}
                wrapperStyle={{ 
                  zIndex: 100,
                  backgroundColor: dark ? '#1e1e1e' : '#fff',
                  border: `1px solid ${dark ? '#333' : '#ddd'}`,
                  borderRadius: 4,
                  padding: '8px 12px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)'
                }}
              />
              <Legend 
                wrapperStyle={{
                  paddingTop: '20px',
                  color: dark ? 'rgba(255, 255, 255, 0.7)' : 'rgba(0, 0, 0, 0.7)'
                }}
              />
              <Area 
                type="monotone" 
                dataKey="received_btc" 
                name="Received BTC" 
                stroke="#10b981" 
                fillOpacity={1} 
                fill="url(#colorReceived)" 
                activeDot={{ 
                  r: 6,
                  stroke: dark ? '#1e1e1e' : '#fff',
                  strokeWidth: 2
                }}
                strokeWidth={2}
              />
              <Area 
                type="monotone" 
                dataKey="sent_btc" 
                name="Sent BTC" 
                stroke="#ef4444" 
                fillOpacity={1} 
                fill="url(#colorSent)" 
                activeDot={{ 
                  r: 6,
                  stroke: dark ? '#1e1e1e' : '#fff',
                  strokeWidth: 2
                }}
                strokeWidth={2}
              />
              <Area 
                type="monotone" 
                dataKey="cumulative_balance_btc" 
                name="Balance BTC" 
                stroke="#8b5cf6" 
                fillOpacity={0.3} 
                fill="url(#colorBalance)" 
                activeDot={{ 
                  r: 6,
                  stroke: dark ? '#1e1e1e' : '#fff',
                  strokeWidth: 2
                }}
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </Box>
      </Box>
    </Box>
  );
};

export default WalletCredibilityChart;
