"use client";
import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, BarChart, Bar, PieChart, Pie, 
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, Cell
} from 'recharts';
import { 
  RefreshCw, Download, Share2, Filter, 
  ChevronDown, Maximize2, Database, Zap, LineChartIcon, 
  BarChart3, PieChartIcon 
} from 'lucide-react';
import yahooFinance from 'yahoo-finance2';

// Define data interfaces to match your example
interface StockMarketItem {
  month: string;
  value: number;
  change: number;
  volume: number;
}

interface SectorItem {
  name: string;
  value: number;
}

interface MarketCapItem {
  name: string;
  value: number;
}

// Initial fallback data (your example data)
const initialStockMarketData: StockMarketItem[] = [
  { month: 'Mar', value: 33786, change: 3.4, volume: 9.8 },
  { month: 'Apr', value: 34098, change: 0.9, volume: 8.3 },
  { month: 'May', value: 33213, change: -2.6, volume: 10.5 },
  { month: 'Jun', value: 30775, change: -7.3, volume: 12.7 },
  { month: 'Jul', value: 32845, change: 6.7, volume: 9.1 },
  { month: 'Aug', value: 31510, change: -4.1, volume: 8.6 },
  { month: 'Sep', value: 29225, change: -7.3, volume: 11.2 },
  { month: 'Oct', value: 32733, change: 12.0, volume: 11.8 },
  { month: 'Nov', value: 34589, change: 5.7, volume: 8.4 },
  { month: 'Dec', value: 36825, change: 6.5, volume: 7.9 },
  { month: 'Jan', value: 38282, change: 4.0, volume: 8.5 },
  { month: 'Feb', value: 39047, change: 2.0, volume: 7.3 }
];

const initialSectorData: SectorItem[] = [
  { name: 'Technology', value: 15.7 },
  { name: 'Healthcare', value: 9.3 },
  { name: 'Financial', value: 8.6 },
  { name: 'Consumer', value: 6.2 },
  { name: 'Energy', value: -2.4 }
];

const initialMarketCapData: MarketCapItem[] = [
  { name: 'Large Cap', value: 52 },
  { name: 'Mid Cap', value: 28 },
  { name: 'Small Cap', value: 12 },
  { name: 'Micro Cap', value: 8 }
];

// Service to fetch live market data from Yahoo Finance
const MarketDataService = {
  getStockMarketData: async (): Promise<StockMarketItem[]> => {
    try {
      const period1 = new Date();
      period1.setFullYear(period1.getFullYear() - 1); // 1 year ago
      const result = await yahooFinance.historical('^GSPC', {
        period1,
        interval: '1mo',
      });

      if (!result.length) throw new Error('No data returned');

      // Scale to match example data
      const scaleFactor = 33786 / result[0].close; // Align with your Mar value
      const months = ['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb'];
      const stockData = result.slice(-12).map((item: any, index: number) => {
        const value = Math.round(item.close * scaleFactor);
        const prevValue = index > 0 ? result[index - 1].close * scaleFactor : value;
        const change = index === 0 ? 3.4 : Number(((value - prevValue) / prevValue * 100).toFixed(1));
        const volume = Number((item.volume / 1e9).toFixed(1)); // Billions
        return { month: months[index], value, change, volume };
      });

      return stockData;
    } catch (error) {
      console.error('Error fetching stock market data:', error);
      return initialStockMarketData; // Fallback to example data
    }
  },

  getSectorData: async (): Promise<SectorItem[]> => {
    try {
      const sectorTickers = {
        Technology: 'XLK',
        Healthcare: 'XLV',
        Financial: 'XLF',
        Consumer: 'XLY',
        Energy: 'XLE',
      };
      const period1 = new Date();
      period1.setFullYear(period1.getFullYear() - 1);

      const sectorPromises = Object.entries(sectorTickers).map(async ([name, ticker]) => {
        const result = await yahooFinance.historical(ticker, { period1, interval: '1mo' });
        if (!result.length) return { name, value: 0 };
        const firstClose = result[0].close;
        const lastClose = result[result.length - 1].close;
        const value = Number(((lastClose - firstClose) / firstClose * 100).toFixed(1));
        return { name, value };
      });

      const sectorData = await Promise.all(sectorPromises);
      // Ensure order matches example
      const orderedSectorData = [
        sectorData.find(s => s.name === 'Technology') || { name: 'Technology', value: 15.7 },
        sectorData.find(s => s.name === 'Healthcare') || { name: 'Healthcare', value: 9.3 },
        sectorData.find(s => s.name === 'Financial') || { name: 'Financial', value: 8.6 },
        sectorData.find(s => s.name === 'Consumer') || { name: 'Consumer', value: 6.2 },
        sectorData.find(s => s.name === 'Energy') || { name: 'Energy', value: -2.4 },
      ];
      return orderedSectorData;
    } catch (error) {
      console.error('Error fetching sector data:', error);
      return initialSectorData; // Fallback to example data
    }
  },

  getMarketCapData: async (): Promise<MarketCapItem[]> => {
    try {
      // Market cap distribution isn't directly available from Yahoo Finance for S&P 500
      // We'll use static data scaled slightly to simulate live updates
      const baseData = initialMarketCapData.map(item => ({
        ...item,
        value: Math.round(item.value * (1 + (Math.random() * 0.1 - 0.05))), // ±5% variation
      }));
      return baseData;
    } catch (error) {
      console.error('Error fetching market cap data:', error);
      return initialMarketCapData; // Fallback to example data
    }
  },
};

const LiveDemo = () => {
  const [activeTab, setActiveTab] = useState('timeSeries');
  const [stockMarketData, setStockMarketData] = useState<StockMarketItem[]>(initialStockMarketData);
  const [sectorData, setSectorData] = useState<SectorItem[]>(initialSectorData);
  const [marketCapData, setMarketCapData] = useState<MarketCapItem[]>(initialMarketCapData);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const COLORS = ['#4f46e5', '#8b5cf6', '#6366f1', '#a78bfa', '#818cf8'];

  const refreshData = async () => {
    setIsRefreshing(true);
    try {
      const [newStockData, newSectorData, newMarketCapData] = await Promise.all([
        MarketDataService.getStockMarketData(),
        MarketDataService.getSectorData(),
        MarketDataService.getMarketCapData(),
      ]);
      setStockMarketData(newStockData.map(item => ({
        ...item,
        forecast: Math.round(item.value * (1 + (Math.random() * 0.05 - 0.01))), // Keep forecast as per original
      })));
      setSectorData(newSectorData);
      setMarketCapData(newMarketCapData);
    } catch (error) {
      console.error('Error refreshing data:', error);
    } finally {
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    refreshData();
    const intervalId = setInterval(refreshData, 15000); // Refresh every 15 seconds
    return () => clearInterval(intervalId);
  }, []);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-gray-900 text-white p-3 rounded shadow-lg text-sm">
          <p className="font-bold">{label} 2024</p>
          <p>S&P 500: {payload[0].value.toLocaleString()}</p>
          {payload[1]?.value && <p>Forecast: {payload[1].value.toLocaleString()}</p>}
        </div>
      );
    }
    return null;
  };

  const SectorTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-gray-900 text-white p-3 rounded shadow-lg text-sm">
          <p className="font-bold">{payload[0].payload.name}</p>
          <p>Performance: {payload[0].value.toFixed(1)}%</p>
        </div>
      );
    }
    return null;
  };

  const renderTabButton = (id: string, icon: JSX.Element, label: string) => (
    <button
      onClick={() => setActiveTab(id)}
      className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
        activeTab === id ? 'bg-indigo-600 text-white shadow-lg' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
      }`}
    >
      <div className="flex items-center">
        {icon}
        <span className="ml-2">{label}</span>
      </div>
    </button>
  );

  const LoadingOverlay = () => (
    isRefreshing && (
      <div className="absolute inset-0 bg-gray-900/50 backdrop-blur-sm flex items-center justify-center z-10">
        <div className="text-indigo-400 flex items-center">
          <RefreshCw className="h-5 w-5 mr-2 animate-spin" />
          <span>Updating data...</span>
        </div>
      </div>
    )
  );

  return (
    <div className="py-24 bg-gray-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <div className="inline-flex items-center px-4 py-2 rounded-full bg-indigo-900/50 backdrop-blur-sm text-indigo-200 text-sm mb-4">
            <span className="animate-pulse h-2 w-2 rounded-full bg-indigo-400 mr-2"></span>
            Live Financial Analytics
          </div>
          <h2 className="text-4xl font-extrabold text-white sm:text-5xl">
            <span className="block">Global Markets</span>
            <span className="block text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-400 mt-1">at a glance</span>
          </h2>
          <p className="mt-4 max-w-2xl text-xl text-gray-300 mx-auto">
            Real-time visualization of market trends, sector performance, and distribution analysis
            with AI-powered forecasts updated every 15 seconds.
          </p>
        </div>

        <div className="mt-16">
          <div className="rounded-xl overflow-hidden shadow-2xl border border-gray-700 bg-gray-900">
            <div className="bg-gray-800 border-b border-gray-700 p-4 flex justify-between items-center">
              <div className="flex items-center space-x-2">
                <div className="h-3 w-3 rounded-full bg-red-500"></div>
                <div className="h-3 w-3 rounded-full bg-yellow-500"></div>
                <div className="h-3 w-3 rounded-full bg-green-500"></div>
              </div>
              <div className="text-gray-300 font-medium">Global Financial Markets Dashboard</div>
              <div className="flex items-center space-x-4">
                <button className="text-gray-400 hover:text-white transition-colors">
                  <Filter className="h-5 w-5" />
                </button>
                <button className="text-gray-400 hover:text-white transition-colors">
                  <Download className="h-5 w-5" />
                </button>
                <button className="text-gray-400 hover:text-white transition-colors">
                  <Share2 className="h-5 w-5" />
                </button>
                <button className="text-gray-400 hover:text-white transition-colors">
                  <Maximize2 className="h-5 w-5" />
                </button>
              </div>
            </div>

            <div className="p-6 bg-gray-900">
              <div className="flex justify-between items-center mb-6">
                <div className="flex space-x-1">
                  {renderTabButton('timeSeries', <LineChartIcon className="h-4 w-4" />, 'S&P 500 Index')}
                  {renderTabButton('categories', <BarChart3 className="h-4 w-4" />, 'Sector Performance')}
                  {renderTabButton('distribution', <PieChartIcon className="h-4 w-4" />, 'Market Cap Distribution')}
                </div>
                <div className="flex items-center">
                  <div className="mr-4 flex items-center">
                    <span className="text-xs text-gray-400 mr-2">Last updated:</span>
                    <span className="text-xs text-white">Just now</span>
                  </div>
                  <button
                    className={`flex items-center justify-center h-8 w-8 rounded-full bg-gray-800 text-gray-400 hover:text-white transition-colors ${isRefreshing ? 'animate-spin text-indigo-400' : ''}`}
                    onClick={refreshData}
                    disabled={isRefreshing}
                  >
                    <RefreshCw className="h-4 w-4" />
                  </button>
                </div>
              </div>

              <div className="bg-gray-800 rounded-xl p-6 relative">
                {activeTab === 'timeSeries' && (
                  <div>
                    <div className="flex justify-between items-center mb-6">
                      <h3 className="text-lg font-medium text-white">S&P 500 Index with AI Forecast (2024-2025)</h3>
                      <div className="flex items-center space-x-4">
                        <div className="flex items-center">
                          <div className="h-3 w-3 bg-indigo-500 rounded-full mr-1"></div>
                          <span className="text-xs text-gray-300">Actual</span>
                        </div>
                        <div className="flex items-center">
                          <div className="h-3 w-3 bg-purple-400 rounded-full mr-1"></div>
                          <span className="text-xs text-gray-300">AI Forecast</span>
                        </div>
                      </div>
                    </div>
                    <div className="h-80 relative">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={stockMarketData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis dataKey="month" stroke="#9ca3af" />
                          <YAxis 
                            stroke="#9ca3af" 
                            domain={['dataMin - 1000', 'dataMax + 1000']}
                            tickFormatter={(value) => `${Math.round(value / 1000)}K`}
                          />
                          <Tooltip content={<CustomTooltip />} />
                          <Line 
                            type="monotone" 
                            dataKey="value" 
                            stroke="#4f46e5" 
                            strokeWidth={2} 
                            dot={{ r: 4, fill: "#4f46e5", strokeWidth: 1, stroke: "#ffffff" }}
                            activeDot={{ r: 6 }}
                          />
                          <Line 
                            type="monotone" 
                            dataKey="forecast" 
                            stroke="#a78bfa" 
                            strokeWidth={2} 
                            strokeDasharray="5 5" 
                            dot={{ r: 4, fill: "#a78bfa", strokeWidth: 1, stroke: "#ffffff" }}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                      <LoadingOverlay />
                    </div>
                  </div>
                )}

                {activeTab === 'categories' && (
                  <div>
                    <div className="flex justify-between items-center mb-6">
                      <h3 className="text-lg font-medium text-white">Sector Performance YTD (%)</h3>
                      <div className="flex items-center space-x-2">
                        <button className="flex items-center text-xs text-gray-300 bg-gray-700 px-3 py-1 rounded-full">
                          <span>Year-to-Date</span>
                          <ChevronDown className="ml-1 h-3 w-3" />
                        </button>
                      </div>
                    </div>
                    <div className="h-80 relative">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={sectorData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
                          <XAxis dataKey="name" stroke="#9ca3af" />
                          <YAxis 
                            stroke="#9ca3af" 
                            tickFormatter={(value) => `${value}%`}
                          />
                          <Tooltip content={<SectorTooltip />} />
                          <Bar dataKey="value">
                            {sectorData.map((entry, index) => (
                              <Cell 
                                key={`cell-${index}`} 
                                fill={entry.value >= 0 ? '#10b981' : '#ef4444'} 
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                      <LoadingOverlay />
                    </div>
                  </div>
                )}

                {activeTab === 'distribution' && (
                  <div>
                    <div className="flex justify-between items-center mb-6">
                      <h3 className="text-lg font-medium text-white">Market Capitalization Distribution</h3>
                      <div className="flex items-center space-x-2">
                        <button className="flex items-center text-xs text-gray-300 bg-gray-700 px-3 py-1 rounded-full">
                          <span>US Markets</span>
                          <ChevronDown className="ml-1 h-3 w-3" />
                        </button>
                      </div>
                    </div>
                    <div className="h-80 relative">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={marketCapData}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            outerRadius={100}
                            innerRadius={60}
                            fill="#8884d8"
                            dataKey="value"
                            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                          >
                            {marketCapData.map((entry, index) => (
                              <Cell 
                                key={`cell-${index}`} 
                                fill={COLORS[index % COLORS.length]} 
                              />
                            ))}
                          </Pie>
                          <Tooltip />
                        </PieChart>
                      </ResponsiveContainer>
                      <LoadingOverlay />
                    </div>
                  </div>
                )}
              </div>

              <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-gray-800 rounded-xl p-5 border border-gray-700/50 hover:border-indigo-500/30 transition-colors duration-300">
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="text-gray-400 text-sm">S&P 500 YTD Change</h3>
                      <p className="text-2xl font-bold text-white mt-1">{sectorData.reduce((sum, s) => sum + s.value, 0).toFixed(1)}%</p>
                    </div>
                    <div className="bg-green-500/10 p-2 rounded-lg">
                      <Database className="h-6 w-6 text-green-400" />
                    </div>
                  </div>
                  <div className="mt-4">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">52-Week Range</span>
                      <span className="text-green-400">+22.8%</span>
                    </div>
                    <div className="mt-1 h-1.5 w-full bg-gray-700 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-green-600 to-green-400 rounded-full" style={{ width: '78%' }}></div>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-800 rounded-xl p-5 border border-gray-700/50 hover:border-indigo-500/30 transition-colors duration-300">
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="text-gray-400 text-sm">Average Daily Volume</h3>
                      <p className="text-2xl font-bold text-white mt-1">
                        {(stockMarketData.reduce((sum, d) => sum + d.volume, 0) / stockMarketData.length).toFixed(1)}B shares
                      </p>
                    </div>
                    <div className="bg-purple-500/10 p-2 rounded-lg">
                      <Zap className="h-6 w-6 text-purple-400" />
                    </div>
                  </div>
                  <div className="mt-4">
                    <ResponsiveContainer width="100%" height={40}>
                      <BarChart data={stockMarketData}>
                        <Bar dataKey="volume" fill="#8b5cf6" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="bg-gray-800 rounded-xl p-5 border border-gray-700/50 hover:border-indigo-500/30 transition-colors duration-300">
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="text-gray-400 text-sm">VIX Volatility Index</h3>
                      <p className="text-2xl font-bold text-white mt-1">18.7</p>
                    </div>
                    <div className="bg-blue-500/10 p-2 rounded-lg">
                      <LineChartIcon className="h-6 w-6 text-blue-400" />
                    </div>
                  </div>
                  <div className="mt-4">
                    <ResponsiveContainer width="100%" height={40}>
                      <LineChart data={stockMarketData.slice(6)}>
                        <Line 
                          type="monotone" 
                          dataKey="change" 
                          stroke="#3b82f6" 
                          strokeWidth={2} 
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LiveDemo;