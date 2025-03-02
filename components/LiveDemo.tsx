"use client";
import React, { useState, useEffect, useRef } from 'react';
import { LineChart, BarChart3, PieChart, RefreshCw, Download, Share2, Filter, ChevronDown, Maximize2, Database, Zap } from 'lucide-react';

// Mock data for the charts
const generateTimeSeriesData = () => {
  const now = new Date();
  return Array.from({ length: 12 }, (_, i) => {
    const date = new Date(now);
    date.setMonth(now.getMonth() - 11 + i);
    return {
      month: date.toLocaleString('default', { month: 'short' }),
      value: Math.floor(Math.random() * 100) + 50,
      forecast: Math.floor(Math.random() * 100) + 60,
    };
  });
};

const generateCategoryData = () => {
  const categories = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E'];
  return categories.map(category => ({
    name: category,
    value: Math.floor(Math.random() * 100) + 20,
  }));
};

const LiveDemo = () => {
  const [activeTab, setActiveTab] = useState('timeSeries');
  const [timeSeriesData, setTimeSeriesData] = useState(generateTimeSeriesData());
  const [categoryData, setCategoryData] = useState(generateCategoryData());
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [showTooltip, setShowTooltip] = useState(false);
  const [tooltipData, setTooltipData] = useState({ x: 0, y: 0, value: 0, month: '' });
  const chartRef = useRef<HTMLDivElement>(null);
  
  // Simulate real-time data updates
  useEffect(() => {
    const interval = setInterval(() => {
      setIsRefreshing(true);
      
      setTimeout(() => {
        setTimeSeriesData(prev => {
          const newData = [...prev];
          // Update the last few points to simulate new data coming in
          for (let i = 9; i < 12; i++) {
            newData[i] = {
              ...newData[i],
              value: Math.floor(Math.random() * 100) + 50,
              forecast: Math.floor(Math.random() * 100) + 60,
            };
          }
          return newData;
        });
        
        setCategoryData(generateCategoryData());
        setIsRefreshing(false);
      }, 800);
    }, 8000);
    
    return () => clearInterval(interval);
  }, []);

  const handleBarHover = (index: number, event: React.MouseEvent) => {
    if (!chartRef.current) return;
    
    const rect = chartRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    setTooltipData({
      x,
      y,
      value: timeSeriesData[index].value,
      month: timeSeriesData[index].month
    });
    
    setShowTooltip(true);
  };

  const handleBarLeave = () => {
    setShowTooltip(false);
  };

  return (
    <div className="py-24 bg-gray-900" id="demo">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <div className="inline-flex items-center px-4 py-2 rounded-full bg-indigo-900/50 backdrop-blur-sm text-indigo-200 text-sm mb-4">
            <span className="animate-pulse h-2 w-2 rounded-full bg-indigo-400 mr-2"></span>
            Live Analytics
          </div>
          <h2 className="text-4xl font-extrabold text-white sm:text-5xl">
            <span className="block">See your data</span>
            <span className="block text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-400 mt-1">come to life</span>
          </h2>
          <p className="mt-4 max-w-2xl text-xl text-gray-300 mx-auto">
            Experience how our platform transforms raw data into meaningful visualizations in real-time.
            All charts update automatically as new data flows in.
          </p>
        </div>

        <div className="mt-16">
          <div className="glass-dark rounded-xl overflow-hidden shadow-2xl border border-gray-700">
            <div className="bg-gray-800 border-b border-gray-700 p-4 flex justify-between items-center">
              <div className="flex items-center space-x-2">
                <div className="h-3 w-3 rounded-full bg-red-500"></div>
                <div className="h-3 w-3 rounded-full bg-yellow-500"></div>
                <div className="h-3 w-3 rounded-full bg-green-500"></div>
              </div>
              <div className="text-gray-300 font-medium">DataSphere Analytics Dashboard</div>
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
                  <button
                    onClick={() => setActiveTab('timeSeries')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                      activeTab === 'timeSeries'
                        ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/20'
                        : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                    }`}
                  >
                    <div className="flex items-center">
                      <LineChart className="mr-2 h-4 w-4" />
                      Time Series
                    </div>
                  </button>
                  <button
                    onClick={() => setActiveTab('categories')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                      activeTab === 'categories'
                        ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/20'
                        : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                    }`}
                  >
                    <div className="flex items-center">
                      <BarChart3 className="mr-2 h-4 w-4" />
                      Categories
                    </div>
                  </button>
                  <button
                    onClick={() => setActiveTab('distribution')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                      activeTab === 'distribution'
                        ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/20'
                        : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                    }`}
                  >
                    <div className="flex items-center">
                      <PieChart className="mr-2 h-4 w-4" />
                      Distribution
                    </div>
                  </button>
                </div>
                
                <div className="flex items-center">
                  <div className="mr-4 flex items-center">
                    <span className="text-xs text-gray-400 mr-2">Last updated:</span>
                    <span className="text-xs text-white">Just now</span>
                  </div>
                  <button 
                    className={`flex items-center justify-center h-8 w-8 rounded-full bg-gray-800 text-gray-400 hover:text-white transition-colors ${isRefreshing ? 'animate-spin text-indigo-400' : ''}`}
                    onClick={() => {
                      setIsRefreshing(true);
                      setTimeout(() => {
                        setTimeSeriesData(generateTimeSeriesData());
                        setCategoryData(generateCategoryData());
                        setIsRefreshing(false);
                      }, 800);
                    }}
                  >
                    <RefreshCw className="h-4 w-4" />
                  </button>
                </div>
              </div>

              <div className="bg-gray-800 rounded-xl p-6 relative">
                {activeTab === 'timeSeries' && (
                  <div>
                    <div className="flex justify-between items-center mb-6">
                      <h3 className="text-lg font-medium text-white">Monthly Performance with AI Forecast</h3>
                      <div className="flex items-center space-x-2">
                        <div className="flex items-center">
                          <div className="h-3 w-3 bg-indigo-500 rounded-full mr-1"></div>
                          <span className="text-xs text-gray-300">Actual</span>
                        </div>
                        <div className="flex items-center">
                          <div className="h-3 w-3 bg-purple-400 rounded-full mr-1"></div>
                          <span className="text-xs text-gray-300">Forecast</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="h-80 relative" ref={chartRef}>
                      {/* Simplified chart visualization with gradient and glow */}
                      <div className="absolute inset-0 flex items-end">
                        {timeSeriesData.map((item, index) => (
                          <div 
                            key={index} 
                            className="flex-1 flex flex-col items-center"
                            onMouseEnter={(e) => handleBarHover(index, e)}
                            onMouseLeave={handleBarLeave}
                          >
                            <div className="w-full flex justify-center items-end h-64 relative">
                              <div 
                                className="w-4/5 bg-gradient-to-t from-indigo-600 to-indigo-400 rounded-t-lg shadow-lg relative overflow-hidden group transition-all duration-300 hover:w-full"
                                style={{ height: `${item.value * 0.5}%` }}
                              >
                                <div className="absolute inset-0 bg-white opacity-0 group-hover:opacity-10 transition-opacity"></div>
                              </div>
                              {index > 8 && (
                                <div 
                                  className="w-4/5 bg-gradient-to-t from-purple-500 to-purple-300 rounded-t-lg ml-1 border border-purple-400 relative overflow-hidden group transition-all duration-300 hover:w-full"
                                  style={{ height: `${item.forecast * 0.5}%` }}
                                >
                                  <div className="absolute inset-0 bg-white opacity-0 group-hover:opacity-10 transition-opacity"></div>
                                </div>
                              )}
                            </div>
                            <span className="text-xs text-gray-400 mt-2">{item.month}</span>
                          </div>
                        ))}
                      </div>
                      
                      {/* Y-axis labels */}
                      <div className="absolute left-0 top-0 bottom-0 w-10 flex flex-col justify-between text-right pr-2">
                        <span className="text-xs text-gray-500">100</span>
                        <span className="text-xs text-gray-500">75</span>
                        <span className="text-xs text-gray-500">50</span>
                        <span className="text-xs text-gray-500">25</span>
                        <span className="text-xs text-gray-500">0</span>
                      </div>
                      
                      {/* Horizontal grid lines */}
                      <div className="absolute left-10 right-0 top-0 bottom-0 flex flex-col justify-between pointer-events-none">
                        <div className="border-b border-gray-700 h-0"></div>
                        <div className="border-b border-gray-700 h-0"></div>
                        <div className="border-b border-gray-700 h-0"></div>
                        <div className="border-b border-gray-700 h-0"></div>
                        <div className="border-b border-gray-700 h-0"></div>
                      </div>
                      
                      {/* Tooltip */}
                      {showTooltip && (
                        <div 
                          className="absolute bg-gray-900 text-white p-2 rounded shadow-lg text-xs z-10 pointer-events-none"
                          style={{ 
                            left: `${tooltipData.x}px`, 
                            top: `${tooltipData.y - 40}px`,
                            transform: 'translateX(-50%)'
                          }}
                        >
                          <div className="font-bold">{tooltipData.month}</div>
                          <div>Value: {tooltipData.value}</div>
                        </div>
                      )}
                      
                      {isRefreshing && (
                        <div className="absolute inset-0 bg-gray-900/50 backdrop-blur-sm flex items-center justify-center">
                          <div className="text-indigo-400 flex items-center">
                            <RefreshCw className="h-5 w-5 mr-2 animate-spin" />
                            <span>Updating data...</span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {activeTab === 'categories' && (
                  <div>
                    <div className="flex justify-between items-center mb-6">
                      <h3 className="text-lg font-medium text-white">Category Performance Analysis</h3>
                      <div className="flex items-center space-x-2">
                        <button className="flex items-center text-xs text-gray-300 bg-gray-700 px-3 py-1 rounded-full">
                          <span>This Quarter</span>
                          <ChevronDown className="ml-1 h-3 w-3" />
                        </button>
                      </div>
                    </div>
                    
                    <div className="h-80 relative">
                      {/* Simplified bar chart visualization with gradients and animations */}
                      <div className="absolute inset-0 flex items-end pt-10">
                        {categoryData.map((item, index) => (
                          <div key={index} className="flex-1 flex flex-col items-center">
                            <div className="w-full flex justify-center items-end h-64">
                              <div className="relative group w-4/5">
                                <div className="absolute inset-0 bg-indigo-500 blur-md opacity-30 rounded-lg transform group-hover:scale-110 transition-transform"></div>
                                <div 
                                  className="w-full bg-gradient-to-t from-indigo-600 to-indigo-400 rounded-lg relative z-10 transform transition-all duration-500 group-hover:scale-105"
                                  style={{ height: `${item.value * 0.6}%` }}
                                >
                                  <div className="absolute inset-x-0 top-0 h-1/3 bg-white/20 rounded-t-lg"></div>
                                  <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-7 bg-gray-900 text-white px-2 py-1 rounded text-xs">
                                    {item.value}
                                  </div>
                                </div>
                              </div>
                            </div>
                            <span className="text-sm text-gray-300 mt-4 font-medium">{item.name}</span>
                          </div>
                        ))}
                      </div>
                      
                      {/* Y-axis labels */}
                      <div className="absolute left-0 top-0 bottom-0 w-10 flex flex-col justify-between text-right pr-2 pt-10">
                        <span className="text-xs text-gray-500">100</span>
                        <span className="text-xs text-gray-500">75</span>
                        <span className="text-xs text-gray-500">50</span>
                        <span className="text-xs text-gray-500">25</span>
                        <span className="text-xs text-gray-500">0</span>
                      </div>
                      
                      {/* Horizontal grid lines */}
                      <div className="absolute left-10 right-0 top-0 bottom-0 flex flex-col justify-between pointer-events-none pt-10">
                        <div className="border-b border-gray-700 h-0"></div>
                        <div className="border-b border-gray-700 h-0"></div>
                        <div className="border-b border-gray-700 h-0"></div>
                        <div className="border-b border-gray-700 h-0"></div>
                        <div className="border-b border-gray-700 h-0"></div>
                      </div>
                      
                      {isRefreshing && (
                        <div className="absolute inset-0 bg-gray-900/50 backdrop-blur-sm flex items-center justify-center">
                          <div className="text-indigo-400 flex items-center">
                            <RefreshCw className="h-5 w-5 mr-2 animate-spin" />
                            <span>Updating data...</span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {activeTab === 'distribution' && (
                  <div>
                    <div className="flex justify-between items-center mb-6">
                      <h3 className="text-lg font-medium text-white">Market Distribution</h3>
                      <div className="flex items-center space-x-2">
                        <button className="flex items-center text-xs text-gray-300 bg-gray-700 px-3 py-1 rounded-full">
                          <span>All Regions</span>
                          <ChevronDown className="ml-1 h-3 w-3" />
                        </button>
                      </div>
                    </div>
                    
                    <div className="h-80 flex items-center justify-center">
                       <div className="relative w-80 h-80">
                        {/* Enhanced pie chart visualization with 3D effect */}
                        <svg viewBox="0 0 100 100" className="w-full h-full drop-shadow-2xl">
                          <defs>
                            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                              <feGaussianBlur stdDeviation="2.5" result="blur" />
                              <feComposite in="SourceGraphic" in2="blur" operator="over" />
                            </filter>
                            
                            {categoryData.map((_, index) => (
                              <linearGradient 
                                key={`gradient-${index}`} 
                                id={`pieGradient${index}`} 
                                x1="0%" 
                                y1="0%" 
                                x2="100%" 
                                y2="100%"
                              >
                                <stop 
                                  offset="0%" 
                                  stopColor={
                                    index === 0 ? '#4f46e5' : 
                                    index === 1 ? '#8b5cf6' : 
                                    index === 2 ? '#6366f1' : 
                                    index === 3 ? '#a78bfa' : 
                                    '#818cf8'
                                  } 
                                />
                                <stop 
                                  offset="100%" 
                                  stopColor={
                                    index === 0 ? '#6366f1' : 
                                    index === 1 ? '#a78bfa' : 
                                    index === 2 ? '#818cf8' : 
                                    index === 3 ? '#c4b5fd' : 
                                    '#a5b4fc'
                                  } 
                                />
                              </linearGradient>
                            ))}
                          </defs>
                          
                          {/* 3D effect base shadow */}
                          <ellipse cx="50" cy="50" rx="40" ry="10" fill="#1f2937" opacity="0.3" />
                          
                          {categoryData.map((item, index) => {
                            const total = categoryData.reduce((sum, item) => sum + item.value, 0);
                            const startAngle = categoryData
                              .slice(0, index)
                              .reduce((sum, item) => sum + (item.value / total) * 360, 0);
                            const endAngle = startAngle + (item.value / total) * 360;
                            
                            // Convert angles to radians and calculate x,y coordinates
                            const startRad = (startAngle - 90) * Math.PI / 180;
                            const endRad = (endAngle - 90) * Math.PI / 180;
                            
                            const x1 = 50 + 40 * Math.cos(startRad);
                            const y1 = 50 + 40 * Math.sin(startRad);
                            const x2 = 50 + 40 * Math.cos(endRad);
                            const y2 = 50 + 40 * Math.sin(endRad);
                            
                            // Determine if the arc should be drawn as a large arc
                            const largeArcFlag = endAngle - startAngle > 180 ? 1 : 0;
                            
                            // Calculate midpoint for the pull-out effect
                            const midAngle = (startAngle + endAngle) / 2;
                            const midRad = (midAngle - 90) * Math.PI / 180;
                            const pullOut = index === 1 ? 5 : 0; // Pull out the second slice
                            const cx = pullOut * Math.cos(midRad);
                            const cy = pullOut * Math.sin(midRad);
                            
                            return (
                              <g key={index} className="cursor-pointer hover:opacity-90 transition-opacity" transform={`translate(${cx}, ${cy})`}>
                                {/* 3D effect side */}
                                <path
                                  d={`M 50 50 L ${x1} ${y1} A 40 40 0 ${largeArcFlag} 1 ${x2} ${y2} Z`}
                                  fill={`url(#pieGradient${index})`}
                                  stroke="#1f2937"
                                  strokeWidth="0.5"
                                  filter="url(#glow)"
                                >
                                  <animate 
                                    attributeName="opacity" 
                                    values="0;1" 
                                    dur="1s" 
                                    begin={`${index * 0.2}s`} 
                                    fill="freeze" 
                                  />
                                </path>
                                
                                {/* Highlight effect */}
                                <path
                                  d={`M 50 50 L ${x1} ${y1} A 40 40 0 ${largeArcFlag} 1 ${x2} ${y2} Z`}
                                  fill="white"
                                  opacity="0.1"
                                />
                              </g>
                            );
                          })}
                          
                          {/* Center circle */}
                          <circle cx="50" cy="50" r="20" fill="#1f2937" />
                          <circle cx="50" cy="50" r="18" fill="#374151" />
                          <text x="50" y="50" textAnchor="middle" dominantBaseline="middle" fill="white" fontSize="6" fontWeight="bold">
                            MARKET SHARE
                          </text>
                        </svg>
                      </div>
                      
                      <div className="ml-8 space-y-4">
                        {categoryData.map((item, index) => {
                          const colors = [
                            'from-indigo-600 to-indigo-400',
                            'from-purple-600 to-purple-400',
                            'from-indigo-500 to-blue-400',
                            'from-violet-600 to-violet-400',
                            'from-blue-600 to-indigo-400'
                          ];
                          
                          const total = categoryData.reduce((sum, item) => sum + item.value, 0);
                          const percentage = Math.round((item.value / total) * 100);
                          
                          return (
                            <div key={index} className="flex items-center">
                              <div className={`h-4 w-4 rounded-sm bg-gradient-to-r ${colors[index % colors.length]}`}></div>
                              <div className="ml-3 flex-1">
                                <div className="flex justify-between">
                                  <span className="text-sm font-medium text-white">{item.name}</span>
                                  <span className="text-sm text-gray-400">{percentage}%</span>
                                </div>
                                <div className="mt-1 h-1.5 w-full bg-gray-700 rounded-full overflow-hidden">
                                  <div 
                                    className={`h-full bg-gradient-to-r ${colors[index % colors.length]}`}
                                    style={{ width: `${percentage}%` }}
                                  ></div>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                      
                      {isRefreshing && (
                        <div className="absolute inset-0 bg-gray-900/50 backdrop-blur-sm flex items-center justify-center">
                          <div className="text-indigo-400 flex items-center">
                            <RefreshCw className="h-5 w-5 mr-2 animate-spin" />
                            <span>Updating data...</span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
              
              <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-gray-800 rounded-xl p-5 border border-gray-700/50 hover:border-indigo-500/30 transition-colors duration-300">
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="text-gray-400 text-sm">Total Processed Data</h3>
                      <p className="text-2xl font-bold text-white mt-1">1.28 TB</p>
                    </div>
                    <div className="bg-indigo-500/10 p-2 rounded-lg">
                      <Database className="h-6 w-6 text-indigo-400" />
                    </div>
                  </div>
                  <div className="mt-4">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">Progress</span>
                      <span className="text-indigo-400">85%</span>
                    </div>
                    <div className="mt-1 h-1.5 w-full bg-gray-700 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-indigo-600 to-indigo-400 rounded-full" style={{ width: '85%' }}></div>
                    </div>
                  </div>
                </div>
                
                <div className="bg-gray-800 rounded-xl p-5 border border-gray-700/50 hover:border-indigo-500/30 transition-colors duration-300">
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="text-gray-400 text-sm">Processing Speed</h3>
                      <p className="text-2xl font-bold text-white mt-1">42.8 MB/s</p>
                    </div>
                    <div className="bg-purple-500/10 p-2 rounded-lg">
                      <Zap className="h-6 w-6 text-purple-400" />
                    </div>
                  </div>
                  <div className="mt-4 flex space-x-1">
                    {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].map((i) => (
                      <div 
                        key={i} 
                        className="flex-1 h-10 bg-gray-700 rounded-sm overflow-hidden flex items-end"
                      >
                        <div 
                          className="w-full bg-gradient-to-t from-purple-600 to-purple-400" 
                          style={{ height: `${Math.random() * 100}%` }}
                        ></div>
                      </div>
                    ))}
                  </div>
                </div>
                
                <div className="bg-gray-800 rounded-xl p-5 border border-gray-700/50 hover:border-indigo-500/30 transition-colors duration-300">
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="text-gray-400 text-sm">AI Accuracy</h3>
                      <p className="text-2xl font-bold text-white mt-1">98.7%</p>
                    </div>
                    <div className="bg-blue-500/10 p-2 rounded-lg">
                      <LineChart className="h-6 w-6 text-blue-400" />
                    </div>
                  </div>
                  <div className="mt-4 relative h-10">
                    <svg className="w-full h-full" viewBox="0 0 100 30" preserveAspectRatio="none">
                      <path 
                        d="M0,15 C10,10 20,20 30,15 C40,10 50,20 60,15 C70,10 80,5 90,10 L90,30 L0,30 Z" 
                        fill="url(#blueGradient)" 
                        fillOpacity="0.5"
                      />
                      <path 
                        d="M0,15 C10,10 20,20 30,15 C40,10 50,20 60,15 C70,10 80,5 90,10" 
                        fill="none" 
                        stroke="url(#blueGradient)" 
                        strokeWidth="1"
                      />
                      <defs>
                        <linearGradient id="blueGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                          <stop offset="0%" stopColor="#3b82f6" />
                          <stop offset="100%" stopColor="#60a5fa" />
                        </linearGradient>
                      </defs>
                    </svg>
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