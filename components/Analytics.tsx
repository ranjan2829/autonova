import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from 'recharts';
import { ArrowUpRight, Zap, TrendingUp, Users } from 'lucide-react';

const data = [
  { name: 'Jan', value: 400 },
  { name: 'Feb', value: 300 },
  { name: 'Mar', value: 600 },
  { name: 'Apr', value: 800 },
  { name: 'May', value: 700 },
  { name: 'Jun', value: 900 },
];

const Analytics = () => {
  return (
    <section className="py-20 bg-black relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-b from-indigo-900/20 to-transparent"></div>
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <div className="space-y-8">
            <h2 className="text-4xl font-bold">
              Real-time Analytics
              <span className="block text-indigo-500 mt-2">That Drive Results</span>
            </h2>
            
            <div className="grid grid-cols-2 gap-6">
              <div className="bg-gray-900/50 backdrop-blur-xl p-6 rounded-xl border border-gray-800">
                <div className="flex items-center justify-between mb-4">
                  <TrendingUp className="h-6 w-6 text-indigo-500" />
                  <span className="text-green-500 flex items-center text-sm">
                    +24% <ArrowUpRight className="h-4 w-4 ml-1" />
                  </span>
                </div>
                <p className="text-gray-400 text-sm">Processing Speed</p>
                <p className="text-2xl font-bold mt-1">2.4ms</p>
              </div>
              
              <div className="bg-gray-900/50 backdrop-blur-xl p-6 rounded-xl border border-gray-800">
                <div className="flex items-center justify-between mb-4">
                  <Users className="h-6 w-6 text-indigo-500" />
                  <span className="text-green-500 flex items-center text-sm">
                    +12% <ArrowUpRight className="h-4 w-4 ml-1" />
                  </span>
                </div>
                <p className="text-gray-400 text-sm">Active Users</p>
                <p className="text-2xl font-bold mt-1">14.2k</p>
              </div>
            </div>
          </div>
          
          <div className="bg-gray-900/30 backdrop-blur-xl p-8 rounded-2xl border border-gray-800">
            <div className="flex items-center justify-between mb-8">
              <h3 className="text-xl font-semibold">Data Processing Metrics</h3>
              <Zap className="h-5 w-5 text-indigo-500" />
            </div>
            
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="name" stroke="#6b7280" />
                  <YAxis stroke="#6b7280" />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#6366f1"
                    strokeWidth={2}
                    dot={{ fill: '#6366f1' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Analytics;