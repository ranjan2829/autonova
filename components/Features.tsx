"use client";
import React from 'react';
import { BarChart3, Database, LineChart, PieChart, Layers, Zap, Shield, Users } from 'lucide-react';

const features = [
  {
    name: 'Automated Data Processing',
    description: 'Our AI automatically cleans, transforms, and prepares your data for analysis without manual intervention.',
    icon: Database,
    color: 'from-blue-500 to-indigo-500',
  },
  {
    name: 'Real-time Analytics',
    description: 'Monitor your data streams in real-time with dynamic dashboards that update as new information arrives.',
    icon: LineChart,
    color: 'from-indigo-500 to-purple-500',
  },
  {
    name: 'Predictive Insights',
    description: 'Leverage machine learning to forecast trends and identify opportunities before they emerge.',
    icon: BarChart3,
    color: 'from-purple-500 to-pink-500',
  },
  {
    name: 'Multi-dimensional Visualization',
    description: 'Explore complex data relationships through interactive and customizable visualization tools.',
    icon: PieChart,
    color: 'from-pink-500 to-rose-500',
  },
  {
    name: 'Data Integration Hub',
    description: 'Connect to any data source with our extensive library of pre-built connectors and APIs.',
    icon: Layers,
    color: 'from-rose-500 to-orange-500',
  },
  {
    name: 'Lightning-fast Performance',
    description: 'Process millions of data points in seconds with our optimized cloud infrastructure.',
    icon: Zap,
    color: 'from-orange-500 to-amber-500',
  },
  {
    name: 'Enterprise-grade Security',
    description: 'Rest easy knowing your data is protected with end-to-end encryption and compliance controls.',
    icon: Shield,
    color: 'from-amber-500 to-yellow-500',
  },
  {
    name: 'Collaborative Workspace',
    description: 'Share insights and collaborate with team members through interactive reports and dashboards.',
    icon: Users,
    color: 'from-yellow-500 to-lime-500',
  },
];

const Features = () => {
  return (
    <div className="py-24 bg-gray-900" id="features">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <div className="inline-flex items-center px-4 py-2 rounded-full bg-indigo-900/50 backdrop-blur-sm text-indigo-200 text-sm mb-4">
            <span className="animate-pulse h-2 w-2 rounded-full bg-indigo-400 mr-2"></span>
            Powerful Features
          </div>
          <h2 className="text-4xl font-extrabold text-white sm:text-5xl">
            <span className="block">Transform your data</span>
            <span className="block text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-400 mt-1">into actionable insights</span>
          </h2>
          <p className="mt-4 max-w-2xl text-xl text-gray-300 mx-auto">
            Our platform combines cutting-edge AI with powerful data engineering tools to deliver insights faster than ever before.
          </p>
        </div>

        <div className="mt-20">
          <div className="grid grid-cols-1 gap-12 md:grid-cols-2 lg:grid-cols-4">
            {features.map((feature, index) => (
              <div 
                key={feature.name} 
                className="relative bg-gray-800 rounded-xl p-8 overflow-hidden border border-gray-700 hover:border-indigo-500/50 transition-all duration-300 group"
              >
                <div className="absolute inset-0 bg-gradient-to-br opacity-0 group-hover:opacity-10 transition-opacity duration-300"></div>
                
                <div className={`absolute -right-6 -top-6 w-24 h-24 rounded-full bg-gradient-to-br ${feature.color} opacity-10 blur-xl group-hover:opacity-20 transition-opacity duration-300`}></div>
                
                <div className={`relative inline-flex items-center justify-center h-14 w-14 rounded-xl bg-gradient-to-br ${feature.color} text-white mb-6 shadow-lg`}>
                  <feature.icon className="h-7 w-7" aria-hidden="true" />
                </div>
                
                <h3 className="text-xl font-bold text-white mb-3 group-hover:text-indigo-300 transition-colors duration-300">{feature.name}</h3>
                
                <p className="text-gray-400 group-hover:text-gray-300 transition-colors duration-300">{feature.description}</p>
                
                <div className="absolute bottom-0 left-0 h-1 bg-gradient-to-r from-transparent via-indigo-500 to-transparent w-0 group-hover:w-full transition-all duration-700 opacity-0 group-hover:opacity-100"></div>
              </div>
            ))}
          </div>
        </div>
        
        <div className="mt-24 relative">
          <div className="absolute inset-0 flex items-center" aria-hidden="true">
            <div className="w-full border-t border-gray-700"></div>
          </div>
          <div className="relative flex justify-center">
            <span className="px-6 bg-gray-900 text-lg font-medium text-indigo-400">
              Trusted by industry leaders
            </span>
          </div>
        </div>
        
        <div className="mt-12 grid grid-cols-2 gap-8 md:grid-cols-6">
          <div className="col-span-1 flex justify-center items-center grayscale opacity-40 hover:grayscale-0 hover:opacity-100 transition-all duration-300">
            <img className="h-12" src="https://tailwindui.com/img/logos/tuple-logo-gray-400.svg" alt="Tuple" />
          </div>
          <div className="col-span-1 flex justify-center items-center grayscale opacity-40 hover:grayscale-0 hover:opacity-100 transition-all duration-300">
            <img className="h-12" src="https://tailwindui.com/img/logos/mirage-logo-gray-400.svg" alt="Mirage" />
          </div>
          <div className="col-span-1 flex justify-center items-center grayscale opacity-40 hover:grayscale-0 hover:opacity-100 transition-all duration-300">
            <img className="h-12" src="https://tailwindui.com/img/logos/statickit-logo-gray-400.svg" alt="StaticKit" />
          </div>
          <div className="col-span-1 flex justify-center items-center grayscale opacity-40 hover:grayscale-0 hover:opacity-100 transition-all duration-300">
            <img className="h-12" src="https://tailwindui.com/img/logos/transistor-logo-gray-400.svg" alt="Transistor" />
          </div>
          <div className="col-span-1 flex justify-center items-center grayscale opacity-40 hover:grayscale-0 hover:opacity-100 transition-all duration-300">
            <img className="h-12" src="https://tailwindui.com/img/logos/workcation-logo-gray-400.svg" alt="Workcation" />
          </div>
          <div className="col-span-1 flex justify-center items-center grayscale opacity-40 hover:grayscale-0 hover:opacity-100 transition-all duration-300">
            <img className="h-12" src="https://tailwindui.com/img/logos/laravel-logo-gray-400.svg" alt="Laravel" />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Features;