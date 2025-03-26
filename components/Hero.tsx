import React, { useEffect, useRef } from 'react';
import { ArrowRight, Database, LineChart, Zap, ChevronDown } from 'lucide-react';

const Hero = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas dimensions
    const setCanvasDimensions = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    setCanvasDimensions();
    window.addEventListener('resize', setCanvasDimensions);

    // Particle settings
    const particlesArray: Particle[] = [];
    const numberOfParticles = 100;

    class Particle {
      x: number;
      y: number;
      size: number;
      speedX: number;
      speedY: number;
      color: string;
      
      constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.size = Math.random() * 3 + 1;
        this.speedX = Math.random() * 1 - 0.5;
        this.speedY = Math.random() * 1 - 0.5;
        this.color = `rgba(99, 102, 241, ${Math.random() * 0.5 + 0.2})`;
      }
      
      update() {
        this.x += this.speedX;
        this.y += this.speedY;
        
        if (this.x > canvas.width) this.x = 0;
        else if (this.x < 0) this.x = canvas.width;
        
        if (this.y > canvas.height) this.y = 0;
        else if (this.y < 0) this.y = canvas.height;
      }
      
      draw() {
        if (!ctx) return;
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    const init = () => {
      for (let i = 0; i < numberOfParticles; i++) {
        particlesArray.push(new Particle());
      }
    };

    const connectParticles = () => {
      if (!ctx) return;
      for (let a = 0; a < particlesArray.length; a++) {
        for (let b = a; b < particlesArray.length; b++) {
          const dx = particlesArray[a].x - particlesArray[b].x;
          const dy = particlesArray[a].y - particlesArray[b].y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          if (distance < 100) {
            ctx.strokeStyle = `rgba(99, 102, 241, ${0.2 - distance/500})`;
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(particlesArray[a].x, particlesArray[a].y);
            ctx.lineTo(particlesArray[b].x, particlesArray[b].y);
            ctx.stroke();
          }
        }
      }
    };

    const animate = () => {
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      for (let i = 0; i < particlesArray.length; i++) {
        particlesArray[i].update();
        particlesArray[i].draw();
      }
      connectParticles();
      requestAnimationFrame(animate);
    };

    init();
    animate();

    return () => {
      window.removeEventListener('resize', setCanvasDimensions);
    };
  }, []);

  return (
    <div className="relative overflow-hidden bg-gray-900 min-h-screen flex items-center" id="home">
      <canvas ref={canvasRef} className="absolute inset-0 z-0"></canvas>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 py-20">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <div className="space-y-8">
            <div className="inline-flex items-center px-4 py-2 rounded-full bg-indigo-900/50 backdrop-blur-sm text-indigo-200 text-sm">
              <span className="animate-pulse h-2 w-2 rounded-full bg-indigo-400 mr-2"></span>
              Next-Gen Data Engineering Platform
            </div>
            
            <h1 className="text-5xl md:text-6xl lg:text-7xl font-extrabold tracking-tight">
              <span className="block text-white">Transform Data</span>
              <span className="block mt-2">
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-400 neon-text">
                  Into Intelligence
                </span>
              </span>
            </h1>
            
            <p className="text-xl text-gray-300 max-w-xl">
              Our AI-powered platform automates your entire data pipeline, from ingestion to visualization, 
              delivering real-time insights with unprecedented speed and accuracy.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4">
              <a
                href="#demo"
                className="px-8 py-4 rounded-lg bg-gradient-to-r from-indigo-600 to-indigo-500 text-white font-medium text-lg flex items-center justify-center transform transition-all duration-300 hover:scale-105 hover:shadow-lg hover:shadow-indigo-500/50 animate-pulse-glow"
              >
                Experience Live Demo
                <ArrowRight className="ml-2 h-5 w-5" />
              </a>
              
              <a
                href="#features"
                className="px-8 py-4 rounded-lg bg-gray-800/80 backdrop-blur-sm text-white font-medium text-lg flex items-center justify-center border border-gray-700 hover:border-indigo-500 transition-all duration-300"
              >
                Explore Features
              </a>
            </div>
            
            <div className="flex items-center space-x-8 pt-6">
              <div className="flex -space-x-2">
                <img className="h-8 w-8 rounded-full ring-2 ring-indigo-600" src="https://images.unsplash.com/photo-1491528323818-fdd1faba62cc?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=facearea&facepad=2&w=256&h=256&q=80" alt="" />
                <img className="h-8 w-8 rounded-full ring-2 ring-indigo-600" src="https://images.unsplash.com/photo-1550525811-e5869dd03032?ixlib=rb-1.2.1&auto=format&fit=facearea&facepad=2&w=256&h=256&q=80" alt="" />
                <img className="h-8 w-8 rounded-full ring-2 ring-indigo-600" src="https://images.unsplash.com/photo-1500648767791-00dcc994a43e?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=facearea&facepad=2.25&w=256&h=256&q=80" alt="" />
              </div>
              <div className="text-gray-400 text-sm">
                <span className="text-indigo-400 font-semibold">500+</span> companies already using our platform
              </div>
            </div>
          </div>
          
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-3xl blur-3xl opacity-20 animate-pulse"></div>
            <div className="relative glass-dark rounded-3xl p-6 border border-gray-700 shadow-2xl animate-float">
              <div className="absolute top-0 right-0 -mt-4 -mr-4 bg-indigo-500 rounded-full p-2 shadow-lg">
                <Zap className="h-6 w-6 text-white" />
              </div>
              
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center">
                  <Database className="h-6 w-6 text-indigo-400" />
                  <span className="ml-2 text-lg font-semibold text-white">DataSphere Analytics</span>
                </div>
                <div className="flex space-x-1">
                  <div className="h-3 w-3 rounded-full bg-red-500"></div>
                  <div className="h-3 w-3 rounded-full bg-yellow-500"></div>
                  <div className="h-3 w-3 rounded-full bg-green-500"></div>
                </div>
              </div>
              
              <div className="space-y-6">
                <div className="bg-gray-800/50 rounded-lg p-4">
                  <div className="flex justify-between items-center mb-3">
                    <h3 className="text-white font-medium">Revenue Forecast</h3>
                    <span className="text-xs text-indigo-400">Live</span>
                  </div>
                  
                  <div className="h-40 relative">
                    <svg className="w-full h-full" viewBox="0 0 400 150" preserveAspectRatio="none">
                      <path 
                        d="M0,150 L0,120 C13.333333333333334,115 26.666666666666668,110 40,100 C53.333333333333336,90 66.66666666666667,75 80,70 C93.33333333333333,65 106.66666666666667,70 120,75 C133.33333333333334,80 146.66666666666666,85 160,85 C173.33333333333334,85 186.66666666666666,80 200,75 C213.33333333333334,70 226.66666666666666,65 240,65 C253.33333333333334,65 266.6666666666667,70 280,75 C293.3333333333333,80 306.6666666666667,85 320,80 C333.3333333333333,75 346.6666666666667,60 360,50 C373.3333333333333,40 386.6666666666667,35 400,30 L400,150 Z" 
                        fill="url(#gradient1)" 
                        fillOpacity="0.2"
                        className="data-line"
                      />
                      <path 
                        d="M0,120 C13.333333333333334,115 26.666666666666668,110 40,100 C53.333333333333336,90 66.66666666666667,75 80,70 C93.33333333333333,65 106.66666666666667,70 120,75 C133.33333333333334,80 146.66666666666666,85 160,85 C173.33333333333334,85 186.66666666666666,80 200,75 C213.33333333333334,70 226.66666666666666,65 240,65 C253.33333333333334,65 266.6666666666667,70 280,75 C293.3333333333333,80 306.6666666666667,85 320,80 C333.3333333333333,75 346.6666666666667,60 360,50 C373.3333333333333,40 386.6666666666667,35 400,30" 
                        fill="none" 
                        stroke="url(#gradient2)" 
                        strokeWidth="3"
                        className="data-line"
                      />
                      <defs>
                        <linearGradient id="gradient1" x1="0%" y1="0%" x2="100%" y2="0%">
                          <stop offset="0%" stopColor="#4f46e5" />
                          <stop offset="100%" stopColor="#8b5cf6" />
                        </linearGradient>
                        <linearGradient id="gradient2" x1="0%" y1="0%" x2="100%" y2="0%">
                          <stop offset="0%" stopColor="#4f46e5" />
                          <stop offset="100%" stopColor="#8b5cf6" />
                        </linearGradient>
                      </defs>
                    </svg>
                    
                    <div className="absolute bottom-0 left-0 right-0 flex justify-between text-xs text-gray-400">
                      <span>Jan</span>
                      <span>Mar</span>
                      <span>Jun</span>
                      <span>Sep</span>
                      <span>Dec</span>
                    </div>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-800/50 rounded-lg p-4">
                    <h3 className="text-gray-400 text-sm mb-1">Conversion Rate</h3>
                    <div className="flex items-end">
                      <span className="text-2xl font-bold text-white">24.8%</span>
                      <span className="ml-2 text-xs text-green-400 flex items-center">+2.4%</span>
                    </div>
                    <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full" style={{ width: '24.8%' }}></div>
                    </div>
                  </div>
                  
                  <div className="bg-gray-800/50 rounded-lg p-4">
                    <h3 className="text-gray-400 text-sm mb-1">Active Users</h3>
                    <div className="flex items-end">
                      <span className="text-2xl font-bold text-white">14.2k</span>
                      <span className="ml-2 text-xs text-green-400 flex items-center">+5.3%</span>
                    </div>
                    <div className="mt-2 flex justify-between">
                      <div className="h-8 w-2 bg-indigo-900 rounded-full overflow-hidden">
                        <div className="h-3 w-full bg-indigo-500 rounded-full"></div>
                      </div>
                      <div className="h-8 w-2 bg-indigo-900 rounded-full overflow-hidden">
                        <div className="h-5 w-full bg-indigo-500 rounded-full"></div>
                      </div>
                      <div className="h-8 w-2 bg-indigo-900 rounded-full overflow-hidden">
                        <div className="h-4 w-full bg-indigo-500 rounded-full"></div>
                      </div>
                      <div className="h-8 w-2 bg-indigo-900 rounded-full overflow-hidden">
                        <div className="h-6 w-full bg-indigo-500 rounded-full"></div>
                      </div>
                      <div className="h-8 w-2 bg-indigo-900 rounded-full overflow-hidden">
                        <div className="h-7 w-full bg-indigo-500 rounded-full"></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div className="absolute bottom-10 left-1/2 transform -translate-x-1/2 z-10 text-white animate-bounce">
        <a href="#features" className="flex flex-col items-center">
          <span className="text-sm text-gray-400 mb-2">Scroll to explore</span>
          <ChevronDown className="h-6 w-6" />
        </a>
      </div>
    </div>
  );
};

export default Hero;