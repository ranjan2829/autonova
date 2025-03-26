"use client";
import Navbar from '@/components/Navbar';
import Hero from '@/components/Hero';
import Features from '@/components/Features';

import Footer from '@/components/Footer';
import DataVisualizer from '@/components/Analytics';
import Analytics from '@/components/Analytics';

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-900">
      <Navbar />
      <Hero />
    
      <Features />
      <Analytics />
      
      
      <Footer />
      
      
      
    </main>
  );
}