'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Hero } from '@/components/layout/Hero';
import { AnalysisSection } from '@/components/analysis/AnalysisSection';
import { FeatureCard } from '@/components/ui/FeatureCard';
import { Leaf, Zap, ChartBar, Shield } from 'lucide-react';
import type { AnalysisResult } from '@/types';

export default function Home() {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState<AnalysisResult[]>([]);

  const handleFileSelect = (files: File[]) => {
    setSelectedFiles(files);
    const newPreviews: string[] = [];
    
    files.forEach(file => {
      const reader = new FileReader();
      reader.onloadend = () => {
        newPreviews.push(reader.result as string);
        if (newPreviews.length === files.length) {
          setPreviews(newPreviews);
        }
      };
      reader.readAsDataURL(file);
    });
  };

  const handleRemoveFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
    setPreviews(prev => prev.filter((_, i) => i !== index));
  };

  const handleAnalyze = async () => {
    if (selectedFiles.length === 0) return;
    setIsAnalyzing(true);
    setProgress(0);

    const newResults: AnalysisResult[] = [];
    for (let i = 0; i < selectedFiles.length; i++) {
      // Simulate progress
      setProgress(((i + 1) / selectedFiles.length) * 100);
      
      // Simulated analysis delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      newResults.push({
        prediction: "Premium Fresh Kale",
        confidence: 0.98,
        segmentation: {
          url: previews[i],
          width: 800,
          height: 600
        },
        recommendations: [
          "Optimal storage temperature: 4°C",
          "Best consumed within 5 days",
          "Excellent for salads and smoothies"
        ]
      });
    }

    setResults(newResults);
    setProgress(100);
    setIsAnalyzing(false);
  };

  const features = [
    {
      icon: Leaf,
      title: "Advanced Analysis",
      description: "State-of-the-art AI models for precise kale quality assessment"
    },
    {
      icon: Zap,
      title: "Real-time Results",
      description: "Get instant analysis with detailed quality metrics"
    },
    {
      icon: ChartBar,
      title: "Detailed Insights",
      description: "Comprehensive reports with actionable recommendations"
    },
    {
      icon: Shield,
      title: "Quality Assurance",
      description: "Ensure consistent produce quality with AI-powered verification"
    }
  ];

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <Hero />

      {/* Main Content */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <AnalysisSection
          selectedFiles={selectedFiles}
          previews={previews}
          isAnalyzing={isAnalyzing}
          progress={progress}
          results={results}
          onFileSelect={handleFileSelect}
          onRemoveFile={handleRemoveFile}
          onAnalyze={handleAnalyze}
        />
      </motion.div>

      {/* Features Section */}
      <section className="py-24 bg-gradient-to-b from-transparent to-green-50 dark:to-green-950/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl font-bold bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-300 bg-clip-text text-transparent">
              Why Choose Kale AI?
            </h2>
            <p className="mt-4 text-lg text-gray-600 dark:text-gray-400">
              Experience the future of produce quality assessment
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <FeatureCard
                  icon={feature.icon}
                  title={feature.title}
                  description={feature.description}
                />
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
} 