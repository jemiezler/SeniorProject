'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Leaf, ArrowRight } from 'lucide-react';
import Waves from '@/components/ui/Waves';

export function Hero() {
  return (
    <div className="relative overflow-hidden pt-20 pb-12">
      <Waves 
        lineColor="rgba(34, 197, 94, 0.2)"
        backgroundColor="transparent"
        className="absolute inset-0 z-0"
      />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="text-center space-y-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="space-y-4"
          >
            
            
            <h1 className="text-4xl md:text-5xl font-bold">
              <span className="bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-300 bg-clip-text text-transparent">
                Kale Analysis Platform
              </span>
            </h1>

            <div className="flex items-center justify-center space-x-2">
              <Leaf className="h-8 w-8 text-green-600" />
              <h2 className="text-lg font-semibold text-green-600">Senior Project - Mae Fah Luang University</h2>
            </div>
            
            <p className="max-w-2xl mx-auto text-lg text-gray-600 dark:text-gray-300">
              Advanced AI-powered kale analysis system. Upload multiple images for instant quality assessment and detailed recommendations.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <button className="px-8 py-3 rounded-lg bg-gradient-to-r from-green-600 to-emerald-600 text-white font-medium hover:opacity-90 transition-opacity flex items-center space-x-2 group">
              <span>Start Analysis</span>
              <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
            </button>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="grid grid-cols-1 md:grid-cols-1 gap-8 max-w-3xl mx-auto"
          >
            <div className="text-center">
              <div className="text-3xl font-bold bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent">
                NaN
              </div>
              <p className="text-gray-600 dark:text-gray-400">Accuracy Rate</p>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
} 