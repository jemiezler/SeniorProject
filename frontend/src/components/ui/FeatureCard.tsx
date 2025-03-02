'use client';

import React from 'react';
import { LucideIcon } from 'lucide-react';

interface FeatureCardProps {
  icon: LucideIcon;
  title: string;
  description: string;
}

export function FeatureCard({ icon: Icon, title, description }: FeatureCardProps) {
  return (
    <div className="w-[223px] h-[94px] p-4 bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm rounded-2xl border border-green-100 dark:border-green-900 hover:scale-105 transition-transform duration-300 flex items-center">
      <div className="w-12 h-12 rounded-xl bg-green-100 dark:bg-green-900 flex items-center justify-center mr-4">
        <Icon className="w-6 h-6 text-green-600 dark:text-green-400" />
      </div>
      <div className="flex flex-col justify-center">
        <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
          {title}
        </h3>
        <p className="text-xs text-gray-600 dark:text-gray-400">
          {description}
        </p>
      </div>
    </div>
  );
} 
