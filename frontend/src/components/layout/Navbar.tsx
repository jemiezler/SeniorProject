'use client';

import React from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { Sun, Moon, Menu } from 'lucide-react';
import Image from 'next/image';

export function Navbar() {
  return (
    <div className="fixed w-full z-50 bg-white/80 dark:bg-gray-900/80 backdrop-blur-lg border-b border-green-100 dark:border-green-900">
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8"
      >
        <div className="flex items-center justify-between h-16">
          <Link href="/" className="flex items-center space-x-2">
            <div className="relative w-8 h-8">
              <Image 
                src="/mfu.png"
                alt="MFU Logo"
                fill
                className="rounded-full object-contain"
                priority
              />
            </div>
            <span className="text-2xl font-bold bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent">
              Kale AI
            </span>
          </Link>

          <div className="hidden md:flex items-center space-x-8">
            <Link 
              href="/about"
              className="text-gray-600 dark:text-gray-300 hover:text-green-600 dark:hover:text-green-400 transition-colors"
            >
              About Project
            </Link>
            <Link 
              href="/how-it-works"
              className="text-gray-600 dark:text-gray-300 hover:text-green-600 dark:hover:text-green-400 transition-colors"
            >
              How It Works
            </Link>
            <Link 
              href="/team"
              className="text-gray-600 dark:text-gray-300 hover:text-green-600 dark:hover:text-green-400 transition-colors"
            >
              Our Team
            </Link>
          </div>

          <div className="flex items-center space-x-4">
            <button
              className="p-2 rounded-lg bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400 hover:bg-green-200 dark:hover:bg-green-800 transition-colors"
              aria-label="Toggle theme"
            >
              <Sun className="h-5 w-5 hidden dark:block" />
              <Moon className="h-5 w-5 block dark:hidden" />
            </button>
            
            <Link
              href="/analyze"
              className="hidden md:block px-6 py-2 rounded-lg bg-gradient-to-r from-green-600 to-emerald-600 text-white font-medium hover:opacity-90 transition-opacity"
            >
              Start Analysis
            </Link>

            <button className="md:hidden p-2">
              <Menu className="h-6 w-6" />
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
} 