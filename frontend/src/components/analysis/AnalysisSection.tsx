'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Loader2, Check, X, Image as ImageIcon } from 'lucide-react';
import type { AnalysisResult } from '@/types';

interface AnalysisSectionProps {
  selectedFiles: File[];
  previews: string[];
  isAnalyzing: boolean;
  progress: number;
  results: AnalysisResult[];
  onFileSelect: (files: File[]) => void;
  onRemoveFile: (index: number) => void;
  onAnalyze: () => void;
}

export function AnalysisSection({
  selectedFiles,
  previews,
  isAnalyzing,
  progress,
  results,
  onFileSelect,
  onRemoveFile,
  onAnalyze,
}: AnalysisSectionProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files).filter(file => file.type.startsWith('image/'));
    if (files.length > 0) {
      onFileSelect(files);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files ? Array.from(e.target.files) : [];
    if (files.length > 0) {
      onFileSelect(files);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm rounded-2xl p-6 border border-green-100 dark:border-green-900"
      >
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
          {/* Upload Section */}
          <div className="space-y-6">
            <h2 className="text-2xl font-semibold gradient-text">
              Upload Your Kale Images
            </h2>
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              className={`relative h-64 border-2 border-dashed rounded-xl flex flex-col items-center justify-center transition-colors duration-200 ${
                isDragging
                  ? 'border-green-400 bg-green-50 dark:border-green-600 dark:bg-green-900/20'
                  : 'hover:border-green-400 dark:hover:border-green-600'
              }`}
            >
              <Upload className="w-12 h-12 text-green-600 mb-4" />
              <p className="text-gray-600 dark:text-gray-400">
                Drag and drop your images here
              </p>
              <p className="text-gray-500 dark:text-gray-500 text-sm mt-2">or</p>
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
                id="file-upload"
                multiple
              />
              <label
                htmlFor="file-upload"
                className="mt-2 px-4 py-2 bg-green-100 dark:bg-green-900 rounded-lg text-green-600 dark:text-green-400 hover:bg-green-200 dark:hover:bg-green-800 transition-colors cursor-pointer"
              >
                Browse Files
              </label>
            </div>

            {/* Selected Files Preview */}
            {selectedFiles.length > 0 && (
              <div className="space-y-4">
                <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
                  Selected Images ({selectedFiles.length})
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  {previews.map((preview, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="relative group"
                    >
                      <div className="relative aspect-square rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700">
                        <img
                          src={preview}
                          alt={`Preview ${index + 1}`}
                          className="w-full h-full object-cover"
                        />
                        <button
                          onClick={() => onRemoveFile(index)}
                          className="absolute top-2 right-2 p-1 bg-red-100 dark:bg-red-900 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                        >
                          <X className="w-4 h-4 text-red-600 dark:text-red-400" />
                        </button>
                      </div>
                      <p className="mt-1 text-sm text-gray-500 truncate">
                        {selectedFiles[index].name}
                      </p>
                    </motion.div>
                  ))}
                </div>
              </div>
            )}

            <button
              onClick={onAnalyze}
              disabled={selectedFiles.length === 0 || isAnalyzing}
              className="w-full py-3 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-xl font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90 transition-opacity flex items-center justify-center space-x-2"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <ImageIcon className="w-5 h-5" />
                  <span>Analyze Images</span>
                </>
              )}
            </button>

            {/* Progress Bar */}
            {isAnalyzing && (
              <div className="space-y-2">
                <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${progress}%` }}
                    className="h-full bg-gradient-to-r from-green-500 to-emerald-500"
                  />
                </div>
                <p className="text-sm text-gray-500 dark:text-gray-400 text-center">
                  {progress}% Complete
                </p>
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            <h2 className="text-2xl font-semibold gradient-text">
              Analysis Results
            </h2>
            <AnimatePresence>
              {results.length > 0 ? (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="space-y-6"
                >
                  {results.map((result, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="glass p-4 rounded-xl space-y-4"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2 text-green-600">
                          <Check className="w-6 h-6" />
                          <span className="font-medium">Analysis #{index + 1}</span>
                        </div>
                        <span className="text-sm text-gray-500">
                          {selectedFiles[index]?.name}
                        </span>
                      </div>

                      <div className="grid gap-4">
                        <div>
                          <h3 className="text-gray-600 dark:text-gray-400 mb-2">Prediction</h3>
                          <p className="text-2xl font-semibold gradient-text">{result.prediction}</p>
                        </div>
                        <div>
                          <h3 className="text-gray-600 dark:text-gray-400 mb-2">Confidence</h3>
                          <p className="text-2xl font-semibold gradient-text">
                            {(result.confidence * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div>
                          <h3 className="text-gray-600 dark:text-gray-400 mb-2">Recommendations</h3>
                          <ul className="space-y-2">
                            {result.recommendations.map((rec, recIndex) => (
                              <li key={recIndex} className="flex items-center space-x-2">
                                <Check className="w-4 h-4 text-green-600" />
                                <span className="text-gray-700 dark:text-gray-300">{rec}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </motion.div>
              ) : (
                <div className="h-full flex items-center justify-center text-gray-500 dark:text-gray-400">
                  <p>Upload and analyze images to see the results</p>
                </div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </motion.div>
    </div>
  );
} 