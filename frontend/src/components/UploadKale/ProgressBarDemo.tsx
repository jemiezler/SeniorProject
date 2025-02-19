// components/ProgressBarDemo.tsx
import React from "react";
import { motion } from "framer-motion";

interface ProgressBarProps {
  progress: number; // Accept a number instead of a MotionValue
}

export const ProgressBar = ({ progress }: ProgressBarProps) => {
  return (
    <motion.div
      style={{ width: `${progress}%` }}
      className="bg-progressGreen h-4 rounded-full mt-4"
    />
  );
};

// Optionally, if you want to keep a demo version that uses a MotionValue,
// you can separate it into its own component.
