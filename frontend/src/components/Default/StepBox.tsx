import React from "react";

interface StepBoxProps {
  text: string;
  icon: React.ReactNode;
}

const StepBox: React.FC<StepBoxProps> = ({ text, icon }) => {
  return (
    <div className="w-full max-w-[200px] bg-bgBoxGreen border-2 border-Greenstokre rounded-lg shadow-lg flex flex-col items-center gap-y-4 p-3 mt-[-0.5rem] sm:p-2 transition-transform duration-300 hover:scale-105 hover:shadow-green-500/50">
      <div className="flex items-center justify-center w-10 h-10 border rounded-full mt-2">
        {icon}
      </div>

      <h3 className="text-[16px] font-bold font-epilogue text-white text-center">
        {text}
      </h3>
    </div>
  );
};

export default StepBox;
