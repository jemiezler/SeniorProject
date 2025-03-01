import React from "react";

interface StepBoxProps {
  text: string;
  icon: React.ReactNode;
}

const StepBox: React.FC<StepBoxProps> = ({ text, icon }) => {
  return (
    <div className="w-[223px] h-[94px] bg-bgBoxGreen border-2 border-solid border-Greenstokre rounded-lg shadow-lg flex justify-center items-center gap-x-3">
      {icon}
      <h3 className="text-[16px] font-bold font-epilogue">{text}</h3>
    </div>
  );
};

export default StepBox;
