
import React from "react";

interface SubmitButtonProps {
  text: string;
  onClick?: () => void;
  variant?: "primary" | "secondary" | "outline" | "homepage";
  size?: "sm" | "md" | "lg";
}

const SubmitButton: React.FC<SubmitButtonProps> = ({ text, onClick, variant = "primary", size = "md" }) => {
  const baseStyles = "font-bold rounded-lg transition-all duration-300";
  const sizeStyles = {
    sm: "px-3 py-1 text-sm",
    md: "px-4 py-2 text-base",
    lg: "px-6 py-3 text-lg",
  };

  const variantStyles = {
    primary: "bg-buttonGreen text-white hover:bg-buttonGreen",
    secondary: "bg-gray-500 text-white hover:bg-gray-600",
    outline: "border-2 border-gray-500 text-gray-500 hover:bg-gray-100",
    homepage: "bg-bgBoxGreen text-white hover:bg-bgBoxGreen rounded-[40px]"
  };

  return (
    <button 
      onClick={onClick} 
      className={`${baseStyles} ${sizeStyles[size]} ${variantStyles[variant]}`}
    >
      {text}
    </button>
  );
};

export default SubmitButton; 
