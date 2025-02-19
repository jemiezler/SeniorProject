import { Button } from "@heroui/react";

interface ButtonProps {
  text: string;
  color?: string; 
  onClick?: () => void; 
}

export default function App({ text, color = "bg-ButtonGreen text-white", onClick }: ButtonProps) {
  return (
    <Button 
      className={`${color} font-bold font-epilogue px-4 py-2 rounded-[20px] cursor-pointer`}
      onClick={onClick} 
    >
      {text}
    </Button>
  );
}
