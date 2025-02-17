import { Button } from "@heroui/react";

interface ButtonProps {
  text: string;
  color?: string; // Accepts Tailwind color classes
}

export default function App({ text, color = "bg-ButtonGreen text-white" }: ButtonProps) {
  return <Button className={`${color} font-bold font-epilogue px-4 py-2 rounded-[20px]`}>{text}</Button>;
}
