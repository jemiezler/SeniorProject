import { heroui } from "@heroui/react";
import type { Config } from "tailwindcss";

export default {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./node_modules/@heroui/theme/dist/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        epilogue: ["Epilogue", "sans-serif"],
      },
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        bgGreen1: "#122112",
        bgBoxGreen: "#1A331C",
        stroke1: "#E5E8EB",
        Greenstokre :"#336636",
        ButtonGreen: "#016D06",
        stroke2:"#336636"
      },
    },
  },
  plugins: [heroui()],
} satisfies Config;
