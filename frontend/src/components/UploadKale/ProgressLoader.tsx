import React, { useEffect, useState } from "react";
import { Progress } from "@heroui/react";

interface ProgressLoaderProps {
  isUploading: boolean;
  onComplete: () => void;
}

const ProgressLoader: React.FC<ProgressLoaderProps> = ({ isUploading, onComplete }) => {
  const [value, setValue] = useState(0);

  useEffect(() => {
    if (isUploading) {
      setValue(0);
      const interval = setInterval(() => {
        setValue((v) => {
          if (v >= 100) {
            clearInterval(interval);
            onComplete(); 
            return 100;
          }
          return v + 10;
        });
      }, 300);

      return () => clearInterval(interval);
    }
  }, [isUploading, onComplete]);

  return (
    <div className="w-full max-w-[928px] mt-4">
      {isUploading && (
        <Progress
          aria-label="Uploading..."
          className="w-full"
          color="success"
          showValueLabel={true}
          size="md"
          value={value}
        />
      )}
    </div>
  );
};

export default ProgressLoader;
