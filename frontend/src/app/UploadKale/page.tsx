"use client";
import React, { useState } from "react";
import Navbar from "@/components/navbar";
import SubmitButton from "@/components/SubmitButton";
import { ProgressBar } from "@/components/ProgressBarDemo"; 

const UploadKale = () => {
  const [image, setImage] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0); 
  const [imageUrl, setImageUrl] = useState<string | null>(null); 

  const handleImageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      const file = event.target.files[0];
      setImage(file);

      
      const reader = new FileReader();
      reader.onloadend = () => {
        setImageUrl(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);

    if (event.dataTransfer.files.length > 0) {
      const file = event.dataTransfer.files[0];
      setImage(file);

      
      const reader = new FileReader();
      reader.onloadend = () => {
        setImageUrl(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleSubmit = () => {
    if (image) {
      setIsUploading(true);
      setProgress(0); // Reset progress on submit

      // Simulate progress update
      const interval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 100) {
            clearInterval(interval);
            return 100;
          }
          return prev + 10;
        });
      }, 500);

      console.log("Image uploaded:", image.name);
    }
  };

  return (
    <div className="bg-customGreen min-h-screen">
      <Navbar />
      <div className="flex flex-col justify-center items-center pt-10 pb-5 px-4 w-full">
        <h1 className="text-lg md:text-2xl lg:text-3xl font-bold font-epilogue text-white mb-6 text-center">
          Evaluate Kale Freshness
        </h1>
        <span>
          Upload a photo of your kale to receive a freshness rating from 1-5. 1 is the least fresh and 5 is the freshest.
        </span>

        <div
          className={`relative flex flex-col justify-center items-center w-full max-w-[928px] h-[200px] md:h-[250px] lg:h-[309px] border-2 rounded-lg p-4 transition 
            ${isDragging ? "border-solid border-white bg-opacity-20 bg-white" : "border-dashed border-strokeGreen"}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {imageUrl ? (
            <img
              src={imageUrl}
              alt="Preview"
              className="w-full h-full object-contain rounded-md"
            />
          ) : (
            <p className="text-white text-center font-epilogue font-bold text-sm md:text-base">
              Upload an image of your kale file (png, jpg, jpeg)
            </p>
          )}

          
          <input
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
        </div>

        
        {isUploading && (
          <div className="w-full max-w-[928px] mt-4">
            <ProgressBar progress={progress} /> 
            <p className="text-white text-center mt-2">{`Uploading: ${progress}%`}</p>
          </div>
        )}

       
        <div className="mt-6">
          <SubmitButton text="Submit" variant="primary" onClick={handleSubmit} />
        </div>
      </div>
    </div>
  );
};

export default UploadKale;
