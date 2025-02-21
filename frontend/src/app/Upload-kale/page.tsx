"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import SubmitButton from "@/components/UploadKale/SubmitButton";
import ProgressLoader from "@/components/UploadKale/ProgressLoader";
import Input from "@/components/Default/Input";
import Button from "@/components/Default/Button";

const UploadKale = () => {
  const [image, setImage] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [isCompleted, setIsCompleted] = useState(false);
  const router = useRouter();

  const handleImageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      const file = event.target.files[0];
      setImage(file);

      const reader = new FileReader();
      reader.onloadend = () => {
        setImageUrl(reader.result as string);
        localStorage.setItem("imageUrl", reader.result as string); // Save to localStorage
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
        localStorage.setItem("imageUrl", reader.result as string); // Save to localStorage
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

      setTimeout(() => {
        setIsUploading(false);
        setIsCompleted(true);

        setTimeout(() => {
          router.push("/result");
        }, 3000);
      }, 3000);
    }
  };

  return (
    <div className="bg-customGreen min-h-screen flex flex-col items-center pt-10 pb-5 px-4">
      <h1 className="text-lg md:text-2xl lg:text-3xl font-bold font-epilogue text-white mb-6 text-center">
        Evaluate Kale Freshness
      </h1>
      <span className="text-white text-center">
        Upload a photo of your kale to receive a freshness rating from 1-5. 1 is
        the least fresh and 5 is the freshest.
      </span>

      <div
        className={`relative flex flex-col justify-center items-center w-full max-w-[928px] h-[200px] md:h-[250px] lg:h-[309px] border-2 mt-5 rounded-lg p-4 transition
      ${
        isDragging
          ? "border-solid border-stroke2 bg-opacity-20 bg-white"
          : "border-dashed border-stroke2"
      }`}
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
            Upload an image of your kale (png, jpg, jpeg)
          </p>
        )}

        <input
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
      </div>

      <ProgressLoader isUploading={isUploading} onComplete={() => {}} />

      {isCompleted && (
        <div className="mt-6 text-white text-xl font-bold blinking font-epilogue flex items-center justify-center text-center">
          Upload completed! <br />
          Hold a second for your result
        </div>
      )}

      {!isUploading && (
        <div className="mt-10">
          <Button text="Submit" onClick={handleSubmit} />
        </div>
      )}
    </div>
  );
};

export default UploadKale;
