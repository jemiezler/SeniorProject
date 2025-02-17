"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

const Result = () => {
  const [freshnessRating, setFreshnessRating] = useState<number | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const router = useRouter();
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    setFreshnessRating(Math.floor(Math.random() * 5) + 1);
    setImageUrl(localStorage.getItem("imageUrl") || null);
  }, []);

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(true);
  };

  return (
    <div className="bg-customGreen min-h-screen flex flex-col items-center pt-10 pb-5 px-4">
      <h1 className="text-lg md:text-2xl lg:text-3xl font-bold font-epilogue text-white mb-6 text-center">
        Kale Freshness Result
      </h1>

      <div
        className={`relative flex flex-col justify-center items-center w-full max-w-[928px] h-[200px] md:h-[250px] lg:h-[309px] border-2 rounded-lg p-4 transition
          ${
            isDragging
              ? "border-solid border-stroke2 bg-opacity-20 bg-white"
              : "border-dashed border-stroke2"
          }`}
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
      </div>

      {freshnessRating ? (
        <div className="mt-6 text-white text-xl font-bold">
          Your kale freshness rating is: {freshnessRating} / 5
        </div>
      ) : (
        <div className="mt-6 text-white text-xl font-bold">
          Loading freshness rating...
        </div>
      )}

      <div className="mt-6">
        <button
          onClick={() => router.push("/")}
          className="bg-white text-customGreen py-2 px-4 rounded-lg font-bold hover:bg-opacity-90"
        >
          Go Back to Upload
        </button>
      </div>
    </div>
  );
};

export default Result;
