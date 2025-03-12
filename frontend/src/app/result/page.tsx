"use client"
import React, { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { HoverEffect } from "@/components/ui/card-hover-effect";
import Button from "@/components/ui/Button";

const fetchFreshnessData = async () => {
  const response = await fetch("https://catfact.ninja/fact");
  const data = await response.json();
  return data.fact || "No information available";
};

const Result = () => {
  const [freshnessRating, setFreshnessRating] = useState<number | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [items, setItems] = useState<{ title: string; description: string; link: string }[]>([]);
  const router = useRouter();

  useEffect(() => {
    setFreshnessRating(Math.floor(Math.random() * 5) + 1);


    if (typeof window !== "undefined") {
      const storedImageUrl = localStorage.getItem("imageUrl");
      if (storedImageUrl) {
        setImageUrl(storedImageUrl);
      }
    }

    // ดึงข้อมูล freshness จาก API
    fetchFreshnessData().then((fact) =>
      setItems([
        { title: "Date of purchase", description: fact, link: "/purchase" },
        { title: "Expiration date", description: fact, link: "/expiration" },
        { title: "Estimated freshness", description: fact, link: "/freshness" },
      ])
    );
  }, []);

  const handleGoBack = useCallback(() => {
    router.push("/");
  }, [router]);

  return (
    <div className="bg-customGreen min-h-screen flex flex-col items-center pt-10 pb-5 px-4">
      <h1 className="text-lg md:text-2xl lg:text-3xl font-bold font-epilogue text-white mb-6 text-center">
        Kale Freshness Result
      </h1>

      <div className="relative flex flex-col justify-center items-center w-full max-w-[928px] h-[200px] md:h-[250px] lg:h-[309px] border-2 rounded-lg p-4">
        {imageUrl ? (
          <img
            src={imageUrl}
            alt="Preview"
            className="w-full h-full object-contain rounded-md"
          />
        ) : (
          <p className="text-white text-center font-epilogue font-bold text-sm md:text-base">
            No image uploaded.
          </p>
        )}
      </div>

      <div className="mt-6 text-white text-xl font-bold">
        Your kale freshness rating is: {freshnessRating} / 5
      </div>

      <div className="mt-6 w-full max-w-[928px]">
        <HoverEffect items={items} className="h-full w-full" />
      </div>

      <div className="mt-6">
        <Button text="Go back to Upload" onClick={handleGoBack} />
      </div>
    </div>
  );
};

export default Result;
