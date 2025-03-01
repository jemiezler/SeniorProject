"use client";

import Navbar from "@/components/layout/Navbar";
import React, { memo } from "react";
import Image from "next/image";
import Link from "next/link";
import { Upload, Search, CheckCircle, Lightbulb, Soup, FileUp, Check, ChartArea } from "lucide-react";
import Button from "@/components/ui/Button";
import { FeatureCard } from "@/components/ui/FeatureCard";

const MemoizedFeatureCard = memo(FeatureCard);

const Homepage = () => {
  return (
    <div className="flex flex-col items-center py-10 px-4">
      <div className="w-full max-w-screen-lg text-white flex flex-col items-center space-y-12">
        <div className="w-full flex justify-center">
          <Image
            src="/Depth 6, Frame 0.png"
            alt="Is your kale fresh?"
            width={928}
            height={480}
            priority
            sizes="(max-width: 768px) 100vw, 928px"
            className="max-w-full h-auto rounded-lg"
          />
        </div>

        <section className="w-full text-left">
          <h2 className="text-2xl font-bold font-epilogue pl-6">How it works</h2>
          <div className="flex flex-wrap justify-center gap-6 w-full">
            <MemoizedFeatureCard icon={Search} title="Upload your image" description="" />
            <MemoizedFeatureCard icon={FileUp} title="We'll analyze it" description="" />
            <MemoizedFeatureCard icon={Check} title="We'll tell you if it's fresh" description="" />
            <MemoizedFeatureCard icon={Lightbulb} title="Tips to keep it fresh longer" description="" />
          </div>
        </section>

        <div>
          <Link href="/Upload-kale">
            <Button text="Start Freshness Test" />
          </Link>
        </div>

        <section className="w-full text-left">
          <h2 className="text-2xl font-bold font-epilogue pl-6">What is FreshKale?</h2>
          <p className="text-[16px] font-regular font-epilogue pl-6">
            FreshKale is a fun, easy way to check the freshness of your kale. Just upload an image, and we'll analyze it for you.
            We’ll also give you tips on how to keep it fresh longer. Whether you're a seasoned chef or just starting out,
            FreshKale helps you make the most of your leafy greens.
          </p>
        </section>

        <section className="w-full text-left">
          <h2 className="text-2xl font-bold font-epilogue pl-6">Why use FreshKale?</h2>
          <div className="flex flex-wrap justify-center gap-6 w-full">
            <MemoizedFeatureCard icon={ChartArea} title="Keep your food fresh" description="" />
            <MemoizedFeatureCard icon={Search} title="Quickly find what you need" description="" />
            <MemoizedFeatureCard icon={CheckCircle} title="Create delicious dishes" description="" />
            <MemoizedFeatureCard icon={Soup} title="Enjoy your meals more" description="" />
          </div>
        </section>

        <div className="flex flex-wrap justify-center gap-6 w-full">
          {["/Kale1.png", "/kale2.png", "/Kale3.png"].map((src, index) => (
            <div key={index} className="w-[30%] h-auto min-w-[150px]">
              <Image
                src={src}
                alt={`Kale${index + 1}`}
                width={301}
                height={402}
                loading="lazy"
                sizes="(max-width: 768px) 100vw, 301px"
                className="w-full h-auto object-cover rounded-lg"
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Homepage;
