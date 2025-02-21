"use client";

import Navbar from "@/components/Default/Navbar";
import React from "react";
import Image from "next/image";
import Link from "next/link";
import StepBox from "@/components/Default/StepBox";
import {
  Upload,
  Search,
  CheckCircle,
  Lightbulb,
  Soup,
  FileUp,
  Check,
  ChartArea,
} from "lucide-react";
import Button from "@/components/Default/Button";
import { Spotlight } from "@/components/Default/Spotlight";
import ImageCard from "@/components/Default/ImageHover";

const Homepage = () => {
  return (
    <div>
      <Spotlight />
      <div className="flex flex-col items-center py-10 px-4">
        <div className="w-full max-w-screen-lg text-white flex flex-col items-center space-y-12">
          {/* Hero Image */}
          <div className="w-full flex justify-center">
            <Image
              src="/Depth 6, Frame 0.png"
              alt="Is your kale fresh?"
              width={980}
              height={480}
              priority
              className="max-w-full h-auto rounded-lg"
            />
          </div>

          {/* How it works */}
          <div className="w-full text-left">
            <h2 className="text-2xl font-bold font-epilogue pl-6">
              How it works
            </h2>
          </div>

          <div className="flex flex-wrap gap-10 w-full justify-center">
            <StepBox
              icon={<Search size={24} color="white" />}
              text="Upload your image"
            />
            <StepBox
              icon={<FileUp size={24} color="white" />}
              text="We'll analyze it"
            />
            <StepBox
              icon={<Check size={24} color="white" />}
              text="We'll tell you if it's fresh"
            />
            <StepBox
              icon={<Lightbulb size={24} color="white" />}
              text="Tips to keep it fresh longer"
            />
          </div>

          {/* Start Freshness Test Button */}
          <div>
            <Link href="/upload-kale">
              <Button text="Start Freshness Test" />
            </Link>
          </div>

          {/* What is FreshKale? */}
          <div className="w-full text-left">
            <h2 className="text-2xl font-bold font-epilogue pl-6 mt-[-1rem]">
              What is FreshKale?
            </h2>
          </div>

          <div className="text-[16px] font-regular font-epilogue pl-6">
            <p>
              FreshKale is a fun, easy way to check the freshness of your kale.
              Just upload an image, and we'll analyze it for you. We’ll also
              give you tips on how to keep it fresh longer. Whether you're a
              seasoned chef or just starting out, FreshKale helps you make the
              most of your leafy greens.
            </p>
          </div>

          {/* Why use FreshKale? */}
          <div className="w-full text-left">
            <h2 className="text-2xl font-bold font-epilogue pl-6">
              Why use FreshKale?
            </h2>
          </div>

          <div className="flex flex-wrap justify-center gap-10 w-full">
            <StepBox
              icon={<ChartArea size={24} color="white" />}
              text="Keep your food fresh"
            />
            <StepBox
              icon={<Search size={24} color="white" />}
              text="Quickly find what you need"
            />
            <StepBox
              icon={<CheckCircle size={24} color="white" />}
              text="Create delicious dishes"
            />
            <StepBox
              icon={<Soup size={24} color="white" />}
              text="Enjoy your meals more"
            />
          </div>

          {/* Images */}
          <div className="flex flex-wrap justify-center gap-6 w-full">
            <ImageCard src="/Kale1.png" title="Kale One" />
            <ImageCard src="/Kale2.png" title="Kale Two" />
            <ImageCard src="/Kale3.png" title="Kale Three" />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Homepage;
