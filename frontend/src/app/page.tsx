"use client";

import Navbar from "@/components/layout/Navbar";
import React from "react";
import Image from "next/image";
import Link from "next/link";
import StepBox from "@/components/ui/StepBox";
import { Upload, Search, CheckCircle, Lightbulb, Soup, FileUp, Check, ChartArea } from "lucide-react";
import Button from "@/components/ui/Button";
import { FeatureCard } from "@/components/ui/FeatureCard";

const Homepage = () => {
  return (
    <div>
      <div className="flex flex-col items-center py-10 px-4">
        <div className="w-full max-w-screen-lg text-white flex flex-col items-center space-y-12">

          {/* Hero Image */}
          <div className="w-full flex justify-center">
            <Image
              src="/Depth 6, Frame 0.png"
              alt="Is your kale fresh?"
              width={928}
              height={480}
              priority
              className="max-w-full h-auto rounded-lg"
            />
          </div>

          {/* How it works */}
          <div className="w-full text-left">
            <h2 className="text-2xl font-bold font-epilogue pl-6">How it works</h2>
          </div>

          <div className="flex flex-wrap justify-center gap-6 w-full">
            <FeatureCard icon={Search} title="Upload your image" description={""} />
            <FeatureCard icon={FileUp} title="We'll analyze it" description={""}/>
            <FeatureCard icon={Check} title="We'll tell you if it's fresh" description={""}/>
            <FeatureCard icon={Lightbulb} title="Tips to keep it fresh longer" description={""}/>
          </div>

          {/* Start Freshness Test Button */}
          <div>
            <Link href="/Upload-kale">
              <Button text="Start Freshness Test" />
            </Link>
          </div>

          {/* What is FreshKale? */}
          <div className="w-full text-left">
            <h2 className="text-2xl font-bold font-epilogue pl-6">What is FreshKale?</h2>
          </div>

          <div className="text-[16px] font-regular font-epilogue pl-6">
            <p>
              FreshKale is a fun, easy way to check the freshness of your kale. Just upload an image, and we'll analyze it for you.
              We’ll also give you tips on how to keep it fresh longer. Whether you're a seasoned chef or just starting out,
              FreshKale helps you make the most of your leafy greens.
            </p>
          </div>

 
          <div className="w-full text-left">
            <h2 className="text-2xl font-bold font-epilogue pl-6">Why use FreshKale?</h2>
          </div>

          <div className="flex flex-wrap justify-center gap-6 w-full">
            <FeatureCard icon={ChartArea} title={"Keep your food fresh"} description={""}  />
            <FeatureCard icon={Search} title="Quickly find what you need" description={""}/>
            <FeatureCard icon={CheckCircle} title="Create delicious dishes"description={""} />
            <FeatureCard icon={Soup} title="Enjoy your meals more" description={""}/>
          </div>


          <div className="flex flex-wrap justify-center gap-6 w-full">
            <div className="w-[30%] h-auto min-w-[150px]">
              <Image
                src="/Kale1.png"
                alt="Kale1"
                width={301}
                height={402}
                loading="lazy"
                className="w-full h-auto object-cover rounded-lg"
              />
            </div>
            <div className="w-[30%] h-auto min-w-[150px]">
              <Image
                src="/kale2.png"
                alt="Kale2"
                width={301}
                height={402}
                loading="lazy"
                className="w-full h-auto object-cover rounded-lg"
              />
            </div>
            <div className="w-[30%] h-auto min-w-[150px]">
              <Image
                src="/Kale3.png"
                alt="Kale3"
                width={301}
                height={402}
                loading="lazy"
                className="w-full h-auto object-cover rounded-lg"
              />
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};

export default Homepage;
