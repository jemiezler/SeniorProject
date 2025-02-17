"use client";

import Banner from "@/components/Banner";
import BenefitCard from "@/components/CardBenefit";
import SampleCard from "@/components/SampleCard";
import { Button } from "@heroui/react";

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center p-8 space-y-10">
      {/* Hero Section */}
      <Banner />
      <Button color="success" className="mt-4">
        Start Freshness Test
      </Button>
      {/* How it works */}
      <div className="w-full max-w-3xl text-center">
        <h2 className="text-2xl font-semibold">How it works</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
          <SampleCard
            title="Upload Your Image"
            subtitle="Step 1"
            imageUrl="https://picsum.photos/200"
          />
          <SampleCard
            title="We’ll Analyze It"
            subtitle="Step 2"
            imageUrl="https://picsum.photos/201"
          />
          <SampleCard
            title="Get Freshness Tips"
            subtitle="Step 3"
            imageUrl="https://picsum.photos/202"
          />
        </div>
      </div>

      {/* About FreshKale */}
      <div className="w-full max-w-3xl text-center bg-white/20 backdrop-blur-lg p-6 rounded-lg">
        <h2 className="text-2xl font-semibold">What is FreshKale?</h2>
        <p className="mt-2 text-gray-300">
          FreshKale is a fun, easy way to check the freshness of your kale. Just
          upload an image, and we'll analyze it for you. We'll also give you
          tips on how to keep it fresh longer.
        </p>
      </div>

      {/* Why use FreshKale? */}
      <div className="w-full max-w-3xl text-center">
        <h2 className="text-2xl font-semibold">Why use FreshKale?</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 mt-6">
          <BenefitCard
            title="Keep Food Fresh"
            subtitle="Benefit 1"
            imageUrl="https://picsum.photos/203"
          />
          <BenefitCard
            title="Find What You Need"
            subtitle="Benefit 2"
            imageUrl="https://picsum.photos/204"
          />
          <BenefitCard
            title="Create Dishes"
            subtitle="Benefit 3"
            imageUrl="https://picsum.photos/205"
          />
          <BenefitCard
            title="Enjoy Meals More"
            subtitle="Benefit 4"
            imageUrl="https://picsum.photos/206"
          />
        </div>
      </div>
    </div>
  );
}
