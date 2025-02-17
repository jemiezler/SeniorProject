"use client";

import React from "react";
import { Card, CardHeader, CardBody, Image } from "@heroui/react";

interface CardProps {
  title: string;
  subtitle: string;
  imageUrl: string;
}

const SampleCard: React.FC<CardProps> = ({ title, subtitle, imageUrl }) => {
  return (
    <Card className="py-4 shadow-lg backdrop-blur-md bg-white/30 border border-white/40 rounded-xl p-3 transition-transform duration-300 ease-in-out hover:scale-105 hover:shadow-2xl">
      <CardHeader className="pb-0 pt-2 px-4 flex-col items-start">
        <p className="text-tiny uppercase font-bold">{subtitle}</p>
        <h4 className="font-bold text-large text-gray-300">{title}</h4>
      </CardHeader>
      <CardBody className="overflow-visible py-2">
        <Image
          alt="Card background"
          className="object-cover rounded-xl"
          src={imageUrl}
          width={270}
          height={150}
        />
      </CardBody>
    </Card>
  );
};

export default SampleCard;
