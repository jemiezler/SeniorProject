"use client";

import React from "react";
import { Card, CardHeader, CardBody, Image } from "@heroui/react";

interface CardProps {
  title: string;
  subtitle: string;
  tracks: number;
  imageUrl: string;
}

const SampleCard: React.FC<CardProps> = ({ title, subtitle, tracks, imageUrl }) => {
  return (
    <Card className="py-4 shadow-lg backdrop-blur-md bg-white/30 border border-white/40 rounded-xl p-3">
      <CardHeader className="pb-0 pt-2 px-4 flex-col items-start text-black">
        <p className="text-tiny uppercase font-bold">{subtitle}</p>
        <small className="text-default">{tracks} Tracks</small>
        <h4 className="font-bold text-large">{title}</h4>
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
