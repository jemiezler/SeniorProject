"use client";

import SampleCard from "@/components/SampleCard";
import { Avatar, Badge, Button } from "@heroui/react";

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-900 flex flex-col items-center p-6 space-y-8">
      <div className="flex gap-3 items-center">
        <Badge color="default" content="5">
          <Avatar
            radius="md"
            src="https://i.pravatar.cc/150?u=a042f81f4e29026024d"
          />
        </Badge>
        <Badge color="primary" content="5">
          <Avatar
            radius="md"
            src="https://i.pravatar.cc/150?u=a04258a2462d826712d"
          />
        </Badge>
        <Badge color="secondary" content="5">
          <Avatar
            radius="md"
            src="https://i.pravatar.cc/300?u=a042581f4e29026709d"
          />
        </Badge>
        <Badge color="success" content="5">
          <Avatar
            radius="md"
            src="https://i.pravatar.cc/150?u=a04258114e29026302d"
          />
        </Badge>
        <Badge color="warning" content="5">
          <Avatar
            radius="md"
            src="https://i.pravatar.cc/150?u=a04258114e29026708c"
          />
        </Badge>
        <Badge color="danger" content="5">
          <Avatar
            radius="md"
            src="https://i.pravatar.cc/150?u=a042581f4e29026024d"
          />
        </Badge>
      </div>
      <div className="flex justify-between gap-5">
        <Button color="default">Default</Button>
        <Button color="primary">Primary</Button>
        <Button color="secondary">Secondary</Button>
      </div>
      <div className="flex justify-center items-center gap-5">
        <SampleCard
          title="Frontend Radio"
          subtitle="Daily Mix"
          tracks={12}
          imageUrl="https://picsum.photos/id/870/200/300?grayscale&blur=2"
        />
        <SampleCard
          title="Frontend Radio"
          subtitle="Daily Mix"
          tracks={12}
          imageUrl="https://picsum.photos/id/870/200/300?grayscale&blur=2"
        />
        <SampleCard
          title="Frontend Radio"
          subtitle="Daily Mix"
          tracks={12}
          imageUrl="https://picsum.photos/id/870/200/300?grayscale&blur=2"
        />
      </div>
    </div>
  );
}
