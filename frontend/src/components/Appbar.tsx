"use client";

import React from "react";
import {
  Navbar,
  NavbarBrand,
  NavbarContent,
  NavbarItem,
  NavbarMenuToggle,
  NavbarMenu,
  NavbarMenuItem,
  Link,
} from "@heroui/react";

export default function AppBar() {
  const [isMenuOpen, setIsMenuOpen] = React.useState(false);

  return (
    <Navbar
      onMenuOpenChange={setIsMenuOpen}
      className="shadow-lg backdrop-blur-lg bg-white/30 border border-white/40"
    >
      {/* Logo & Mobile Menu Toggle */}
      <NavbarContent>
        <NavbarMenuToggle
          aria-label={isMenuOpen ? "Close menu" : "Open menu"}
          className="sm:hidden"
        />
        <NavbarBrand>
          <p className="text-2xl font-bold">KaleCheck</p>
        </NavbarBrand>
      </NavbarContent>

      {/* Desktop Navigation */}
      <NavbarContent className="hidden sm:flex gap-6" justify="center">
        <NavbarItem>
          <Link className="text-gray hover:text-blue-700 transition-colors" href="/">
            Home
          </Link>
        </NavbarItem>
        <NavbarItem>
          <Link
            className="text-gray hover:text-blue-700 transition-colors"
            href="/learn"
          >
            Learn
          </Link>
        </NavbarItem>
        <NavbarItem>
          <Link
            className="text-gray hover:text-blue-700 transition-colors"
            href="/docs"
          >
            API Docs
          </Link>
        </NavbarItem>
      </NavbarContent>

      {/* Mobile Menu */}
      <NavbarMenu>
        <NavbarMenuItem>
          <Link
            className="w-full text-gray hover:text-blue-700 transition-colors"
            href="/"
            size="lg"
          >
            Home
          </Link>
        </NavbarMenuItem>
        <NavbarMenuItem>
          <Link
            className="w-full text-gray hover:text-blue-700 transition-colors"
            href="/learn"
            size="lg"
          >
            Learn
          </Link>
        </NavbarMenuItem>
        <NavbarMenuItem>
          <Link
            className="w-full text-gray hover:text-blue-700 transition-colors"
            href="/docs"
            size="lg"
          >
            API Docs
          </Link>
        </NavbarMenuItem>
      </NavbarMenu>
    </Navbar>
  );
}
