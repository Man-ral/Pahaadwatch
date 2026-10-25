import type { Metadata } from "next";
import "leaflet/dist/leaflet.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "PahaadWatch | Himalayan Landslide Risk",
  description:
    "Explore an experimental landslide-risk assessment for locations in Uttarakhand and Himachal Pradesh using satellite rainfall and terrain features.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
