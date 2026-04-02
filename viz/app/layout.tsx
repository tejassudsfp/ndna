import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const jetbrains = JetBrains_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Neural DNA",
  description:
    "A 354-parameter genome controls 35.4M connections in GPT-2 Small. Interactive visualization of learned neural network topology. 99,970:1 compression. Beats GPT-2 on 3 benchmarks.",
  keywords: [
    "NDNA",
    "Neural DNA",
    "neural architecture search",
    "sparse neural networks",
    "GPT-2",
    "genome",
    "network topology",
    "lottery ticket hypothesis",
    "pruning",
    "machine learning",
    "deep learning",
    "transformer",
    "connectivity",
  ],
  authors: [
    {
      name: "Tejas Parthasarathi Sudarshan",
      url: "https://tejassuds.com",
    },
  ],
  creator: "Tejas Parthasarathi Sudarshan",
  openGraph: {
    type: "website",
    title: "Neural DNA",
    description:
      "Watch a 354-parameter genome learn which connections matter in GPT-2. 99,970:1 compression ratio. Beats GPT-2 on 3 of 9 benchmarks with 33% of connections permanently off.",
    siteName: "NDNA Visualization",
    url: "https://ndna.tejassuds.com",
  },
  twitter: {
    card: "summary_large_image",
    title: "Neural DNA",
    description:
      "A 354-parameter genome controls 35.4M connections in GPT-2. Interactive visualization of learned topology.",
    creator: "@tejassuds",
  },
  metadataBase: new URL("https://ndna.tejassuds.com"),
  alternates: {
    canonical: "/",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrains.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
