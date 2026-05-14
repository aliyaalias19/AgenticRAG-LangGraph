/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',       
  env: {
    NEXT_PUBLIC_BACKEND_URL: process.env.BACKEND_URL ?? "http://localhost:8000",
  },
};

module.exports = nextConfig;