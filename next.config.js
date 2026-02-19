/** @type {import('next').NextConfig} */
const nextConfig = {
  // Transpile three.js and its dependencies
  transpilePackages: ['three'],
  webpack: (config) => {
    return config
  },
}

export default nextConfig

