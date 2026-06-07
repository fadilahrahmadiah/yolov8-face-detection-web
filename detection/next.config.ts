const nextConfig: any = {
  turbopack: {},

  typescript: {
    ignoreBuildErrors: true,
  },

  webpack: (config: any, { isServer }: any) => {
    config.resolve.alias = {
      ...config.resolve.alias,
    };

    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
      };
    }

    return config;
  },
};

export default nextConfig;