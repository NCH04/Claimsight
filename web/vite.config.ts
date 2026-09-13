import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// base './': le bundle est servi par FastAPI depuis web/dist (StaticFiles)
export default defineConfig({
  plugins: [react()],
  base: "./",
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
