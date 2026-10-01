import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Use absolute base on Vercel so deep links (e.g. /runs/...)
// load assets from /assets/... instead of /runs/.../assets/...
// Fall back to relative base for GitHub Pages and local static hosting.
const isVercel = process.env.VERCEL === "1";

export default defineConfig({
  plugins: [react()],
  base: isVercel ? "/" : "./",
});
