/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        positive: "#4CAF50",
        neutral:  "#9E9E9E",
        negative: "#F44336",
        brand:    "#3F51B5",
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
}

