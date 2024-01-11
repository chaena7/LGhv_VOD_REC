module.exports = {
  mode: "jit",
  content: [
    "./src/**/**/*.{js,ts,jsx,tsx,html,mdx}",
    "./src/**/*.{js,ts,jsx,tsx,html,mdx}",
  ],
  darkMode: "class",
  theme: {
    screens: { md: { max: "1050px" }, sm: { max: "550px" } },
    extend: {
      colors: {
        white: { A700_4c: "#ffffff4c", A700_dd: "#ffffffdd", A700: "#ffffff" },
        gray: {
          50: "#fff9f9",
          100: "#f4f3f3",
          500: "#a8a8a8",
          800: "#534747",
          "800_cc": "#534747cc",
        },
        blue_gray: {
          100: "#cfcfcf",
          700: "#525252",
          900: "#372e30",
          "100_01": "#d9d9d9",
        },
        deep_orange: { "50_fc": "#f2dfdffc" },
        black: { 900: "#000000", "900_3f": "#0000003f", "900_ea": "#000000ea" },
        red: { A400: "#ed174d" },
      },
      boxShadow: { bs: "0px 4px  4px 0px #0000003f" },
      fontFamily: {
        inter: "Inter",
        abeezee: "ABeeZee",
        paytoneone: "Paytone One",
        yellowtail: "Yellowtail",
        yesevaone: "Yeseva One",
        ibmplexsans: "IBM Plex Sans",
        lg: "lg",
        yogi: "yogi",
      },
      textShadow: { ts: "0px 8px  8px #000000ea" },
      backgroundImage: {
        gradient: "linear-gradient(180deg ,#534747cc,#534747cc,#534747cc)",
      },
    },
  },
  plugins: [require("@tailwindcss/forms"), require("tailwindcss-textshadow")],
};
