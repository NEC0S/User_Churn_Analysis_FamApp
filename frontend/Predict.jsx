@tailwind base;
@tailwind components;
@tailwind utilities;

html {
  scroll-padding-top: env(safe-area-inset-top, 0px);
}

body {
  padding-top: env(safe-area-inset-top, 0px);
  padding-bottom: env(safe-area-inset-bottom, 0px);
  -webkit-font-smoothing: antialiased;
}
