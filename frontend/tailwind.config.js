/** Same token names the components already use, so every page picks up the new theme. */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#F4F5F8',
        panel: '#FFFFFF',
        panel2: '#EEF0F5',
        line: '#DDE1E9',
        ink: '#0E1B2E',
        muted: '#56637A',
        faint: '#98A2B3',
        focus: '#1F4FD8',
        focusDim: '#E6ECFC',
        risk: '#C93B27',
        riskDim: '#FBE9E6',
        safe: '#0B8A64',
        safeDim: '#E0F4EC',
      },
      fontFamily: {
        sans: ['Manrope', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      boxShadow: { card: '0 1px 2px rgba(14,27,46,.05), 0 8px 24px -12px rgba(14,27,46,.12)' },
    },
  },
  plugins: [],
}