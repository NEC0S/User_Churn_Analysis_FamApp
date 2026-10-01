# --- Stage 1: build the React app ---
FROM node:20-alpine AS build
WORKDIR /app

# Install deps first (separate layer -- cached unless package*.json changes)
COPY package.json package-lock.json* ./
RUN npm install

# Build-time API URL: bake in your deployed backend's URL, e.g.
#   docker build --build-arg VITE_API_URL=https://churnlens-backend.onrender.com -t churnlens-frontend .
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL

COPY . .
RUN npm run build

# --- Stage 2: serve the static build with nginx ---
FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
