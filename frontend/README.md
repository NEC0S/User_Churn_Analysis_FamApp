# ChurnLens Frontend

A React + Vite + Tailwind dashboard that calls a live churn-prediction backend and visualizes the
full pipeline: raw input → feature engineering → model scoring → probability → SHAP explanation.

## Project structure

```
churnlens-frontend/
├── src/
│   ├── App.jsx                 # orchestrates state + the pipeline animation
│   ├── api.js                  # backend client, env-based URL resolution
│   ├── components/
│   │   ├── InputPanel.jsx
│   │   ├── PipelineView.jsx
│   │   ├── ResultGauge.jsx
│   │   └── ExplanationBars.jsx
│   └── data/presets.js         # demo customer profiles
├── Dockerfile                  # multi-stage: node build -> nginx serve
├── nginx.conf
├── docker-compose.yml          # runs this + ../backend together
└── .env.example
```

## Run locally (no Docker)

```bash
npm install
cp .env.example .env       # edit VITE_API_URL if your backend isn't on localhost:8000
npm run dev
```
Open http://localhost:5173

## Build for production (no Docker)

```bash
npm run build      # outputs to dist/
npm run preview    # serve the build locally to sanity-check it
```

## Run with Docker (frontend only)

```bash
docker build --build-arg VITE_API_URL=http://localhost:8000 -t churnlens-frontend .
docker run -p 8080:80 churnlens-frontend
```
Open http://localhost:8080

**Important**: `VITE_API_URL` is baked in at **build** time, not container start time (this is how
Vite env vars work — they're compiled into the JS bundle, not read at runtime). If you deploy the
backend to a new URL later, you must rebuild the image with the new `--build-arg`, not just restart
the container.

## Run everything together with Docker Compose

From this folder, assuming `../backend` (the FastAPI service) is a sibling directory:
```bash
docker compose up --build
```
This builds and starts both containers, waits for the backend's health check to pass, then starts
the frontend. Open http://localhost:8080.

## Deploying

**Frontend** — this is a static site once built (`dist/` or the nginx image), so any of these work:
- Push the built image to a container registry and deploy on **Render** (Web Service, Docker
  environment), **Railway**, or **Fly.io**
- Or skip Docker entirely for the frontend and deploy `dist/` directly to **Netlify**, **Vercel**,
  or **GitHub Pages** — run `npm run build`, then drag the `dist/` folder to Netlify Drop, or
  `vercel --prod` from this directory. Set `VITE_API_URL` as an environment variable in that
  platform's dashboard before building, since it's baked in at build time.

**Backend** — see `../backend/Dockerfile`. Deploy the same way (Render/Railway/Fly.io Docker
service), or with `render.yaml` for a non-Docker Python deploy (already in the backend folder).

## After deploying both

Update the "Backend API URL" field in the running app's sidebar to your deployed backend's URL
(this overrides the build-time default via `localStorage`, no rebuild needed for this one field) —
or bake the correct URL in at build time via `--build-arg VITE_API_URL=...` so it's correct by
default for every visitor.
