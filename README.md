# SatQuery AI

SatQuery AI lets you ask a question about satellite images in normal English and get an answer back. You type something like "Show vegetation around Cubbon Park, Bengaluru in December 2024" and it finds the place, picks a real satellite image, measures what you asked for and tells you the result. Every number in the answer comes from the actual image pixels, and you can see exactly where it came from.

We built this for Smart India Hackathon 2026.


## What you can ask

Here are some questions it can answer:

- What is the NDWI of Marina Beach, Chennai in January 2025?
- Show vegetation around Cubbon Park, Bengaluru in December 2024.
- Show built-up area around Ameerpet, Hyderabad in January 2025.
- Analyze SAR backscatter around Marina Beach, Chennai in January 2025.
- Compare water at Marina Beach, Chennai between January 2024 and January 2025.
- Has the vegetation around Cubbon Park, Bengaluru changed between December 2023 and December 2024?
- What do you see in the satellite image of Marina Beach, Chennai in January 2025?

You need to tell it a place and a time. If something is missing, it asks you instead of guessing. If you ask for something it cannot do, like counting ships or predicting crop yield, it says so plainly.


## What it can measure

- Vegetation (NDVI), water (NDWI) and built-up area (NDBI) from Sentinel-2 optical images.
- Radar backscatter (VV and VH) from Sentinel-1 radar images.
- Change between two dates for water, vegetation and built-up area.
- The normal colour satellite image of the place, shown on the map.

The area can be as big as a whole city, about 45 km on each side, like all of Chennai. Bigger areas like a district, state or country are refused as too large. Large areas are read at full 10 m resolution in pieces and then combined, so nothing is blurred or downsampled. A whole city takes a few minutes.


## How it works

1. It reads your question. Simple rules plus a small model we trained ourselves work out what you want, where and when. This runs on the server itself, with no internet call and no API key.
2. It looks up the place on OpenStreetMap (Nominatim) to get the area.
3. It searches the public satellite catalogues for images of that area and time, and picks one the same way every time (least cloudy first, and one whose values can be checked).
4. It checks the image before using it. Clouds, shadows and bad pixels are masked out using the image's own scene classification layer. The pixel values and the grid are checked too. If the image fails a check, it tells you why instead of giving a wrong number.
5. It measures what you asked for.
6. It writes the answer using fixed sentences filled with the measured values, and then checks that every number in the answer really matches a measurement.

On screen you get a plain answer first ("The selected area shows a positive vegetation signal"), then a short "how we know", and then all the technical details if you want them.


## Does it need an AI key?

No. By default nothing uses an outside AI service and no key is needed. Everything above works without one.

There is optional support for Gemini, NVIDIA, Anthropic and a local model through Ollama. These can plan the analysis, describe the picture and word the answer, but they never calculate any number. Every number still comes from the pixels, and an answer with a number the measurements don't back up is held back. In production, users cannot pick an AI provider at all, so no one can run up a bill.

If you want to try a local model on your own machine: install Ollama, run "ollama pull qwen3-vl:4b-instruct", and start the backend with AI_PROVIDER=local. Use the "instruct" version of the model, because the plain one spends a long time thinking before it answers. On an 8 GB Mac the 4b model works, the 2b model is too weak, and the 8b model is too big.


## Running it on your computer

You need Python 3.12 with uv, and Node.js.

Backend:

- go into the backend folder
- run: uv sync
- copy .env.example to .env
- run: uv run uvicorn app.main:app --reload --port 8000

Frontend:

- go into the frontend folder
- run: npm install
- run: npm run dev
- open http://localhost:5173 in your browser

The frontend has to run on port 5173 because the backend only accepts requests from there in development.

To check whether the backend is up, open http://localhost:8000/health. To check that everything it depends on is working, open http://localhost:8000/ready. The API documentation is at http://localhost:8000/docs.


## Running the tests

- secret scan, from the main folder: python3 scripts/check_secrets.py
- backend, from the backend folder: uv run pytest -q and then uv run ruff check .
- frontend, from the frontend folder: npm test, npm run typecheck, npm run lint and npm run build

Right now there are 3221 backend tests and 577 frontend tests, and they all pass. None of them call a real outside service.


## Deploying

The backend runs on Render and the frontend on Vercel. For your own server there is a production Docker setup in docker-compose.prod.yml. All the limits (requests per minute, how many analyses run at once, memory, time per request) are explained in DEPLOYMENT.md.


## The API

- GET /health: is the server alive
- GET /ready: can it actually do its work right now
- GET /api/v1/ai/models: which AI models are available
- POST /api/v1/query/agent: ask a question (this is what the website uses)
- POST /api/v1/query/parse: turn a question into a structured request
- POST /api/v1/query/build-plan: turn that request into an area and dates
- POST /api/v1/query/execute: find and pick the satellite images
- POST /api/v1/query/analyze: measure the indices on the picked images
- POST /api/v1/geospatial/resolve: place name to area
- POST /api/v1/satellite/search: search the satellite catalogue
- POST /api/v1/satellite/imagery: get a picture of an area


## What it does not do

We want to be clear about the limits.

- NDVI, NDWI and NDBI are indices, not classifiers. A high NDBI does not mean a building was detected, and a positive NDWI does not mean a water body or a flood was confirmed.
- Built-up area (NDBI) uses a 20 m band, so its detail is 20 m even though the result is shown on the 10 m grid.
- The two-date comparison compares averages. A pixel-by-pixel change is shown only when both images line up on exactly the same grid. It does not tell you what caused a change.
- There is no two-date radar comparison yet.
- Radar values come from a product that the provider (Microsoft Planetary Computer) has already terrain corrected. We do not do our own calibration, speckle filtering or terrain correction.
- The cloud mask is only as good as the scene classification that comes with the image.
- Absolute surface reflectance is not worked out. The indices are ratios, so they don't need it.
- Nobody has checked the results against ground measurements yet.
- It only understands English.
- It uses the free public OpenStreetMap geocoder and map tiles, which have usage limits. A bigger deployment would need its own.

SUPPORTED_DATA.md has the full list of what is and isn't supported, with reasons.


## Built with

- Backend: Python 3.12, FastAPI, Pydantic, rasterio and numpy
- Frontend: React, TypeScript, Vite and MapLibre
- Data: Sentinel-2 from Earth Search, Sentinel-1 from Microsoft Planetary Computer, places from OpenStreetMap


## Folders

- backend/app/api: the web routes
- backend/app/services/geospatial: place lookup
- backend/app/services/satellite: catalogue search and reading the images
- backend/app/services/analysis: the index, radar and comparison calculations
- backend/app/services/query: running a request from start to end
- backend/app/services/agent: reading the question, planning and checking the answer
- backend/tests: the backend tests
- frontend/src: the website
- PROJECT_RECORD.md: the full history of how each part was built and checked
