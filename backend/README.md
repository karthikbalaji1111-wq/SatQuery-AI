# SatQuery backend

This is the server side of SatQuery. It's written in Python 3.12 with FastAPI, and we use uv to manage packages.


## Setting it up

- run: uv sync
- copy .env.example to .env

You don't need any API key. The satellite catalogues and the place lookup are all public.


## Running it

- run: uv run uvicorn app.main:app --reload --port 8000

Then:

- http://localhost:8000/health tells you the server is alive
- http://localhost:8000/ready tells you whether everything it depends on is working
- http://localhost:8000/docs has the API documentation

For running it in production, see DEPLOYMENT.md in the main folder.


## Running the checks

- run: uv run pytest -q
- run: uv run ruff check .

The tests never call a real outside service. They use small fake versions of the geocoder, the catalogues and the AI providers instead. They also ignore your own .env file, so they give the same result on any machine.


## Where things are

- app/main.py: creates the app and wires everything together
- app/api/routes: the web routes (health, geospatial, satellite, query, ai)
- app/core/config.py: all the settings, read from environment variables
- app/core/errors.py: the error types and how errors are returned
- app/core/limits.py: rate limits, how many analyses run at once, time limits and request size
- app/core/telemetry.py and app/core/logging.py: request ids, logs and metrics
- app/services/geospatial: turns a place name into an area, using OpenStreetMap
- app/services/satellite: searches the catalogues and reads the image files
- app/services/analysis: the NDVI, NDWI, NDBI, radar and two-date comparison calculations, plus the cloud, value and grid checks
- app/services/query: runs a request from start to end
- app/services/agent: reads the question, plans what to run and checks the answer
- app/services/agent/providers: the optional AI providers (Gemini, NVIDIA, Anthropic and a local Ollama model)


## A few things worth knowing

- Results from a client are checked before they're analysed. Someone can't send a made-up scene and get it measured.
- Each caller gets 120 requests a minute on the heavy routes. Only a few analyses run at the same time, and each request has a time limit. These limits are per server process.
- The place lookup is shared by everyone, so it's careful: one request at a time, results cached for a day, and each caller can look up at most 30 new places every 10 minutes.
- Image links come from the outside catalogue, so the server only opens public addresses and never anything on a private network.
- Large areas are read in pieces at full resolution and the results are combined exactly. Memory stays the same as reading one piece.


## AI providers (optional)

By default no AI is used and no key is needed. The question is read by simple rules and a small model we trained ourselves (in app/services/agent).

If you want to try an AI provider in development, set AI_PROVIDER to gemini, nvidia, anthropic or local, along with that provider's key or address in .env. The AI can choose what to run and word the answer, but it never calculates a number. If a provider isn't working, the request fails and says so. It never quietly switches to a different provider. In production, users can't choose a provider at all.
