"""Domain service boundaries for SatQuery.

Each subpackage owns one capability area and currently exposes only an
interface plus a not-yet-implemented stub:

- ``query``      - natural-language query parsing and orchestration
- ``satellite``  - Sentinel-1 SAR and Sentinel-2 optical retrieval
- ``multimodal`` - fusion / analysis across SAR + optical + text
- ``temporal``   - multitemporal change detection
- ``geospatial`` - geocoding, AOI handling, spatial grounding
- ``ai``         - model inference and reasoning
- ``map``        - map layer / tile preparation for the frontend

No capability logic is implemented in this foundation build.
"""
