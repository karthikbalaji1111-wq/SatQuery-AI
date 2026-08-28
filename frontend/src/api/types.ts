/** Shared API response types. Mirrors the backend Pydantic models. */

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  environment: string;
}

export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
  };
}

export interface Coordinate {
  lat: number;
  lon: number;
}

export interface BoundingBox {
  west: number;
  south: number;
  east: number;
  north: number;
}

export interface GeoResolveRequest {
  place?: string;
  bbox?: BoundingBox;
}

export interface GeoResolveResponse {
  query_type: "place" | "bbox";
  display_name: string | null;
  center: Coordinate;
  bbox: BoundingBox;
  source: "nominatim" | "input";
}

export interface SceneAsset {
  key: string;
  href: string;
  type: string | null;
  title: string | null;
  roles: string[] | null;
}

export interface SatelliteScene {
  id: string;
  datetime: string | null;
  bbox: BoundingBox | null;
  geometry: Record<string, unknown> | null;
  cloud_cover: number | null;
  collection: string | null;
  platform: string | null;
  processing_level: string | null;
  thumbnail_url: string | null;
  assets: SceneAsset[];
}

export interface SceneSearchQueryEcho {
  collections: string[];
  bbox: number[];
  datetime: string;
  max_cloud_cover: number | null;
  limit: number;
  filter: Record<string, unknown> | null;
}

export interface SceneSearchRequest {
  bbox: BoundingBox;
  start_date: string;
  end_date: string;
  max_cloud_cover?: number;
  limit?: number;
}

export interface SceneSearchResponse {
  query: SceneSearchQueryEcho;
  scene_count: number;
  scenes: SatelliteScene[];
  catalog: string;
}
