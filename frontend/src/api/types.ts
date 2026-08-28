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
