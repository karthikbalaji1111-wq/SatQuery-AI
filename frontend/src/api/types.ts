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

export interface ImageryRequest {
  scene_id: string;
  bbox: BoundingBox;
  asset?: string;
  collection?: string;
  max_dimension?: number;
}

export interface ImageryWindowInfo {
  col_off: number;
  row_off: number;
  width: number;
  height: number;
}

export interface ImageryResponse {
  scene_id: string;
  bbox: BoundingBox;
  asset: string;
  asset_href: string;
  width: number;
  height: number;
  format: "png";
  media_type: "image/png";
  bands: string[];
  crs: string | null;
  resolution: number | null;
  normalization: string;
  window: ImageryWindowInfo;
  source_shape: number[];
  image_base64: string;
}

export type TemporalMode = "single" | "compare" | "timeseries";
export type Modality = "sentinel-2-optical" | "sentinel-1-sar";
export type QueryTask =
  | "visualize"
  | "change_detection"
  | "object_identification";

export interface TimeRange {
  start_date: string;
  end_date: string;
}

export interface TemporalComparison {
  baseline: TimeRange;
  target: TimeRange;
}

export interface SatQueryIntent {
  location_query: string;
  temporal_mode: TemporalMode;
  time_windows: TemporalComparison | TimeRange[];
  modalities: Modality[];
  task: QueryTask;
}

export interface ResolvedQueryPlan {
  intent: SatQueryIntent;
  bbox: BoundingBox;
}

export interface SkippedModality {
  modality: Modality;
  reason: string;
}

export interface ExecutedWindow {
  modality: Modality;
  label: string;
  time_range: TimeRange;
  scene_count: number;
  scenes: SatelliteScene[];
  selected_scene_id: string | null;
  imagery: ImageryResponse | null;
  imagery_error: string | null;
}

export interface QueryExecutionRequest {
  intent: SatQueryIntent;
  include_imagery?: boolean;
  max_cloud_cover?: number;
  limit?: number;
}

export interface QueryExecutionResult {
  plan: ResolvedQueryPlan;
  executed_modalities: Modality[];
  skipped_modalities: SkippedModality[];
  windows: ExecutedWindow[];
  catalog: string;
}

/**
 * `not_implemented` is returned in a normal 200 body for a task that has no
 * analysis engine yet - the analysis ran, it just performed no such analysis.
 */
export type AnalysisStatus = "ok" | "not_implemented";

export interface Measurement {
  name: string;
  value: number;
  unit: string;
}

/** Slim traceability reference - never echoes scenes or imagery. */
export interface AnalysisWindowRef {
  modality: Modality;
  label: string;
  time_range: TimeRange;
  selected_scene_id: string | null;
}

/** The task is derived from `execution.plan.intent.task`; there is no task field. */
export interface AnalysisRequest {
  execution: QueryExecutionResult;
}

export interface AnalysisResult {
  status: AnalysisStatus;
  task: QueryTask;
  answer: string;
  windows_considered: AnalysisWindowRef[];
  warnings: string[];
  measurements: Measurement[];
}
