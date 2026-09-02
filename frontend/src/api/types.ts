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
  /**
   * Affine coefficients `[a, b, c, d, e, f]` of the window actually read, in
   * `crs`. Optional for backward compatibility. This - never `bbox` - is the
   * georeferencing source: `bbox` echoes the request, while the read window is
   * floor/ceil clamped onto the source grid and covers more.
   */
  transform: number[] | null;
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

/**
 * One actual satellite acquisition selected for one requested window.
 *
 * A `TimeRange` is what was *requested*; an `Observation` is what was
 * *acquired*. `scene.datetime` is the real acquisition time and will differ
 * from `requested_window`. Observations are NOT co-registered and may differ in
 * CRS, resolution and footprint.
 */
export interface Observation {
  modality: Modality;
  window_label: string;
  requested_window: TimeRange;
  scene: SatelliteScene;
  imagery: ImageryResponse | null;
}

/** The observations from one execution. `requested_bbox` is the AOI asked for. */
export interface ObservationSet {
  requested_bbox: BoundingBox;
  observations: Observation[];
}

export interface QueryExecutionResult {
  plan: ResolvedQueryPlan;
  executed_modalities: Modality[];
  skipped_modalities: SkippedModality[];
  windows: ExecutedWindow[];
  catalog: string;
  /**
   * Derived server-side from `windows`; the backend always sends it and
   * recomputes it on input. Optional here so existing fixtures and consumers
   * remain valid - no UI reads it yet.
   */
  observations?: ObservationSet;
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

/**
 * Compatibility between two observations, established from METADATA ONLY.
 * `"unknown"` never means `"different"`, and no combination of matches implies
 * co-registration - `co_registration_status` is derived from modality alone.
 */
export interface CompatibilityReport {
  same_modality: boolean;
  temporal_separation_days: number | null;
  bbox_overlap: "none" | "partial" | "full" | "unknown";
  crs_match: "same" | "different" | "unknown";
  resolution_match: "same" | "different" | "unknown";
  processing_level_match: "same" | "different" | "unknown";
  limitations: string[];
  co_registration_status: "not_evaluated" | "not_supported_cross_modal";
}

/** Index statistics for ONE observation, computed on its own pixels. */
export interface ObservationIndexResult {
  window_label: string;
  scene_id: string;
  acquired_at: string | null;
  cloud_cover: number | null;
  measurements: Measurement[];
  /**
   * Affine coefficients `[a, b, c, d, e, f]` of the band window this
   * observation was indexed over, carried verbatim from the raster read. Per
   * observation: two observations are NOT co-registered and may sit on
   * different grids, so this is never shared between them.
   */
  transform: number[] | null;
}

/**
 * Two independently indexed observations, side by side. `differences` holds at
 * most one entry - the difference between the two aggregate means. It is a
 * difference of statistics over two separate sets of pixels, not a spatial
 * comparison, and it is suppressed entirely when that framing would mislead.
 */
export interface TemporalIndexComparison {
  first: ObservationIndexResult;
  second: ObservationIndexResult;
  compatibility: CompatibilityReport;
  differences: Measurement[];
  warnings: string[];
}

/** The task is derived from `execution.plan.intent.task`; there is no task field. */
export interface AnalysisRequest {
  execution: QueryExecutionResult;
  /**
   * Opt in to single-scene Sentinel-2 NDWI statistics. Omitted (or `false`)
   * leaves the request identical to the pre-NDWI behaviour; the backend
   * defaults it to `false`.
   */
  include_ndwi?: boolean;
  /**
   * Opt in to Temporal NDWI Statistics for one deterministic Sentinel-2
   * observation pair. Independent of `include_ndwi`; the backend defaults it to
   * `false`.
   */
  include_temporal_ndwi?: boolean;
}

export interface AnalysisResult {
  status: AnalysisStatus;
  task: QueryTask;
  answer: string;
  windows_considered: AnalysisWindowRef[];
  warnings: string[];
  measurements: Measurement[];
  /**
   * Temporal NDWI Statistics for one observation pair. `null` when the feature
   * was not requested, or was requested but could not produce a valid
   * comparison - the reason is then in `warnings`.
   */
  temporal_comparison?: TemporalIndexComparison | null;
}

// --------------------------------------------------------------------------- //
// Agentic orchestration (POST /api/v1/query/agent)
//
// Mirrors the backend contracts exactly. No business logic is reimplemented
// here: the frontend renders what the server established and decides nothing.
// --------------------------------------------------------------------------- //

/** The closed set of tools a planner may select. */
export type AgentToolName =
  | "execute_query"
  | "ndwi_statistics"
  | "temporal_ndwi_statistics";

/** Parameters for the discovery tool. Note there is no server `limit` here. */
export interface ExecuteQueryParams {
  tool: "execute_query";
  intent: SatQueryIntent;
  include_imagery: boolean;
  max_cloud_cover: number | null;
}

export interface NdwiParams {
  tool: "ndwi_statistics";
}

export interface TemporalNdwiParams {
  tool: "temporal_ndwi_statistics";
}

export type AgentToolCall =
  | ExecuteQueryParams
  | NdwiParams
  | TemporalNdwiParams;

/** The validated plan. 1-3 steps, `execute_query` first. */
export interface AgentPlan {
  steps: AgentToolCall[];
}

/** What actually happened to one step - observable outcome only. */
export interface AgentToolStep {
  status: "ok" | "rejected" | "failed" | "skipped";
  parameters: AgentToolCall;
  rejection_reason: string | null;
  error_message: string | null;
}

/** Outcome of each mechanical check applied to the generated answer. */
export interface AnswerValidation {
  numeric_grounding: "pass" | "fail" | "not_run";
  forbidden_terms: "pass" | "fail" | "not_run";
  evidence_refs: "pass" | "fail" | "not_run";
}

/**
 * Observable decisions and results only. There is deliberately no field for
 * reasoning, and none may be added: the UI shows what happened, never why the
 * model thought so.
 */
export interface AgentTrace {
  plan: AgentPlan | null;
  steps: AgentToolStep[];
  evidence_refs: string[];
  answer_validation: AnswerValidation | null;
}

/** One citable fact: a measurement, or qualifying text. */
export interface EvidenceItem {
  id: string;
  source: "execution" | "ndwi" | "temporal_ndwi" | "compatibility" | "model";
  measurement: Measurement | null;
  text: string | null;
  produced_by: string | null;
}

export interface AgentEvidence {
  items: EvidenceItem[];
  execution: QueryExecutionResult | null;
  analysis: AnalysisResult | null;
}

/**
 * `ok` carries an answer. The other three do not: the answer is withheld or
 * was never produced, and the deterministic evidence is returned instead.
 */
export type AgentStatus =
  | "ok"
  | "planner_unavailable"
  | "synthesis_unavailable"
  | "answer_withheld";

export interface AgentQuestionRequest {
  question: string;
}

export interface AgentResult {
  status: AgentStatus;
  answer: string | null;
  trace: AgentTrace;
  evidence: AgentEvidence;
}
