/** Shared API response types. Mirrors the backend Pydantic models. */

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  environment: string;
}

/** One thing the deployment needs in order to do its work. */
export interface Capability {
  name: string;
  ready: boolean;
  /** Why, in the server's own words - populated whether ready or not. */
  detail: string;
  /**
   * Whether readiness depends on it. The AI provider is optional: supported
   * questions are answered by the standard workflow without one. Absent means
   * required (older servers sent no such field).
   */
  required?: boolean;
}

/**
 * Whether this deployment can actually perform its advertised workflow.
 *
 * Distinct from `HealthResponse`, which answers only "is the process alive?".
 * A process with no credential for its selected provider is perfectly alive and
 * cannot answer a single query, and `/health` reported that as `ok`.
 *
 * The endpoint answers **503 when not ready**, so orchestration reads the
 * status code - but the body is still the explanation, which is why it must be
 * read on both 200 and 503.
 */
export interface ReadinessResponse {
  ready: boolean;
  service: string;
  version: string;
  environment: string;
  capabilities: Capability[];
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
  sar_polarizations?: string[] | null;
  sar_instrument_mode?: string | null;
  orbit_state?: string | null;
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
  /**
   * How many scenes the query matched IN THE CATALOG, when it reported that.
   *
   * `scene_count` is how many came back - one bounded page. Selection picks
   * from that page, so a chosen scene is the best of what was returned, not
   * the best that exists. `null`/absent means the catalog did not say, which is
   * not the same as "all of them".
   */
  scenes_matched?: number | null;
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
  /**
   * The image's footprint as exactly four `[lon, lat]` pairs in EPSG:4326,
   * ordered `[NW, NE, SE, SW]` - the order a MapLibre image source expects.
   * Derived from `transform` at the returned image's size, never from `bbox`.
   * Four corners rather than a rectangle because a reprojected UTM window is a
   * quadrilateral in WGS84. `null` when no honest footprint could be derived.
   */
  corners_wgs84: number[][] | null;
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

/**
 * How a threshold compares an index value. Exact: `gt` is `>`, never `>=`.
 */
export type NdwiComparison = "gt" | "gte" | "lt" | "lte";

/** A numeric NDWI threshold the request stated explicitly. */
export interface NdwiThreshold {
  operator: NdwiComparison;
  value: number;
}

export interface SatQueryIntent {
  location_query: string;
  temporal_mode: TemporalMode;
  time_windows: TemporalComparison | TimeRange[];
  modalities: Modality[];
  task: QueryTask;
  /** Present only when the request named an explicit NDWI threshold. */
  ndwi_threshold?: NdwiThreshold | null;
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
  /**
   * Which catalog answered THIS window. A mixed request reaches two services -
   * Sentinel-2 from Earth Search, Sentinel-1 RTC from the Planetary Computer -
   * so provenance travels per observation rather than once per result.
   * Optional: an older server does not send it.
   */
  catalog?: string | null;
  /**
   * Why this window produced nothing, when DISCOVERY failed. Distinct from
   * `imagery_error`, which means the scene was found and its picture could not
   * be read.
   */
  error?: string | null;
  /**
   * How many scenes matched this window in the catalog, when it said. Together
   * with `scene_count` this is what makes the scope of the selection readable.
   */
  scenes_matched?: number | null;
}

/**
 * Which Sentinel-1 polarization a SAR window renders when imagery is
 * requested. Both are real measurement bands of the same RTC acquisition; VV
 * and VH are NOT a classification and neither is "the" SAR image.
 */
export type SarPolarization = "vv" | "vh";

export interface QueryExecutionRequest {
  intent: SatQueryIntent;
  include_imagery?: boolean;
  max_cloud_cover?: number;
  limit?: number;
  /**
   * Which polarization a Sentinel-1 window renders. Ignored by optical
   * windows, which have their own asset. Omitted leaves the request
   * byte-identical to the previous contract; the backend defaults to `vv`.
   */
  sar_polarization?: SarPolarization;
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

/**
 * Whether every requested window actually executed.
 *
 * About EXECUTION, not scientific completeness: a window that ran and matched
 * no scene is `completed`, because discovery worked and the archive simply
 * holds nothing there. Derived server-side from the windows, so it cannot
 * disagree with them.
 */
export type ExecutionStatus = "completed" | "partial" | "failed";

export interface QueryExecutionResult {
  plan: ResolvedQueryPlan;
  executed_modalities: Modality[];
  skipped_modalities: SkippedModality[];
  windows: ExecutedWindow[];
  /**
   * The catalog that answered first. Kept for compatibility; `catalogs` names
   * every service a mixed run actually used, and each window carries its own.
   */
  catalog: string;
  /** Every catalog that answered, in the order first seen. */
  catalogs?: string[];
  /** Optional so existing fixtures and older servers remain valid. */
  status?: ExecutionStatus;
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
  /** This observation's own pixel quality (M3). Optional for older servers. */
  pixel_quality?: PixelQuality | null;
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
/**
 * Paired-pixel NDWI change between a baseline and a target observation.
 *
 * Present ONLY when the two grids were verified identical - same size, same
 * CRS, same affine - so every difference is between two measurements of the
 * same ground. `change = second_NDWI - first_NDWI`, where *first* is the
 * EARLIER acquisition and *second* the later one - never the requested
 * baseline/target roles, which may be inverted relative to time. It is an
 * INDEX change, not
 * water gained or lost. Statistics cover only pixels valid in BOTH
 * observations. Nothing is resampled or co-registered; an incompatible pair
 * yields `null` and a warning explaining why.
 */
export interface NdwiTemporalChange {
  first_scene_id: string;
  second_scene_id: string;
  first_acquired_at: string | null;
  second_acquired_at: string | null;
  window_label: string;
  /** Pixels valid in BOTH observations - the denominator for every statistic. */
  paired_valid_pixel_count: number;
  change_mean: number;
  change_min: number;
  change_max: number;
  crs: string;
  transform: number[];
  corners_wgs84: number[][];
  overlay: NdwiOverlay | null;
}

export interface TemporalIndexComparison {
  first: ObservationIndexResult;
  second: ObservationIndexResult;
  compatibility: CompatibilityReport;
  differences: Measurement[];
  /** Paired-pixel change, or `null` when the grids were not comparable. */
  change?: NdwiTemporalChange | null;
  warnings: string[];
}

/** The task is derived from `execution.plan.intent.task`; there is no task field. */
/**
 * The spectral indices the backend can compute. Closed set - the server
 * refuses anything else rather than computing a different index.
 */
export type SpectralIndexKey = "ndvi" | "ndwi" | "ndbi";

export interface AnalysisRequest {
  execution: QueryExecutionResult;
  /**
   * Opt in to single-scene Sentinel-2 NDWI statistics. Omitted (or `false`)
   * leaves the request identical to the pre-NDWI behaviour; the backend
   * defaults it to `false`.
   */
  include_ndwi?: boolean;
  /**
   * Additional spectral indices to compute over the same optical window,
   * by key. All are normalised differences over Sentinel-2 bands:
   * NDVI = (nir - red), NDWI = (green - nir), NDBI = (swir16 - nir), each
   * divided by the sum of its pair.
   *
   * Additive and independent of `include_ndwi`, which remains the path that
   * also produces the georeferenced overlay.
   */
  indices?: SpectralIndexKey[];
  /**
   * Additionally render the NDWI grid as a georeferenced PNG overlay. Requires
   * `include_ndwi`: the picture is a view of those same pixels.
   */
  include_ndwi_overlay?: boolean;
  /**
   * Opt in to Temporal NDWI Statistics for one deterministic Sentinel-2
   * observation pair. Independent of `include_ndwi`; the backend defaults it to
   * `false`.
   */
  include_temporal_ndwi?: boolean;
  /**
   * Opt in to quantitative Sentinel-1 backscatter statistics over the SAR
   * window: VV and VH gamma-naught in decibels, plus their mean difference.
   *
   * The provider (Microsoft Planetary Computer) supplies radiometrically
   * terrain-corrected gamma naught; SatQuery converts those provider values to
   * dB and reports statistics over them. It performs no radiometric
   * calibration, speckle filtering or polarimetric decomposition of its own.
   *
   * Omitted (or `false`) leaves the request identical to the previous
   * behaviour; the backend defaults it to `false`.
   */
  include_sar_backscatter?: boolean;
}

/**
 * The NDWI index rendered as a georeferenced picture.
 *
 * A visualisation of the index, not a classification: the colour ramp maps the
 * NDWI value and nothing else, so a bright pixel means a high index, not
 * detected water. Positioned exactly like RGB imagery - by `transform` and
 * `corners_wgs84` taken from the band window the index was computed on, never
 * from the requested bbox. Pixels with no valid measurement are transparent.
 */
export interface NdwiOverlay {
  scene_id: string;
  window_label: string;
  media_type: "image/png";
  image_base64: string;
  width: number;
  height: number;
  crs: string;
  /** Affine of this raster, in `crs` - `[a, b, c, d, e, f]`. */
  transform: number[];
  /** Four `[lon, lat]` pairs in EPSG:4326, ordered `[NW, NE, SE, SW]`. */
  corners_wgs84: number[][];
  /** The index range actually rendered, so a legend can be honest about it. */
  value_min: number;
  value_max: number;
  valid_pixel_count: number;
}

/**
 * A deterministic threshold count over the analysed NDWI pixels.
 *
 * `percentage` is `matching_pixel_count / valid_pixel_count * 100` - the
 * denominator is VALID pixels, never the raster's size, because a nodata or
 * non-finite pixel was never measured and cannot count either way. Computed
 * server-side from real pixels; no language model produces these numbers.
 */
export interface SpatialMeasurement {
  metric: "ndwi";
  operator: NdwiComparison;
  threshold: number;
  matching_pixel_count: number;
  valid_pixel_count: number;
  percentage: number;
  scene_id: string;
  acquired_at: string | null;
  window_label: string;
  crs: string | null;
  corners_wgs84: number[][] | null;
}

export interface SarPolarizationStatistics {
  polarization: SarPolarization;
  measurements: Measurement[];
  valid_pixel_count: number;
  nonpositive_pixel_count: number;
  window_pixel_count: number;
  crs: string | null;
  resolution: number | null;
  transform: number[] | null;
}

export interface SarBackscatterResult {
  scene_id: string;
  window_label: string;
  acquired_at: string | null;
  collection: string | null;
  polarizations: SarPolarizationStatistics[];
  difference: {
    vv_mean_db: number;
    vh_mean_db: number;
    vv_minus_vh_mean_db: number;
    paired_valid_pixel_count: number;
    crs: string | null;
    transform: number[] | null;
  } | null;
  measurements: Measurement[];
  warnings: string[];
}

/**
 * What became of ONE requested analysis.
 *
 * `status` (below) answers a different question - whether the TASK has an
 * engine - so on its own it reported "ok" for a request whose analysis
 * produced nothing at all.
 */
export interface AnalysisOutcome {
  /** "ndvi" | "ndwi" | "ndbi" | "temporal_ndwi" | "sar_backscatter". */
  name: string;
  status: "completed" | "unavailable";
  /** Present only when unavailable; the server's own words. */
  reason?: string | null;
}

/**
 * Whether the analyses the request asked for were produced. `not_requested` is
 * distinct from `none`: asking for nothing and getting nothing is a complete
 * answer to the question that was asked.
 */
export type AnalysisCompleteness =
  | "complete"
  | "partial"
  | "none"
  | "not_requested";

export interface AnalysisResult {
  sar_backscatter?: SarBackscatterResult | null;
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
  /**
   * The NDWI grid as a georeferenced picture, when it was requested and could
   * be positioned honestly. `null` otherwise - a missing overlay never
   * suppresses the measurements, which remain the product.
   */
  ndwi_overlay?: NdwiOverlay | null;
  /**
   * A threshold count over the analysed NDWI pixels, when the intent stated a
   * threshold and there were valid pixels to count. `null` otherwise - never a
   * fabricated 0%.
   */
  spatial_measurement?: SpatialMeasurement | null;
  /**
   * One entry per analysis the request asked for. Optional so existing
   * fixtures and older servers remain valid.
   */
  analysis_outcomes?: AnalysisOutcome[];
  /** Derived server-side from `analysis_outcomes`. */
  completeness?: AnalysisCompleteness;
  /**
   * The validation stages the scientific core ran (M3-M5), one entry per
   * index grid / scene. Displayed verbatim; optional for older servers.
   */
  pixel_quality?: PixelQuality[];
  radiometry?: RadiometricState[];
  grids?: GridState[];
}

/**
 * How many of an index grid's pixels were usable, per the Sentinel-2 Scene
 * Classification Layer - every pixel counted once. The subset the workspace
 * shows; the backend carries more (per-class counts, notes).
 */
export interface PixelQuality {
  index: string;
  scene_id: string;
  window_label: string;
  mask_source: string;
  total_pixels: number;
  valid_pixels: number;
  masked_pixels: number;
  nodata_pixels: number;
  cloud_pixels: number;
  cloud_shadow_pixels: number;
  snow_pixels: number;
  saturated_or_defective_pixels: number;
  /** `null` when the grid had no pixels - never a fabricated 0 or 1. */
  valid_fraction: number | null;
  contamination_fraction: number | null;
  quality_notes: string[];
}

/** Whether the values were on the representation the formula assumes (M4). */
export interface RadiometricState {
  status:
    | "verified"
    | "verified_with_unknown_metadata"
    | "incompatible"
    | "undetermined";
  modality: string;
  scene_id: string;
  collection: string;
  representation: string | null;
  processing_baseline: string | null;
}

/** Whether the rasters an analysis combined share a verified grid (M5). */
export interface GridState {
  status: "valid" | "refused";
  analysis: string;
  scene_id: string | null;
  stage: "pre_read" | "post_read";
  crs: string | null;
  width: number | null;
  height: number | null;
  resolution_x: number | null;
  resolution_y: number | null;
  refusal: string | null;
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
  | "spectral_indices"
  | "ndwi_statistics"
  | "sar_backscatter_statistics"
  | "temporal_ndwi_statistics"
  | "rs_model_analysis";

/** Parameters for the discovery tool. Note there is no server `limit` here. */
export interface ExecuteQueryParams {
  tool: "execute_query";
  intent: SatQueryIntent;
  include_imagery: boolean;
  max_cloud_cover: number | null;
}

/**
 * Which spectral indices to compute over the discovered scene.
 *
 * The one analysis tool that takes a parameter: choosing WHICH index answers a
 * question is a planning decision, while how each is computed stays a
 * scientific constant owned by the backend engine.
 */
export interface SpectralIndicesParams {
  tool: "spectral_indices";
  indices: SpectralIndexKey[];
}

export interface NdwiParams {
  tool: "ndwi_statistics";
}

export interface SarBackscatterParams {
  tool: "sar_backscatter_statistics";
}

export interface TemporalNdwiParams {
  tool: "temporal_ndwi_statistics";
}

/**
 * One visual question about the image the server already retrieved.
 *
 * Carries the question and nothing else - deliberately no scene, asset, URL or
 * bytes. The server decides which image the model looks at.
 */
export interface RsModelParams {
  tool: "rs_model_analysis";
  question: string;
}

export type AgentToolCall =
  | ExecuteQueryParams
  | SpectralIndicesParams
  | NdwiParams
  | SarBackscatterParams
  | TemporalNdwiParams
  | RsModelParams;

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
  /**
   * Provenance, NOT a fourth check. `attributed` records that the answer rests
   * partly on a model observation, which nothing mechanical can validate. It
   * never means the observation was verified.
   */
  visual_claims: "attributed" | "not_run";
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
/**
 * What a vision-language model said about one retrieved image.
 *
 * An ATTRIBUTED observation, never a verified fact - nothing mechanical can
 * check "water is visible". Deliberately not a `Measurement`: even when the
 * model states a number, that number lives inside `statement` and can never
 * authorise a numeric claim.
 */
export interface VisualObservation {
  statement: string;
  provider: string;
  model: string;
  scene_id: string;
}

export interface EvidenceItem {
  id: string;
  source:
    | "execution"
    | "ndvi"
    | "ndwi"
    | "ndbi"
    | "temporal_ndwi"
    | "sar_backscatter"
    | "compatibility"
    | "model";
  measurement: Measurement | null;
  text: string | null;
  /** Present only on model-sourced evidence. Never rendered as a measurement. */
  visual: VisualObservation | null;
  produced_by: string | null;
}

export interface AgentEvidence {
  items: EvidenceItem[];
  execution: QueryExecutionResult | null;
  analysis: AnalysisResult | null;
}

/**
 * `ok` carries an answer. The next three do not: the answer is withheld or
 * was never produced, and the deterministic evidence is returned instead.
 * `needs_clarification` means nothing was measured: the question did not say
 * something the workflow needs, and `clarification` says what.
 * `location_unavailable` means the question was fine but the location service
 * (the geocoder) could not be used, so nothing was searched; `failure` says so
 * and, when known, how long to wait. It is neither a question back to the user
 * nor a verdict on the evidence.
 */
export type AgentStatus =
  | "ok"
  | "planner_unavailable"
  | "synthesis_unavailable"
  | "answer_withheld"
  | "needs_clarification"
  | "location_unavailable";

/** Why a question could not be executed as asked - one missing fact each. */
export type ClarificationReason =
  | "analysis_missing"
  | "analysis_unsupported"
  | "analysis_ambiguous"
  | "location_missing"
  | "location_not_found"
  | "area_too_large"
  | "date_missing"
  | "date_ambiguous"
  | "date_invalid"
  | "comparison_incomplete"
  | "conflicting_request"
  | "requires_ai_model";

/**
 * A question put back to the user instead of a guess. System-authored; the
 * `understood_*` fields repeat what the question DID establish.
 */
export interface AgentClarification {
  reason: ClarificationReason;
  message: string;
  options: string[];
  /**
   * One complete question per option, built only from what the question
   * already established; empty when the options are not questions to ask.
   * Optional so a response from an older server still type-checks.
   */
  option_questions?: string[];
  understood_analyses: string[];
  understood_location: string | null;
  understood_periods: TimeRange[];
}

export interface AgentQuestionRequest {
  question: string;
  /**
   * Which AI backend interprets this run. Omitted - the default - runs the
   * STANDARD workflow: deterministic interpretation, no model, no credential.
   * Naming one opts this run into AI interpretation and changes nothing else -
   * the deterministic pipeline, the grounding rules and the evidence shape are
   * identical either way.
   */
  provider?: string | null;
  /** Which model that provider should use for this run. */
  model?: string | null;
}

/**
 * The vision-language backends the server can be asked to use.
 *
 * Selection only - no key ever reaches the browser. The choice changes the
 * inference backend and nothing else: the deterministic pipeline, grounding
 * and evidence contract are identical either way.
 */
export type AiProvider = "gemini" | "nvidia" | "anthropic" | "local";

/** Which step a model would fill. */
export type ModelRole = "visual" | "text";

/**
 * One catalogued model and this deployment's view of it.
 *
 * `configured`, `compatible` and `available` are separate questions and are
 * reported separately: a model can exist, be incompatible with the step, lack a
 * credential, or have been RETIRED by its provider. None of these means
 * "reachable" - only a real request answers that, so the server never claims
 * availability here.
 *
 * `available` is a published retirement fact, not a health check: a retired
 * model stays `compatible` (it is still image-capable) but must never be
 * selectable, because the request would fail with 410.
 */
export interface ModelOption {
  provider: string;
  model_id: string;
  display_name: string;
  modality: string;
  supports_image: boolean;
  supports_text: boolean;
  supports_video: boolean;
  supports_tools: boolean;
  supports_structured_output: boolean;
  endpoint_type: string;
  configured: boolean;
  compatible: boolean;
  /** False when the provider has retired the model. Optional for forward
   *  compatibility with a server that predates the field; treated as available
   *  when absent, which matches how such a server behaved. */
  available?: boolean;
  retired_reason?: string | null;
  status: string;
}

export interface ModelCatalogResponse {
  role: string;
  default_provider: string;
  default_model: string;
  models: ModelOption[];
}

export interface AgentFailure {
  stage: "planning" | "synthesis" | "location";
  code: string;
  message: string;
  retry_after_seconds: number | null;
  /** The external dependency that failed, when the failure is one. */
  dependency?: "geocoder" | null;
}

export interface AgentResult {
  failure?: AgentFailure | null;
  /** Present exactly when `status` is `needs_clarification`. */
  clarification?: AgentClarification | null;
  status: AgentStatus;
  answer: string | null;
  trace: AgentTrace;
  evidence: AgentEvidence;
}
