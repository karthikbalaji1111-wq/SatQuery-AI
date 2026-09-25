/**
 * Test fixtures shaped exactly like real production responses (captured from
 * the live API on 2026-09-24, trimmed of base64 imagery). Imported only by
 * tests. Values are the ones the backend actually returned for these places,
 * so a presentation test asserts the real figures a demo shows.
 */

import type { AgentResult } from "../../api/types";

const EARTH_SEARCH = "https://earth-search.aws.element84.com/v1";

type Json = Record<string, unknown>;

function item(id: string, source: string, name: string, value: number, unit: string): Json {
  return { id, source, measurement: { name, value, unit }, text: null, produced_by: "analysis", visual: null };
}

function note(id: string, source: string, text: string): Json {
  return { id, source, measurement: null, text, produced_by: "analysis", visual: null };
}

function scene(id: string, datetime: string, platform: string, cloud: number): Json {
  return {
    id, datetime, bbox: { west: 76.8, south: 12.5, east: 77.9, north: 13.6 }, geometry: null,
    cloud_cover: cloud, collection: "sentinel-2-l2a", platform, processing_level: "L2A",
    thumbnail_url: null, assets: [], sar_polarizations: null, sar_instrument_mode: null, orbit_state: null,
  };
}

function window(label: string, start: string, end: string, count: number, selected: Json | null, modality = "sentinel-2-optical"): Json {
  return {
    modality, label, time_range: { start_date: start, end_date: end }, scene_count: count,
    scenes: selected ? [selected] : [], selected_scene_id: selected ? selected.id : null,
    imagery: null, imagery_error: null, catalog: EARTH_SEARCH, scenes_matched: count, error: null,
  };
}

function plan(intent: Json, analysisTool: Json | null): Json {
  return {
    steps: [
      { tool: "execute_query", intent, include_imagery: false, sar_polarization: "vv", max_cloud_cover: null },
      ...(analysisTool ? [analysisTool] : []),
    ],
  };
}

const PASS = { numeric_grounding: "pass", forbidden_terms: "pass", evidence_refs: "pass", visual_claims: "not_run" };

function okSteps(plan: Json): Json[] {
  return (plan.steps as Json[]).map((parameters) => ({ status: "ok", parameters, rejection_reason: null, error_message: null }));
}

function pixelQuality(index: string, sceneId: string, label: string, valid: number, total: number): Json {
  return {
    index, scene_id: sceneId, window_label: label, mask_source: "sentinel-2-scl", total_pixels: total,
    valid_pixels: valid, masked_pixels: total - valid, nodata_pixels: 0, cloud_pixels: 0,
    cloud_shadow_pixels: 0, snow_pixels: 0, saturated_or_defective_pixels: 0,
    valid_fraction: valid / total, contamination_fraction: (total - valid) / total, quality_notes: [],
  };
}

function radiometry(sceneId: string, modality = "sentinel-2-optical", baseline: string | null = "05.11"): Json {
  return {
    status: "verified_with_unknown_metadata", modality, scene_id: sceneId, collection: "sentinel-2-l2a",
    representation: "read as-is", processing_baseline: baseline,
  };
}

function grid(analysis: string, sceneId: string, crs: string, width: number, height: number): Json {
  return {
    status: "valid", analysis, scene_id: sceneId, stage: "post_read", crs, width, height,
    resolution_x: 10, resolution_y: 10, refusal: null,
  };
}

function analysisResult(extra: Json): Json {
  return {
    status: "ok", task: "visualize", answer: "Analysed.", windows_considered: [], warnings: [],
    measurements: [], temporal_comparison: null, ndwi_overlay: null, spatial_measurement: null,
    sar_backscatter: null, analysis_outcomes: [], completeness: "complete",
    pixel_quality: [], radiometry: [], grids: [], ...extra,
  };
}

function singleIndex(opts: {
  key: "ndvi" | "ndwi" | "ndbi"; place: string; start: string; end: string; sceneId: string;
  acquired: string; platform: string; cloud: number; count: number; mean: number; min: number;
  max: number; valid: number; total: number; crs: string; width: number; height: number;
  answer: string; matched?: { name: string; class: string; type: string };
}): AgentResult {
  const intent = { location_query: opts.place, time_windows: [{ start_date: opts.start, end_date: opts.end }], modalities: ["sentinel-2-optical"], task: "visualize", temporal_mode: "single" };
  const p = plan(intent, { tool: "spectral_indices", indices: [opts.key] });
  const k = opts.key;
  return {
    status: "ok",
    answer: opts.answer,
    failure: null,
    clarification: null,
    trace: { plan: p, steps: okSteps(p), evidence_refs: [`${k}.${k}_mean`], answer_validation: PASS },
    evidence: {
      items: [
        item("execution.sentinel-2-optical.single.scene_count", "execution", "sentinel-2-optical_single_scene_count", opts.count, "count"),
        item(`${k}.${k}_valid_pixel_count`, k, `${k}_valid_pixel_count`, opts.valid, "pixels"),
        item(`${k}.${k}_mean`, k, `${k}_mean`, opts.mean, "index"),
        item(`${k}.${k}_min`, k, `${k}_min`, opts.min, "index"),
        item(`${k}.${k}_max`, k, `${k}_max`, opts.max, "index"),
        item(`${k}.${k}_quality_total_pixel_count`, k, `${k}_quality_total_pixel_count`, opts.total, "pixels"),
      ],
      execution: {
        plan: {
          intent, bbox: { west: 77.5879274, south: 12.9679621, east: 77.5989019, north: 12.9803864 },
          ...(opts.matched
            ? { matched_name: opts.matched.name, matched_class: opts.matched.class, matched_type: opts.matched.type }
            : {}),
        },
        executed_modalities: ["sentinel-2-optical"], skipped_modalities: [],
        windows: [window("single", opts.start, opts.end, opts.count, scene(opts.sceneId, opts.acquired, opts.platform, opts.cloud))],
        catalog: EARTH_SEARCH, catalogs: [EARTH_SEARCH], status: "completed", observations: null,
      },
      analysis: analysisResult({
        measurements: [],
        analysis_outcomes: [{ name: k, status: "completed", reason: null }],
        pixel_quality: [pixelQuality(k, opts.sceneId, "single", opts.valid, opts.total)],
        radiometry: [radiometry(opts.sceneId)],
        grids: [grid(k, opts.sceneId, opts.crs, opts.width, opts.height)],
      }),
    },
  } as unknown as AgentResult;
}

/** "Show vegetation around Cubbon Park, Bengaluru in December 2024" - live. */
export const NDVI_CUBBON = singleIndex({
  key: "ndvi", place: "Cubbon Park, Bengaluru", start: "2024-12-01", end: "2024-12-31",
  sceneId: "S2B_43PGQ_20241208_0_L2A", acquired: "2024-12-08T05:25:20.000000Z", platform: "sentinel-2b",
  cloud: 12.7, count: 7, mean: 0.5204459203473597, min: -0.08767303889255108, max: 0.8779860345461228,
  valid: 16988, total: 17080, crs: "EPSG:32643", width: 122, height: 140,
  answer: "The mean NDVI was 0.5204 index. Scene S2B_43PGQ_20241208_0_L2A was selected. The scene was acquired on 2024-12-08.",
  // Nominatim's own match for "Cubbon Park, Bengaluru" (live, 2026-09-24).
  matched: {
    name: "Cubbon Park, Sampangirama Nagar, Bengaluru Central City Corporation, Bengaluru, Karnataka, India",
    class: "leisure",
    type: "park",
  },
});

/** "Lalbagh, Bengaluru" - live: Nominatim matched a railway STOP node, not the garden. */
export const NDVI_LALBAGH_STOP = singleIndex({
  key: "ndvi", place: "Lalbagh, Bengaluru", start: "2024-12-01", end: "2024-12-31",
  sceneId: "S2B_43PGQ_20241208_0_L2A", acquired: "2024-12-08T05:25:20.000000Z", platform: "sentinel-2b",
  cloud: 12.7, count: 7, mean: 0.3727682271082574, min: 0.34341397849462363, max: 0.41478129713423834,
  valid: 4, total: 4, crs: "EPSG:32643", width: 2, height: 2,
  answer: "The mean NDVI was 0.3728 index. Scene S2B_43PGQ_20241208_0_L2A was selected. The scene was acquired on 2024-12-08.",
  matched: {
    name: "Lalbagh, Rashtriya Vidyalaya Road, Kankanpalya, Ashoka Pillar, Bengaluru, Karnataka, India",
    class: "railway",
    type: "stop",
  },
});

/** "Show water around Marina Beach, Chennai in January 2025" - live. */
export const NDWI_MARINA = singleIndex({
  key: "ndwi", place: "Marina Beach, Chennai", start: "2025-01-01", end: "2025-01-31",
  sceneId: "S2B_44PMV_20250104_0_L2A", acquired: "2025-01-04T05:15:13.077000Z", platform: "sentinel-2b",
  cloud: 14.2, count: 5, mean: 0.14657038314284795, min: -0.7808971620384498, max: 0.9729119638826185,
  valid: 33524, total: 33600, crs: "EPSG:32644", width: 112, height: 300,
  answer: "The mean NDWI was 0.1466 index. Scene S2B_44PMV_20250104_0_L2A was selected. The scene was acquired on 2025-01-04.",
});

/** "Show built-up area around Ameerpet, Hyderabad in January 2025" - live. */
export const NDBI_AMEERPET = singleIndex({
  key: "ndbi", place: "Ameerpet, Hyderabad", start: "2025-01-01", end: "2025-01-31",
  sceneId: "S2B_43QHV_20250107_0_L2A", acquired: "2025-01-07T05:24:04.000000Z", platform: "sentinel-2b",
  cloud: 0.0, count: 10, mean: -0.024820557618541204, min: -0.6520763187429854, max: 0.7704145293220575,
  valid: 196620, total: 196620, crs: "EPSG:32643", width: 435, height: 452,
  answer: "The mean NDBI was -0.02482 index. Scene S2B_43QHV_20250107_0_L2A was selected. The scene was acquired on 2025-01-07.",
});

/** "Analyze SAR backscatter around Marina Beach, Chennai in January 2025" - live. */
export const SAR_MARINA: AgentResult = (() => {
  const intent = { location_query: "Marina Beach, Chennai", time_windows: [{ start_date: "2025-01-01", end_date: "2025-01-31" }], modalities: ["sentinel-1-sar"], task: "visualize", temporal_mode: "single" };
  const p = plan(intent, { tool: "sar_backscatter_statistics" });
  const sceneId = "S1A_IW_GRDH_1SDV_20250111T003153_20250111T003218_057389_071055_rtc";
  const sar = (name: string, value: number, unit = "dB") => item(`sar_backscatter.${name}`, "sar_backscatter", name, value, unit);
  return {
    status: "ok",
    answer: "The mean VV was -5.444 dB. The mean VH was -17.85 dB. The VV minus VH difference was 12.41 dB. Scene " + sceneId + " was selected. The scene was acquired on 2025-01-11.",
    failure: null, clarification: null,
    trace: { plan: p, steps: okSteps(p), evidence_refs: ["sar_backscatter.vv_mean_db"], answer_validation: PASS },
    evidence: {
      items: [
        item("execution.sentinel-1-sar.single.scene_count", "execution", "sentinel-1-sar_single_scene_count", 2, "count"),
        sar("vv_valid_pixel_count", 33600, "pixels"), sar("vv_mean_db", -5.4437196664666985),
        sar("vh_valid_pixel_count", 33600, "pixels"), sar("vh_mean_db", -17.849565),
        sar("vv_minus_vh_mean_db", 12.405845),
        note("sar_backscatter.warning.1", "sar_backscatter", "Values are the provider's radiometrically terrain-corrected gamma naught (Sentinel-1 RTC); SatQuery performs no calibration."),
      ],
      execution: {
        plan: { intent, bbox: { west: 80.28, south: 13.039, east: 80.29, north: 13.066 } },
        executed_modalities: ["sentinel-1-sar"], skipped_modalities: [],
        windows: [window("single", "2025-01-01", "2025-01-31", 2, { ...scene(sceneId, "2025-01-11T00:32:06.000000Z", "sentinel-1a", 0), collection: "sentinel-1-rtc", processing_level: "RTC" }, "sentinel-1-sar")],
        catalog: "https://planetarycomputer.microsoft.com/api/stac/v1", catalogs: [], status: "completed", observations: null,
      },
      analysis: analysisResult({
        analysis_outcomes: [{ name: "sar_backscatter", status: "completed", reason: null }],
        sar_backscatter: {
          scene_id: sceneId, window_label: "single", acquired_at: "2025-01-11T00:32:06.000000Z", collection: "sentinel-1-rtc",
          polarizations: [
            { polarization: "vv", measurements: [], valid_pixel_count: 33600, nonpositive_pixel_count: 0, window_pixel_count: 33600, crs: "EPSG:32644", resolution: 10, transform: null },
            { polarization: "vh", measurements: [], valid_pixel_count: 33600, nonpositive_pixel_count: 0, window_pixel_count: 33600, crs: "EPSG:32644", resolution: 10, transform: null },
          ],
          difference: null, measurements: [], warnings: [],
        },
        radiometry: [radiometry(sceneId, "sentinel-1-sar", null)],
      }),
    },
  } as unknown as AgentResult;
})();

/** "Compare water at Marina Beach, Chennai between January 2024 and January 2025" - live. */
export const TEMPORAL_MARINA: AgentResult = (() => {
  const intent = { location_query: "Marina Beach, Chennai", time_windows: { baseline: { start_date: "2024-01-01", end_date: "2024-01-31" }, target: { start_date: "2025-01-01", end_date: "2025-01-31" } }, modalities: ["sentinel-2-optical"], task: "visualize", temporal_mode: "compare" };
  const p = plan(intent, { tool: "temporal_ndwi_statistics" });
  const t = (id: string, name: string, value: number, unit = "index") => item(`temporal_ndwi.${id}`, "temporal_ndwi", name, value, unit);
  const first = scene("S2A_44PMV_20240115_0_L2A", "2024-01-15T05:15:05.886000Z", "sentinel-2a", 14.28);
  const second = scene("S2B_44PMV_20250104_0_L2A", "2025-01-04T05:15:13.077000Z", "sentinel-2b", 14.19);
  const observation = (label: string, s: Json, valid: number, baseline: string) => ({
    window_label: label, scene_id: s.id, acquired_at: s.datetime, cloud_cover: s.cloud_cover,
    measurements: [], transform: null, pixel_quality: pixelQuality("ndwi", s.id as string, label, valid, 33600),
    radiometry: radiometry(s.id as string, "sentinel-2-optical", baseline),
    grid: grid("ndwi", s.id as string, "EPSG:32644", 112, 300),
  });
  return {
    status: "ok",
    answer: "The earlier mean NDWI was 0.02658 index. The later mean NDWI was 0.1466 index. The mean NDWI difference was 0.12 index.",
    failure: null, clarification: null,
    trace: { plan: p, steps: okSteps(p), evidence_refs: ["temporal_ndwi.difference.mean_ndwi_difference"], answer_validation: PASS },
    evidence: {
      items: [
        t("first.ndwi_valid_pixel_count", "ndwi_valid_pixel_count", 33496, "pixels"),
        t("first.ndwi_mean", "ndwi_mean", 0.02658240949942237),
        t("second.ndwi_valid_pixel_count", "ndwi_valid_pixel_count", 33524, "pixels"),
        t("second.ndwi_mean", "ndwi_mean", 0.14657038314284795),
        t("difference.mean_ndwi_difference", "mean_ndwi_difference", 0.11998797364342559),
        t("change.ndwi_change_mean", "ndwi_change_mean", 0.12069164662445794),
        t("change.paired_valid_pixel_count", "paired_valid_pixel_count", 33420, "pixels"),
      ],
      execution: {
        plan: { intent, bbox: { west: 80.28, south: 13.039, east: 80.29, north: 13.066 } },
        executed_modalities: ["sentinel-2-optical"], skipped_modalities: [],
        windows: [
          window("baseline", "2024-01-01", "2024-01-31", 6, first),
          window("target", "2025-01-01", "2025-01-31", 5, second),
        ],
        catalog: EARTH_SEARCH, catalogs: [EARTH_SEARCH], status: "completed", observations: null,
      },
      analysis: analysisResult({
        analysis_outcomes: [{ name: "temporal_ndwi", status: "completed", reason: null }],
        temporal_comparison: {
          first: observation("baseline", first, 33496, "05.10"),
          second: observation("target", second, 33524, "05.11"),
          compatibility: {}, differences: [], change: null, warnings: [],
          pair_grid: grid("temporal_ndwi_pair", null as unknown as string, "EPSG:32644", 112, 300),
        },
      }),
    },
  } as unknown as AgentResult;
})();

/** "Analyze Chennai" - live. */
export const CLARIFY_CHENNAI: AgentResult = {
  status: "needs_clarification",
  answer: null,
  failure: null,
  clarification: {
    reason: "analysis_missing",
    message: "What would you like to analyse at Chennai?",
    options: ["vegetation (NDVI)", "water (NDWI)", "built-up area (NDBI)", "SAR backscatter (Sentinel-1 VV/VH)", "water change between two periods (NDWI)"],
    option_questions: ["Show vegetation (NDVI) around Chennai", "Show water (NDWI) around Chennai", "Show built-up area (NDBI) around Chennai", "Analyze SAR backscatter around Chennai", "Compare water (NDWI) around Chennai"],
    understood_analyses: [], understood_location: "Chennai", understood_periods: [],
  },
  trace: { plan: null, steps: [], evidence_refs: [], answer_validation: null },
  evidence: { items: [], execution: null, analysis: null },
} as unknown as AgentResult;

function refusal(reason: string, message: string): AgentResult {
  return { ...CLARIFY_CHENNAI, clarification: { ...CLARIFY_CHENNAI.clarification!, reason, message, options: [], option_questions: [] } } as unknown as AgentResult;
}

/** "NDVI Chennai January 2025" - live: M1 refused the whole city. */
export const AREA_TOO_LARGE_CHENNAI = refusal(
  "area_too_large",
  "'Chennai' is too large to analyse: this area is about 20.9 x 42.4 km. Measurements are read on the sensor's native 10 m grid and are never downsampled, which limits an analysis area to roughly 20 km across. Choose a smaller area. Name a neighbourhood, park or landmark in Chennai together with the city name, or give coordinates as 'lat, lon'.",
);

/** "Count ships in Chennai harbor" - live. */
export const UNSUPPORTED_SHIPS = refusal(
  "analysis_unsupported",
  "Object identification and land-cover classification are not implemented. SatQuery measures spectral indices and SAR backscatter over an area.",
);

/** A place the geocoder could not find. */
export const NOT_FOUND = refusal("location_not_found", "No place matching 'Qwxzt' was found.");

/** The geocoder refused (429) - the location_unavailable state. */
export const LOCATION_UNAVAILABLE: AgentResult = (() => {
  const intent = { location_query: "Dal Lake", time_windows: [{ start_date: "2025-01-01", end_date: "2025-01-31" }], modalities: ["sentinel-2-optical"], task: "visualize", temporal_mode: "single" };
  const p = plan(intent, { tool: "spectral_indices", indices: ["ndwi"] });
  return {
    status: "location_unavailable",
    answer: null,
    clarification: null,
    failure: { stage: "location", code: "geocoding_unavailable", dependency: "geocoder", message: "Location service temporarily unavailable. The place could not be looked up, so no scene was searched and nothing was measured.", retry_after_seconds: 45 },
    trace: {
      plan: p,
      steps: [
        { status: "failed", parameters: (p.steps as Json[])[0], rejection_reason: null, error_message: "Location lookup is temporarily unavailable." },
        { status: "skipped", parameters: (p.steps as Json[])[1], rejection_reason: null, error_message: null },
      ],
      evidence_refs: [], answer_validation: null,
    },
    evidence: { items: [note("execution.discovery_failure", "execution", "Scene discovery did not complete: Location lookup is temporarily unavailable.")], execution: null, analysis: null },
  } as unknown as AgentResult;
})();

/** Cubbon Park, January 2025 - live: the scene was found and M4 REFUSED it. */
export const REFUSED_BY_RADIOMETRY: AgentResult = (() => {
  const base = NDVI_CUBBON as unknown as { evidence: Json; trace: Json };
  const reason = "NDVI was not computed: Radiometric validation failed (radiometric_undetermined): scene S2C_43PGQ_20250122_0_L2A: on baseline 05.11 the provider states the reflectance offset was NOT removed.";
  return {
    ...NDVI_CUBBON,
    answer: "Scene S2C_43PGQ_20250122_0_L2A was selected. The scene was acquired on 2025-01-22.",
    evidence: {
      ...base.evidence,
      items: [item("execution.sentinel-2-optical.single.scene_count", "execution", "sentinel-2-optical_single_scene_count", 6, "count"), note("execution.warning.0", "execution", reason)],
      analysis: analysisResult({ analysis_outcomes: [{ name: "ndvi", status: "unavailable", reason }], completeness: "none", warnings: [reason] }),
    },
  } as unknown as AgentResult;
})();

/** A place and month with no matching scene - the genuine evidence verdict. */
export const NO_SCENES: AgentResult = (() => {
  const base = NDVI_CUBBON as unknown as { evidence: { execution: Json } };
  const execution = base.evidence.execution as { windows: Json[] };
  return {
    ...NDVI_CUBBON,
    answer: "Insufficient evidence to answer the question.",
    evidence: {
      items: [item("execution.sentinel-2-optical.single.scene_count", "execution", "sentinel-2-optical_single_scene_count", 0, "count")],
      execution: { ...base.evidence.execution, windows: [{ ...execution.windows[0], scene_count: 0, scenes: [], selected_scene_id: null }] },
      analysis: null,
    },
  } as unknown as AgentResult;
})();
