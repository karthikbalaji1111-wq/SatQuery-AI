/**
 * The auditable record of one analysis, assembled from what is already on
 * screen.
 *
 * Everything here is data the browser already holds after a run, so building
 * the report asks the server for nothing and cannot re-run, re-fetch or
 * disagree with what the reader was shown. It is a transcript, not a second
 * computation.
 *
 * The point is traceability: a number quoted from the interface should be
 * findable in this file next to the scene it came from, the plan that chose to
 * compute it, and the checks it passed. That includes the unflattering parts -
 * a withheld answer, a failed grounding check, a Sentinel-1 window whose
 * imagery could not be retrieved. A report that only records successes is not
 * an audit.
 */

import type {
  AgentResult,
  AnalysisResult,
  ImageryResponse,
  Measurement,
  QueryExecutionResult,
  SatQueryIntent,
  SatelliteScene,
  SarBackscatterResult,
} from "../../api/types";

export interface ManualEvidenceInput {
  scene: SatelliteScene | null;
  imagery: ImageryResponse | null;
  measurements: Measurement[];
  sar_backscatter?: SarBackscatterResult | null;
  /**
   * The provenance of a MANUAL run. Without these the exported record carried a
   * scene, a picture and some numbers, and could not say what was asked, which
   * catalog answered, which windows failed or what the analysis warned about -
   * so it read as more authoritative than the run behind it.
   */
  intent?: SatQueryIntent | null;
  execution?: QueryExecutionResult | null;
  analysis?: AnalysisResult | null;
}

export interface EvidenceReport {
  /** Report format version, so a consumer can tell shapes apart later. */
  report_version: string;
  generated_at: string;
  question: string | null;
  answer: string | null;
  /** One of the four agent statuses, or null when no agent run produced it. */
  status: string | null;
  failure: AgentResult["failure"] | null;
  /**
   * What became of a requested description of the image - observed, or why
   * not - with the scene it concerns. Null when none was asked for.
   */
  visual: AgentResult["visual"] | null;
  /**
   * The mechanical checks the answer was put through, verbatim. Present even
   * when they failed - especially when they failed.
   */
  answer_validation: unknown;
  /** The validated plan and per-step outcomes. */
  trace: unknown;
  /** The flattened, citable evidence items. */
  evidence: unknown;
  /** Whatever the manual (non-agent) path established, when it did. */
  manual: ManualEvidenceInput | null;
  /**
   * Retrieval failures per window, kept as their own section so a reader does
   * not have to notice an absence to learn something failed.
   */
  imagery_errors: { modality: string; window: string; reason: string }[];
  /**
   * Windows whose discovery failed outright, kept apart from the above: a
   * catalog that could not be asked is not an archive that held nothing.
   */
  discovery_failures: { modality: string; window: string; reason: string }[];
  /** Every warning the run produced, from execution and analysis alike. */
  warnings: string[];
}

const REPORT_VERSION = "1";

/**
 * Pull retrieval failures out of the execution result.
 *
 * These are the honest half of a partial run: Sentinel-1 discovery can succeed
 * while its imagery cannot be retrieved, and the export has to say so rather
 * than quietly omitting the window.
 */
function imageryErrorsFrom(
  result: AgentResult | null,
  manual: ManualEvidenceInput | null,
) {
  // Both workflows, because both can fail this way and the export is about the
  // run that happened, not about which path produced it.
  const windows = [
    ...(result?.evidence?.execution?.windows ?? []),
    ...(manual?.execution?.windows ?? []),
  ];
  return windows
    .filter((window) => window.imagery_error !== null)
    .map((window) => ({
      modality: window.modality,
      window: window.label,
      reason: window.imagery_error as string,
    }));
}

/**
 * Windows whose DISCOVERY failed - the catalog could not be asked at all.
 *
 * Kept apart from `imagery_errors`, which means the scene was found and its
 * picture could not be read. Collapsing the two would let a reader conclude an
 * archive was empty when it was never searched.
 */
function discoveryFailuresFrom(
  result: AgentResult | null,
  manual: ManualEvidenceInput | null,
) {
  const windows = [
    ...(result?.evidence?.execution?.windows ?? []),
    ...(manual?.execution?.windows ?? []),
  ];
  return windows
    .filter((window) => window.error)
    .map((window) => ({
      modality: window.modality,
      window: window.label,
      reason: window.error as string,
    }));
}

function warningsFrom(
  result: AgentResult | null,
  manual: ManualEvidenceInput | null,
): string[] {
  const analyses = [result?.evidence?.analysis, manual?.analysis];
  return analyses.flatMap((analysis) => {
    const comparison = analysis?.temporal_comparison;
    return [
      ...(analysis?.warnings ?? []),
      ...(comparison?.warnings ?? []),
      ...(comparison?.compatibility?.limitations ?? []),
    ];
  });
}

export function buildEvidenceReport(
  result: AgentResult | null,
  manual: ManualEvidenceInput | null,
  question: string | null,
  now: Date = new Date(),
): EvidenceReport {
  return {
    report_version: REPORT_VERSION,
    generated_at: now.toISOString(),
    question: question?.trim() ? question.trim() : null,
    answer: result?.answer ?? null,
    status: result?.status ?? null,
    failure: result?.failure ?? null,
    visual: result?.visual ?? null,
    answer_validation: result?.trace?.answer_validation ?? null,
    trace: result?.trace ?? null,
    evidence: result?.evidence ?? null,
    manual,
    imagery_errors: imageryErrorsFrom(result, manual),
    discovery_failures: discoveryFailuresFrom(result, manual),
    warnings: warningsFrom(result, manual),
  };
}

/**
 * A filename derived from the run, safe on every filesystem.
 *
 * The place name comes from a geocoder and the question from the user, so both
 * are untrusted for this purpose: anything outside a conservative set is
 * replaced rather than escaped, path separators included. The date makes a
 * series of exports sort in the order they were taken.
 */
export function evidenceFilename(
  place: string | null,
  now: Date = new Date(),
): string {
  const day = now.toISOString().slice(0, 10);
  const slug = (place ?? "")
    .toLowerCase()
    .normalize("NFKD")
    // Anything that is not a plain letter or digit becomes a separator, which
    // removes path traversal, separators and control characters in one step
    // rather than trying to enumerate what to strip.
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60)
    .replace(/-+$/g, "");
  return slug
    ? `satquery-analysis-${day}-${slug}.json`
    : `satquery-analysis-${day}.json`;
}
