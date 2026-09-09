/**
 * Turns a backend `imagery_error` into something a reader can act on.
 *
 * The server's message is written for an operator: it names the URI scheme,
 * the deployment's read policy and a README section. That is the right message
 * to log and the wrong one to put in front of someone watching a demo, where
 * it reads as a crash rather than as a documented boundary.
 *
 * So the cause is stated plainly and the server's own words are kept as the
 * detail. Nothing is hidden - a reader who wants the technical cause can open
 * it - but the headline says what happened and what still holds, because a
 * failed image retrieval does NOT invalidate the discovery that preceded it.
 */

export interface ImageryErrorNotice {
  /** One sentence: what failed. */
  summary: string;
  /** One sentence: what is still true, so the reader can calibrate. */
  reassurance: string;
  /** The server's own message, for a reader who wants the cause. */
  detail: string;
  /**
   * `limitation` for a documented boundary the system is expected to hit,
   * `failure` for something that genuinely went wrong. These are styled
   * differently on purpose: a known limitation shown in alarm red trains a
   * reader to distrust the whole panel.
   */
  kind: "limitation" | "failure";
}

/**
 * A Sentinel-1 measurement asset published on object storage this deployment
 * holds no credentials for. Verified against the live catalog and documented -
 * expected, not broken.
 */
function isSentinel1AssetLimitation(raw: string): boolean {
  const text = raw.toLowerCase();
  return text.includes("s3://") || text.includes("only anonymous https");
}

/** An asset the bounded reader cannot window into, whatever the sensor. */
function isUnreadableAsset(raw: string): boolean {
  return raw.toLowerCase().includes("windowed-readable");
}

export function describeImageryError(raw: string): ImageryErrorNotice {
  if (isSentinel1AssetLimitation(raw)) {
    return {
      summary:
        "Sentinel-1 imagery is not available for this scene: the catalog " +
        "publishes its measurement asset on storage this deployment cannot " +
        "read.",
      reassurance:
        "Scene discovery and metadata are unaffected - the scene below was " +
        "found and selected normally.",
      detail: raw,
      kind: "limitation",
    };
  }

  if (isUnreadableAsset(raw)) {
    return {
      summary:
        "This scene's asset cannot be read as a bounded window, so no image " +
        "was retrieved.",
      reassurance: "Scene discovery and metadata are unaffected.",
      detail: raw,
      kind: "limitation",
    };
  }

  return {
    summary: "The image for this scene could not be retrieved.",
    reassurance:
      "Scene discovery and metadata are unaffected; the measurements shown " +
      "were computed independently of the picture.",
    detail: raw,
    kind: "failure",
  };
}
