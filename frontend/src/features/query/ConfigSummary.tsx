import type {
  Modality,
  QueryTask,
  SatelliteScene,
  SpectralIndexKey,
} from "../../api/types";

/**
 * The configuration a run actually used, as an analytical read-out.
 *
 * Every field is a value the application already holds - a resolved place, the
 * window that was requested, the modalities selected, the scenes discovery
 * returned. Nothing is defaulted or invented: before a run resolves something
 * the field says so rather than showing a plausible placeholder.
 */
export interface RunContext {
  /** Resolved place name, or `null` before a location has been resolved. */
  location: string | null;
  /** Centre of the resolved extent, `null` when no extent exists yet. */
  centre: { lat: number; lon: number } | null;
  /** The acquisition window as requested, already formatted. */
  window: string | null;
  /** A stated cloud-cover limit, when the request carried one. */
  cloudRule: string | null;
  modalities: Modality[];
  task: QueryTask;
  /** Whether an NDWI computation was requested. */
  ndwi: boolean;
  /**
   * The spectral indices this run actually computed.
   *
   * An index missing from here was NOT COMPUTED FOR THIS QUERY - which says
   * nothing about whether the system can compute it. That is a different
   * statement from a task the backend does not implement, and the rail must
   * not render the two the same way.
   */
  indices?: SpectralIndexKey[];
  /** Candidates discovery returned, in the order it returned them. */
  scenes: SatelliteScene[];
  /** The scene deterministic selection chose, when one was chosen. */
  selectedSceneId: string | null;
}

const SENSORS: { modality: Modality; name: string; detail: string }[] = [
  { modality: "sentinel-2-optical", name: "Sentinel-2", detail: "L2A · optical" },
  { modality: "sentinel-1-sar", name: "Sentinel-1", detail: "SAR · radar" },
];

/**
 * The task chips.
 *
 * `change_detection` and `object_identification` are selectable but the
 * backend answers them with `status: "not_implemented"`, so both are marked
 * here exactly as they are in the task selector. An active chip reading
 * "Change detect" with no qualifier would let a Temporal NDWI Statistics
 * result be read as a change-detection result, which is the one confusion
 * this whole layer exists to prevent.
 *
 * These say "not implemented", NOT merely "unavailable". The distinction is
 * the point: this capability does not exist in the system, and no query, no
 * retry and no configuration will produce it. That is a different fact from an
 * index that simply was not needed for the question asked - see INDICES below,
 * which are all fully implemented and are marked per-run instead.
 */
const TASKS: { task: QueryTask; label: string; available: boolean }[] = [
  { task: "visualize", label: "Visualize", available: true },
  { task: "change_detection", label: "Change detect", available: false },
  { task: "object_identification", label: "Object ID", available: false },
];

/**
 * The spectral indices, all of which the backend implements.
 *
 * Every one of these is a working capability. Whether it appears in a given
 * run is a property of the QUESTION - NDBI is not computed for a question
 * about water because it was not asked for, not because it is missing. So an
 * unused index reads "not used for this query" and never "unavailable".
 */
const INDICES: { key: SpectralIndexKey; label: string }[] = [
  { key: "ndvi", label: "NDVI" },
  { key: "ndwi", label: "NDWI" },
  { key: "ndbi", label: "NDBI" },
];

const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/** `2024-01-15T05:15:05Z` → `15 Jan 2024 · 05:15 UTC`. Sliced, never parsed. */
function sceneStamp(iso: string): string {
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(iso);
  if (match === null) return iso;
  const [, year, month, day, hour, minute] = match;
  return `${day} ${MONTHS[Number(month) - 1]} ${year} · ${hour}:${minute} UTC`;
}

function formatCentre(centre: { lat: number; lon: number }): string {
  const ns = centre.lat >= 0 ? "N" : "S";
  const ew = centre.lon >= 0 ? "E" : "W";
  return (
    `${Math.abs(centre.lat).toFixed(4)}° ${ns} · ` +
    `${Math.abs(centre.lon).toFixed(4)}° ${ew}`
  );
}

export function ConfigSummary({ context }: { context: RunContext }) {
  const {
    location,
    centre,
    window: acquisitionWindow,
    cloudRule,
    modalities,
    task,
    ndwi,
    indices,
    scenes,
    selectedSceneId,
  } = context;

  // A run reports exactly which indices it computed. Older callers carry only
  // the NDWI flag; that is honoured rather than guessed past.
  const computed: SpectralIndexKey[] =
    indices ?? (ndwi ? ["ndwi"] : []);

  return (
    <div className="config-summary">
      <section className="config-block">
        <h3 className="eyebrow">Location</h3>
        {location === null ? (
          <p className="config-awaiting">Awaiting query</p>
        ) : (
          <>
            <p className="config-value">{location}</p>
            {centre !== null && (
              <p className="config-coords">{formatCentre(centre)}</p>
            )}
          </>
        )}
      </section>

      <section className="config-block">
        <h3 className="eyebrow">Date / time window</h3>
        {acquisitionWindow === null ? (
          <p className="config-awaiting">Awaiting query</p>
        ) : (
          <>
            <p className="config-mono">{acquisitionWindow}</p>
            {cloudRule !== null && <p className="config-note">{cloudRule}</p>}
          </>
        )}
      </section>

      <section className="config-block">
        <h3 className="eyebrow">Satellite / sensor</h3>
        <div className="sensor-grid">
          {SENSORS.map((sensor) => (
            <div
              key={sensor.modality}
              className="sensor-card"
              data-active={modalities.includes(sensor.modality)}
            >
              <span className="sensor-name">{sensor.name}</span>
              <span className="sensor-detail">{sensor.detail}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="config-block">
        <h3 className="eyebrow">Analysis type</h3>
        <div className="analysis-grid">
          {TASKS.map((option) => (
            <span
              key={option.task}
              className="analysis-chip"
              data-active={option.task === task}
              data-supported={option.available}
              title={
                option.available
                  ? undefined
                  : "This capability is not built. The server answers this task with not_implemented; no query or retry will produce a result."
              }
            >
              {option.label}
              {option.available ? "" : " (not implemented)"}
            </span>
          ))}
        </div>
        {/* Named apart from the task chips above on purpose. Those two words -
            "not implemented" - describe a capability that does not exist.
            Everything in this second row DOES exist and simply was not needed
            for the question that was asked. Rendering both as one greyed row
            told the reader those were the same kind of absence. */}
        <h3 className="eyebrow config-subhead">Spectral indices</h3>
        <div className="analysis-grid">
          {INDICES.map((option) => {
            const on = computed.includes(option.key);
            return (
              <span
                key={option.key}
                className="analysis-chip"
                data-active={on}
                data-supported={true}
                title={
                  on
                    ? `${option.label} was computed for this run.`
                    : `${option.label} is supported but was not requested for this query.`
                }
              >
                {option.label}
              </span>
            );
          })}
        </div>
        <p className="config-note">
          {computed.length === 0
            ? "No spectral index was computed for this query. All three are available."
            : "Indices not highlighted are supported, but were not needed for this question."}
        </p>
      </section>

      <section className="config-block">
        <h3 className="eyebrow">
          Scene candidates{scenes.length > 0 && ` · ${scenes.length}`}
        </h3>
        {scenes.length === 0 ? (
          <p className="config-awaiting">No scenes discovered yet</p>
        ) : (
          <ul className="candidate-list">
            {scenes.map((scene) => (
              <li
                key={scene.id}
                className="candidate"
                data-selected={scene.id === selectedSceneId}
              >
                <span className="candidate-id">{scene.id}</span>
                {scene.datetime !== null && (
                  <span className="candidate-time">
                    {sceneStamp(scene.datetime)}
                  </span>
                )}
                {scene.cloud_cover !== null && (
                  <span
                    className="candidate-cloud"
                    data-level={
                      scene.cloud_cover <= 10
                        ? "low"
                        : scene.cloud_cover <= 30
                          ? "mid"
                          : "high"
                    }
                  >
                    {scene.cloud_cover.toFixed(0)}%
                  </span>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
