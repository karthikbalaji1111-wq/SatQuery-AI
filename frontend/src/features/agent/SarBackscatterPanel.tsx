import type { SarBackscatterResult } from "../../api/types";

/** Render server measurements verbatim; never estimate dB from display pixels. */
export function SarBackscatterPanel({ result }: { result: SarBackscatterResult | null }) {
  if (result === null) return null;
  const difference = result.difference;
  return (
    <section className="sar-backscatter" aria-label="SAR backscatter statistics">
      <h3>Sentinel-1 RTC backscatter</h3>
      <p className="hint">Provider-processed gamma naught · decibels (dB)</p>
      <dl className="evidence-grid">
        <div><dt>SAR scene</dt><dd>{result.scene_id}</dd></div>
        {result.acquired_at && <div><dt>SAR acquisition</dt><dd>{result.acquired_at}</dd></div>}
        {result.collection && <div><dt>Source collection</dt><dd>{result.collection}</dd></div>}
        <div><dt>Observation window</dt><dd>{result.window_label}</dd></div>
      </dl>
      <div className="sar-polarizations">
        {result.polarizations.map((polarization) => {
          const label = polarization.polarization.toUpperCase();
          const mean = polarization.measurements.find((measurement) =>
            measurement.name === `${polarization.polarization}_mean_db` && measurement.unit === "dB",
          );
          return (
            <section key={label} aria-label={`${label} backscatter`}>
              <h4>{label}</h4>
              <dl className="evidence-grid">
                {mean && <div><dt>{label} mean</dt><dd>{mean.value.toFixed(4)} dB</dd></div>}
                <div><dt>{label} valid pixel count</dt><dd>{polarization.valid_pixel_count}</dd></div>
                <div><dt>{label} window pixel count</dt><dd>{polarization.window_pixel_count}</dd></div>
                <div><dt>{label} excluded nonpositive pixels</dt><dd>{polarization.nonpositive_pixel_count}</dd></div>
                {polarization.crs && <div><dt>{label} CRS</dt><dd>{polarization.crs}</dd></div>}
                {polarization.resolution !== null && <div><dt>{label} pixel size</dt><dd>{polarization.resolution} m</dd></div>}
              </dl>
              {!mean && <p className="hint">No mean was produced for {label}.</p>}
            </section>
          );
        })}
      </div>
      {difference && (
        <section aria-label="Paired polarization difference">
          <h4>VV−VH · paired pixels</h4>
          <dl className="evidence-grid">
            <div><dt>VV−VH mean difference</dt><dd>{difference.vv_minus_vh_mean_db.toFixed(4)} dB</dd></div>
            <div><dt>Paired valid pixel count</dt><dd>{difference.paired_valid_pixel_count}</dd></div>
            <div><dt>Paired VV mean</dt><dd>{difference.vv_mean_db.toFixed(4)} dB</dd></div>
            <div><dt>Paired VH mean</dt><dd>{difference.vh_mean_db.toFixed(4)} dB</dd></div>
          </dl>
        </section>
      )}
      <p className="hint">Means are computed in linear power, then converted using 10·log10. VV−VH compares the means on common valid pixels; it is not a land-cover classification.</p>
      <p className="hint">Terrain correction and radiometric processing are supplied by the RTC provider. SatQuery does not perform additional calibration or speckle filtering. Valid pixels are not a quality score; no per-pixel quality mask is available.</p>
      {result.warnings.map((warning) => <p className="hint hint-limitation" key={warning}>{warning}</p>)}
    </section>
  );
}
