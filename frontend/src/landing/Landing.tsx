/**
 * The landing page: seven chapters scrolling over one continuous world.
 *
 * The 3D scene is a backdrop. Every claim, every step of the pipeline and every
 * capability boundary is written here in ordinary HTML, so the page is complete
 * with the canvas switched off, animation disabled, or JavaScript failing to
 * load the planet. Nothing is communicated by motion alone.
 *
 * The copy is held to what the system actually does. Sentinel-1 appears as
 * provider-processed RTC imagery, indices are named rather than classifications,
 * and there are no metrics, customers or claims that would need a footnote.
 */

import { useEffect, useRef, useState } from "react";

import { EarthCanvas } from "./EarthCanvas";

/** Reveals a section once it has genuinely entered the viewport. */
function useReveal<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [shown, setShown] = useState(false);
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    if (!("IntersectionObserver" in window)) {
      setShown(true);
      return;
    }
    const observer = new IntersectionObserver(
      ([entry]) => {
        // One-way: content that fades back out on scroll-up is irritating to
        // re-read and hostile to anyone scrubbing back for a detail.
        if (entry.isIntersecting) {
          setShown(true);
          observer.disconnect();
        }
      },
      { rootMargin: "-12% 0px -12% 0px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  return { ref, shown };
}

function Chapter({
  id,
  align = "left",
  children,
}: {
  id: string;
  align?: "left" | "right" | "centre";
  children: React.ReactNode;
}) {
  const { ref, shown } = useReveal<HTMLElement>();
  return (
    <section
      id={id}
      ref={ref}
      className="chapter"
      data-align={align}
      data-shown={shown}
    >
      <div className="chapter-inner">{children}</div>
    </section>
  );
}

/** A quiet editorial aside. Used sparingly - four times in the whole page. */
function Aside({ children }: { children: React.ReactNode }) {
  return <p className="aside">{children}</p>;
}

const PIPELINE = [
  { step: "Your question", detail: "Plain language. No syntax to learn." },
  { step: "Location + time", detail: "Geocoded to an area of interest and a date window." },
  { step: "Satellite data", detail: "Scenes discovered in the Sentinel catalog, selected deterministically." },
  { step: "Deterministic analysis", detail: "Spectral indices computed from the raster itself." },
  { step: "Visual observation", detail: "A vision-language model describes what is visible." },
  { step: "Grounded answer", detail: "Checked against the evidence before you see it." },
];

const QUESTIONS = [
  { ask: "Is there visible water here?", answers: "NDWI", note: "water-like response" },
  { ask: "What is the vegetation condition?", answers: "NDVI", note: "vegetation-like response" },
  { ask: "Analyse the built-up area.", answers: "NDBI", note: "built-up / bare response" },
  { ask: "How has water changed over time?", answers: "Temporal NDWI", note: "earlier vs later" },
];

const TRAIL = [
  "Question",
  "Scene + location",
  "Bands + measurements",
  "Visual observation",
  "Grounding checks",
  "Final answer",
];

export function Landing() {
  return (
    <>
      <EarthCanvas />

      <a className="skip-link" href="#main">
        Skip to content
      </a>

      <header className="landing-header">
        <span className="wordmark">
          SatQuery<span aria-hidden="true">AI</span>
        </span>
        <nav aria-label="Landing sections">
          <a href="#pipeline">How it works</a>
          <a href="#multimodal">Data</a>
          <a href="#evidence">Evidence</a>
          <a className="nav-cta" href="/">
            Try SatQuery
          </a>
        </nav>
      </header>

      <main id="main">
        {/* ---------------------------------------------- 1. hero ------- */}
        <Chapter id="hero">
          <h1 className="hero-title">
            Ask the Earth.
            <br />
            Get grounded answers.
          </h1>
          <p className="lede">
            SatQuery AI is a vision-language assistant for multimodal
            remote-sensing analysis. Ask natural-language questions and get
            evidence-backed insights from real satellite data.
          </p>
          <div className="cta-row">
            <a className="cta cta-primary" href="/">
              Try SatQuery
            </a>
            <a className="cta cta-secondary" href="#pipeline">
              See how it works
            </a>
          </div>
          <p className="scroll-cue">
            <span aria-hidden="true">↓</span> Scroll to explore
          </p>
        </Chapter>

        {/* ------------------------------------------- 2. problem ------- */}
        <Chapter id="problem" align="right">
          <h2>
            Satellite data is powerful.
            <br />
            Using it shouldn&rsquo;t be hard.
          </h2>
          <p>
            Every day, satellites return more of the planet than anyone can
            read. The imagery is there, and it is public. What stands between a
            question and an answer is everything in between.
          </p>
          <ul className="plain-list">
            <li>
              <strong>Finding the right scene</strong> means catalogs, tiles,
              orbits and cloud cover.
            </li>
            <li>
              <strong>Different sensors</strong> see differently — optical needs
              daylight, radar does not.
            </li>
            <li>
              <strong>Different resolutions</strong> do not line up, and pretending
              they do is how mistakes are made.
            </li>
            <li>
              <strong>The vocabulary is expert</strong>. You should not need to
              know what a normalised difference is to ask about water.
            </li>
          </ul>
          <Aside>Look closer. The data was never the hard part.</Aside>
        </Chapter>

        {/* ------------------------------------------ 3. pipeline ------- */}
        <Chapter id="pipeline">
          <h2>
            From your question
            <br />
            to a grounded answer.
          </h2>
          <p>
            One path, every time. Each step hands the next something it can
            check.
          </p>
          <ol className="pipeline">
            {PIPELINE.map((item, index) => (
              <li key={item.step}>
                <span className="pipeline-index" aria-hidden="true">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <span className="pipeline-step">{item.step}</span>
                <span className="pipeline-detail">{item.detail}</span>
              </li>
            ))}
          </ol>
          <Aside>
            The model chooses which analysis to run. It never computes the
            result.
          </Aside>
        </Chapter>

        {/* ---------------------------------------- 4. multimodal ------- */}
        <Chapter id="multimodal" align="right">
          <h2>Two ways of seeing.</h2>
          <div className="sensors">
            <article className="sensor">
              <h3>Sentinel-2</h3>
              <p className="sensor-kind">Optical · multispectral · 10–20 m</p>
              <p>
                Sunlight reflected off the surface, in bands the eye does not
                have. Three spectral indices are computed directly from the
                pixels:
              </p>
              <dl className="indices">
                <div>
                  <dt>NDVI</dt>
                  <dd>vegetation-like response</dd>
                </div>
                <div>
                  <dt>NDWI</dt>
                  <dd>water-like response</dd>
                </div>
                <div>
                  <dt>NDBI</dt>
                  <dd>built-up / bare response, limited to 20 m</dd>
                </div>
              </dl>
              <p className="fineprint">
                These are indices, not classifications. A high NDBI is a
                built-up-like reflectance signature — not a detected building.
              </p>
            </article>
            <article className="sensor">
              <h3>Sentinel-1</h3>
              <p className="sensor-kind">SAR · radar · cloud-independent</p>
              <p>
                Radar sees through cloud and darkness. SatQuery discovers
                Sentinel-1 scenes and selects them deterministically alongside
                optical ones.
              </p>
              <p className="boundary">
                <strong>Real VV/VH imagery and backscatter.</strong> Public
                terrain-corrected Sentinel-1 RTC data is rendered as
                georeferenced grayscale, and quantitative VV/VH gamma-naught
                backscatter is measured in decibels from the provider&rsquo;s
                linear power values &mdash; averaged in linear power, then
                converted.
              </p>
              <p className="fineprint">
                The terrain correction is the data provider&rsquo;s, not
                SatQuery&rsquo;s. No radiometric calibration, speckle filtering
                or polarimetric decomposition is performed, and the product
                carries no quality mask beyond nodata. VV&minus;VH is a
                difference of measurements, not a land-cover classification.
                Access-restricted GRD imagery is reported honestly.
              </p>
            </article>
          </div>
          <Aside>Different perspectives. A clearer planet.</Aside>
        </Chapter>

        {/* ----------------------------------------- 5. questions ------- */}
        <Chapter id="questions">
          <h2>Ask it the way you&rsquo;d say it.</h2>
          <p>
            You do not need remote-sensing vocabulary. The question comes
            first; the system decides what to measure.
          </p>
          <ul className="questions">
            {QUESTIONS.map((q) => (
              <li key={q.ask}>
                <p className="question-ask">&ldquo;{q.ask}&rdquo;</p>
                <p className="question-answer">
                  <span className="question-index">{q.answers}</span>
                  <span className="question-note">{q.note}</span>
                </p>
              </li>
            ))}
          </ul>

          {/* The actual interface, not a mockup: a real run over a real
              Sentinel-2 scene. Loaded lazily and given explicit dimensions so
              it cannot shift the layout when it arrives. */}
          <figure className="product-shot">
            <img
              src="/landing/product-analysis.jpg"
              width={1400}
              height={875}
              loading="lazy"
              decoding="async"
              alt="The SatQuery workspace after a run: a Sentinel-2 true-colour
                   scene in the centre, deterministic evidence listing the scene
                   ID, acquisition, sensor, ground sample distance and cloud
                   cover below it, and a vision-language observation attributed
                   to its model in the right-hand rail."
            />
            <figcaption>
              The workspace after a real query. Every value shown is computed
              from the scene it names.
            </figcaption>
          </figure>
        </Chapter>

        {/* ------------------------------------------ 6. evidence ------- */}
        <Chapter id="evidence" align="right">
          <h2>Every answer has a trail.</h2>
          <p>
            Nothing is asserted that cannot be traced. Every number in an answer
            is checked against the evidence that produced it, and an answer that
            fails the check is withheld rather than shown.
          </p>
          <ol className="trail">
            {TRAIL.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ol>
          <figure className="export-figure">
            <figcaption>
              Export evidence → JSON
              <span className="figure-note">
                Illustration of the report&rsquo;s structure. Field names are the
                real ones; values are not shown.
              </span>
            </figcaption>
            <pre>
              <code>{`{
  "report_version": "1",
  "generated_at": "…",
  "question": "…",
  "answer": "…",
  "status": "ok | answer_withheld | …",
  "answer_validation": { "checks": […] },
  "trace":    { "plan": …, "steps": […] },
  "evidence": { "items": […] },
  "imagery_errors": […],
  "warnings": […]
}`}</code>
            </pre>
          </figure>
          <Aside>Real data. Real evidence. Nothing invented.</Aside>
        </Chapter>

        {/* --------------------------------------------- 7. close ------- */}
        <Chapter id="close" align="centre">
          <h2 className="close-title">
            Turn satellite imagery
            <br />
            into insight.
          </h2>
          <p className="lede">For a more informed, more resilient planet.</p>
          <div className="cta-row">
            <a className="cta cta-primary" href="/">
              Try SatQuery
            </a>
            <a
              className="cta cta-secondary"
              href="https://github.com/karthikbalaji1111-wq/SatQuery-AI"
              rel="noreferrer"
            >
              Learn more
            </a>
          </div>
        </Chapter>
      </main>

      <footer className="landing-footer">
        <p>
          SatQuery AI — a research prototype for Smart India Hackathon 2026,
          Problem Statement 26167.
        </p>
        <p className="credits">
          Earth imagery: NASA Visible Earth. Satellite data: Copernicus
          Sentinel-2 via Earth Search; Sentinel-1 RTC via Microsoft Planetary Computer.
        </p>
      </footer>
    </>
  );
}
