import { useEffect, useRef, useState } from "react";

import { ApiError } from "../api/client";
import { fetchModelCatalog } from "../api/models";
import type { ModelOption } from "../api/types";

/** The option meaning "name no provider": the server's standard workflow. */
const STANDARD_OPTION = "";

/**
 * Which interpreter answers the next question.
 *
 * The first option - and the default - is the STANDARD workflow: the server
 * interprets supported questions deterministically, with no model and no
 * credential, so nothing here has to be configured for a query to run. Picking
 * an AI model opts the next run into AI interpretation with that provider; one
 * selection then covers planning, visual analysis and answer synthesis, since
 * the backend resolves all three roles from the same provider.
 *
 * The catalog - ids, capabilities, and whether a provider is configured -
 * comes from the backend; this component renders a choice and reports the
 * selection upward. It holds no key and no model list of its own.
 *
 * A model RETIRED by its provider is disabled for the same reason and shown
 * with the server's retirement status, so a reader learns the model exists and
 * why it cannot be picked - hiding it would invite someone to re-add it.
 *
 * Models that cannot fill the step are shown but disabled rather than hidden:
 * a reader looking for a Nemotron they know exists should see why it is not
 * selectable, not silently fail to find it. The status text is the server's,
 * so it never overstates availability - a configured model reads "Ready", not
 * "Available", because nothing here has contacted the provider.
 */
export function ModelSelector({
  value,
  onChange,
  onDefaults,
}: {
  /** The selected `model_id`, or `null` for the standard, model-free workflow. */
  value: string | null;
  onChange: (selection: { provider: string; model: string } | null) => void;
  /**
   * The deployment default, once the catalog reports it. Display provenance
   * only - it never changes what the request sends.
   */
  onDefaults?: (defaults: { provider: string; model: string }) => void;
}) {
  const [models, setModels] = useState<ModelOption[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [defaults, setDefaults] = useState<{
    provider: string;
    model: string;
  } | null>(null);

  // Held in a ref so the catalog fetch keeps its empty dependency list. Adding
  // the callback to the deps would re-run the fetch whenever a caller passed a
  // fresh inline function - one network request per render.
  const onDefaultsRef = useRef(onDefaults);
  useEffect(() => {
    onDefaultsRef.current = onDefaults;
  });

  // Whether a catalog request is in flight, so a burst of focus events cannot
  // stack requests.
  const loadingRef = useRef(false);
  // Bumped to re-read the catalog. Statuses are a snapshot: a local model's
  // "Not installed" or "Ollama not running" goes stale the moment someone
  // starts Ollama or pulls the model in another window - observed live, the
  // badge kept reading "Not installed" beside a successful local answer.
  const [generation, setGeneration] = useState(0);

  useEffect(() => {
    const refresh = () => {
      if (!loadingRef.current) setGeneration((value) => value + 1);
    };
    // A focused window is visible by definition; a visibility change only
    // counts when the tab has come back into view.
    const onVisible = () => {
      if (document.visibilityState === "visible") refresh();
    };
    window.addEventListener("focus", refresh);
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      window.removeEventListener("focus", refresh);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    loadingRef.current = true;
    fetchModelCatalog("visual", controller.signal)
      .then((catalog) => {
        // API data, so it is checked rather than trusted: a malformed catalog
        // must degrade to an honest "unavailable", not crash the workspace
        // around it.
        if (!Array.isArray(catalog?.models)) {
          setError("Model catalog was malformed");
          return;
        }
        setModels(catalog.models);
        setDefaults({
          provider: catalog.default_provider,
          model: catalog.default_model,
        });
        // Reported separately from `onChange`, which means "the user picked
        // this". The request body deliberately OMITS an untouched default so
        // the server stays the authority on what that default is - but the
        // workspace still needs to know it, to attribute a finished result to
        // the provider that actually ran it.
        onDefaultsRef.current?.({
          provider: catalog.default_provider,
          model: catalog.default_model,
        });
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return;
        setError(
          cause instanceof ApiError ? cause.message : "Model catalog unavailable",
        );
      })
      .finally(() => {
        loadingRef.current = false;
      });
    return () => controller.abort();
  }, [generation]);

  if (error !== null) {
    // Only the optional AI models are unknown; the standard workflow the next
    // question uses needs none of them.
    return (
      <span className="model-selector model-selector-error" title={error}>
        Standard · AI models unavailable
      </span>
    );
  }
  if (models === null || defaults === null) {
    return <span className="model-selector">AI …</span>;
  }

  const selected = value ?? STANDARD_OPTION;
  const current = models.find((model) => model.model_id === selected);

  return (
    <label className="model-selector">
      <span className="model-selector-label">AI</span>
      <select
        aria-label="AI provider and model"
        value={selected}
        onChange={(event) => {
          if (event.target.value === STANDARD_OPTION) {
            onChange(null);
            return;
          }
          const picked = models.find(
            (model) => model.model_id === event.target.value,
          );
          onChange(
            picked === undefined
              ? null
              : { provider: picked.provider, model: picked.model_id },
          );
        }}
      >
        <option value={STANDARD_OPTION}>Standard · no AI model</option>
        {models.map((model) => (
          <option
            key={`${model.provider}:${model.model_id}`}
            value={model.model_id}
            // Incompatible or unconfigured models stay visible but unusable,
            // with the server's own reason attached.
            disabled={
              !model.compatible ||
              !model.configured ||
              model.available === false
            }
            title={`${model.provider} · ${model.status}`}
          >
            {providerLabel(model.provider)} · {model.display_name}
            {model.status === "Ready" ? "" : ` — ${model.status}`}
          </option>
        ))}
      </select>
      {selected === STANDARD_OPTION && (
        // Always usable: the standard workflow depends on no provider.
        <span
          className="model-status"
          data-ready={true}
          title="Supported questions are interpreted deterministically - no AI model or key is used."
        >
          Ready
        </span>
      )}
      {current !== undefined && (
        <span
          className="model-status"
          // Green only when the SERVER says the model is usable. Configured,
          // compatible and not retired is not enough: a local model can be all
          // three and still be uninstalled, or its Ollama not running - and a
          // green dot beside "Not installed" states the opposite of the text.
          data-ready={current.status === "Ready"}
        >
          {current.status}
        </span>
      )}
    </label>
  );
}

const PROVIDER_LABELS: Record<string, string> = {
  gemini: "Gemini",
  nvidia: "NVIDIA",
  anthropic: "Claude",
  local: "Local",
};

function providerLabel(provider: string): string {
  return PROVIDER_LABELS[provider] ?? provider;
}
