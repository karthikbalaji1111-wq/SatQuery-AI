import { useEffect, useState } from "react";

import { ApiError } from "../api/client";
import { fetchModelCatalog } from "../api/models";
import type { ModelOption } from "../api/types";

/**
 * Which provider and model run the AI reasoning path, from what the server
 * offers. One selection covers planning, visual analysis and answer synthesis:
 * the backend resolves all three roles from the same provider.
 *
 * The catalog - ids, capabilities, and whether a provider is configured -
 * comes from the backend; this component renders a choice and reports the
 * selection upward. It holds no key and no model list of its own.
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
}: {
  /** The selected `model_id`, or `null` for the deployment's default. */
  value: string | null;
  onChange: (selection: { provider: string; model: string } | null) => void;
}) {
  const [models, setModels] = useState<ModelOption[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [defaults, setDefaults] = useState<{
    provider: string;
    model: string;
  } | null>(null);

  useEffect(() => {
    const controller = new AbortController();
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
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return;
        setError(
          cause instanceof ApiError ? cause.message : "Model catalog unavailable",
        );
      });
    return () => controller.abort();
  }, []);

  if (error !== null) {
    return (
      <span className="model-selector model-selector-error" title={error}>
        AI unavailable
      </span>
    );
  }
  if (models === null || defaults === null) {
    return <span className="model-selector">AI …</span>;
  }

  const selected = value ?? defaults.model;
  const current = models.find((model) => model.model_id === selected);

  return (
    <label className="model-selector">
      <span className="model-selector-label">AI</span>
      <select
        aria-label="AI provider and model"
        value={selected}
        onChange={(event) => {
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
        {models.map((model) => (
          <option
            key={`${model.provider}:${model.model_id}`}
            value={model.model_id}
            // Incompatible or unconfigured models stay visible but unusable,
            // with the server's own reason attached.
            disabled={!model.compatible || !model.configured}
            title={`${model.provider} · ${model.status}`}
          >
            {providerLabel(model.provider)} · {model.display_name}
            {model.compatible && model.configured ? "" : ` — ${model.status}`}
          </option>
        ))}
      </select>
      {current !== undefined && (
        <span
          className="model-status"
          data-ready={current.compatible && current.configured}
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
};

function providerLabel(provider: string): string {
  return PROVIDER_LABELS[provider] ?? provider;
}
