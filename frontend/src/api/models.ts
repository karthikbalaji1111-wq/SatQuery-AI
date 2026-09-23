import { apiRequest } from "./client";
import { asModelCatalog } from "./validate";
import type { ModelCatalogResponse, ModelRole } from "./types";

/**
 * The models this deployment can offer for a given step.
 *
 * Capability rules and model ids live on the server; the browser receives only
 * what it needs to render a choice, and never a credential. The response says
 * whether each model is configured and compatible - it does not claim any of
 * them is reachable, because nothing has contacted a provider.
 */
export async function fetchModelCatalog(
  role: ModelRole = "visual",
  signal?: AbortSignal,
): Promise<ModelCatalogResponse> {
  return asModelCatalog(
    await apiRequest<unknown>(`/api/v1/ai/models?role=${role}`, { signal }),
  );
}
