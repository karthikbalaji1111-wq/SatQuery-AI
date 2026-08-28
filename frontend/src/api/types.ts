/** Shared API response types. Mirrors the backend Pydantic models. */

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  environment: string;
}

export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
  };
}
