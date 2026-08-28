/**
 * Centralised access to build-time environment configuration.
 * Only `VITE_`-prefixed variables are available in the browser bundle.
 */

const DEFAULT_API_BASE_URL = "http://localhost:8000";

export interface AppConfig {
  apiBaseUrl: string;
}

export const config: AppConfig = {
  apiBaseUrl: (import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE_URL).replace(
    /\/$/,
    "",
  ),
};
