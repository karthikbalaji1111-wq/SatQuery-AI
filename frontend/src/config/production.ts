/** Build-time policy for the browser's public API origin. No DNS lookup. */
export function requireProductionApiUrl(value: string | undefined): void {
  const message = "Production requires VITE_API_BASE_URL: an HTTPS URL with a public DNS hostname and no credentials, query or fragment.";
  if (!value || value !== value.trim() || /[\s\\]/.test(value)) throw new Error(message);
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new Error(message);
  }
  const host = url.hostname.toLowerCase().replace(/\.$/, "");
  if (
    url.protocol !== "https:" || url.username || url.password || url.search || url.hash ||
    !host.includes(".") || host.includes(":") || /^[\d.]+$/.test(host) ||
    /(^|\.)(localhost|local|internal)$/.test(host) || host.includes("*")
  ) throw new Error(message);
}
