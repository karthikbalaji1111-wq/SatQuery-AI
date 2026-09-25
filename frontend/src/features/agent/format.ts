/**
 * Number formatting for the result panel - presentation only.
 *
 * API values are never rounded; only these strings are. The result card and
 * the plain-language interpretation both format through here, so the sentence
 * under a number can never state that number differently from the card.
 */

/** An index mean on the -1..+1 scale, signed, at four decimals: "+0.5204". */
export function signedIndex(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(4)}`;
}

/** Backscatter in decibels, at two decimals: "-5.44 dB". */
export function decibels(value: number): string {
  return `${value.toFixed(2)} dB`;
}

/** A whole count with thousands separators: "16,988". */
export function pixelCount(value: number): string {
  return Math.round(value).toLocaleString("en-US");
}
