import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { SarBackscatterResult } from "../../api/types";
import { SarBackscatterPanel } from "./SarBackscatterPanel";

const result: SarBackscatterResult = {
  scene_id: "S1_TEST_RTC", window_label: "single", acquired_at: "2025-01-04T00:00:00Z",
  collection: "sentinel-1-rtc",
  polarizations: ["vv", "vh"].map((pol) => ({
    polarization: pol as "vv" | "vh",
    measurements: [{ name: `${pol}_mean_db`, value: pol === "vv" ? -8.125 : -14.5, unit: "dB" }],
    valid_pixel_count: pol === "vv" ? 100 : 90,
    nonpositive_pixel_count: 2, window_pixel_count: 120,
    crs: "EPSG:32644", resolution: 10, transform: [10, 0, 0, 0, -10, 0],
  })),
  difference: { vv_mean_db: -9, vh_mean_db: -15, vv_minus_vh_mean_db: 6,
    paired_valid_pixel_count: 80, crs: "EPSG:32644", transform: null },
  measurements: [], warnings: ["Provider test warning."],
};

describe("SAR backscatter evidence", () => {
  it("renders actual polarization means and paired values without mixing pixel populations", () => {
    render(<SarBackscatterPanel result={result} />);
    expect(within(screen.getByRole("region", { name: "VV backscatter" })).getByText("-8.1250 dB")).toBeInTheDocument();
    expect(within(screen.getByRole("region", { name: "VH backscatter" })).getByText("90")).toBeInTheDocument();
    const pair = screen.getByRole("region", { name: "Paired polarization difference" });
    expect(within(pair).getByText("6.0000 dB")).toBeInTheDocument();
    expect(within(pair).getByText("80")).toBeInTheDocument();
    expect(within(pair).getByText("-9.0000 dB")).toBeInTheDocument();
    expect(screen.getByText("S1_TEST_RTC")).toBeInTheDocument();
    expect(screen.getByText("sentinel-1-rtc")).toBeInTheDocument();
    expect(screen.getByText(/no per-pixel quality mask/)).toBeInTheDocument();
    expect(screen.getByText("Provider test warning.")).toBeInTheDocument();
  });

  it("does not invent VH or a polarization difference when only VV was produced", () => {
    render(<SarBackscatterPanel result={{ ...result, polarizations: [result.polarizations[0]], difference: null }} />);
    expect(screen.queryByRole("region", { name: "VH backscatter" })).not.toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Paired polarization difference" })).not.toBeInTheDocument();
  });

  it("does not render an optical index scale or claim a zero mean for no valid pixels", () => {
    render(<SarBackscatterPanel result={{ ...result, difference: null, polarizations: [{ ...result.polarizations[0], valid_pixel_count: 0, measurements: [] }] }} />);
    expect(screen.getByText("No mean was produced for VV.")).toBeInTheDocument();
    expect(screen.queryByText("0.0000 dB")).not.toBeInTheDocument();
    expect(document.querySelector(".index-ramp")).toBeNull();
  });
});
