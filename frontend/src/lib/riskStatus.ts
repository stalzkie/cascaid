// Risk is an actionable threshold signal, not a bare magnitude -- it maps onto
// the status palette (good/warning/serious/critical), never a sequential ramp,
// so severity reads the same way any other status indicator does in this product.
export type RiskBand = "unscored" | "good" | "warning" | "serious" | "critical";

export interface RiskStatus {
  band: RiskBand;
  color: string;
  label: string;
}

const STATUS: Record<RiskBand, { color: string; label: string }> = {
  unscored: { color: "#898781", label: "Not yet scored" },
  good: { color: "#0ca30c", label: "Low risk" },
  warning: { color: "#fab219", label: "Elevated risk" },
  serious: { color: "#ec835a", label: "High risk" },
  critical: { color: "#d03b3b", label: "Critical risk" },
};

export function riskStatus(score: number | null): RiskStatus {
  let band: RiskBand;
  if (score === null) {
    band = "unscored";
  } else if (score < 0.3) {
    band = "good";
  } else if (score < 0.6) {
    band = "warning";
  } else if (score < 0.8) {
    band = "serious";
  } else {
    band = "critical";
  }
  return { band, ...STATUS[band] };
}
