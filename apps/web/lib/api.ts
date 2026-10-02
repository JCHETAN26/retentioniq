// All numbers shown in the app come from the API, which serves `make evaluate` artifacts.
export const API_URL = process.env.RIQ_API_URL ?? process.env.NEXT_PUBLIC_RIQ_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { cache: "no-store", ...init });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

export async function artifact<T>(name: string): Promise<T | null> {
  try {
    return await api<T>(`/v1/artifacts/${name}`);
  } catch {
    return null;
  }
}

export type Row = Record<string, number | string | boolean | null>;

export const STRATEGY_LABEL: Record<string, string> = {
  random: "Random",
  response_model: "Highest propensity",
  two_model_logistic: "Two-model (logistic)",
  s_learner_lgbm: "S-learner",
  t_learner_lgbm: "T-learner",
  x_learner_lgbm: "X-learner",
  transformed_outcome_lgbm: "Transformed outcome",
};

export const pct = (x: number, d = 1) => `${(100 * x).toFixed(d)}%`;
export const num = (x: number, d = 0) =>
  x.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d });
