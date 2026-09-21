import type { ClaimStatus } from "./types";

// Servi par FastAPI (même origine) en prod; proxifié par Vite en dev.
const BASE = "";

export async function createClaim(files: File[]): Promise<{ claim_id: string }> {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  const res = await fetch(`${BASE}/api/claims`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.json().then((j) => j.detail).catch(() => res.statusText);
    throw new Error(typeof detail === "string" ? detail : res.statusText);
  }
  return res.json();
}

export async function getClaim(claimId: string): Promise<ClaimStatus> {
  const res = await fetch(`${BASE}/api/claims/${claimId}`);
  if (!res.ok) throw new Error(`Dossier introuvable (${res.status})`);
  return res.json();
}

export function imageUrl(claimId: string, filename: string, thumb = false): string {
  const suffix = thumb ? "?thumb=1" : "";
  return `${BASE}/api/claims/${claimId}/images/${encodeURIComponent(filename)}${suffix}`;
}

/** Polls the claim until it reaches a final state. Returns a cancel function. */
export function pollClaim(
  claimId: string,
  onUpdate: (status: ClaimStatus) => void,
  onError: (message: string) => void,
  intervalMs = 1200,
  timeoutMs = 5 * 60 * 1000,
): () => void {
  const started = Date.now();
  let cancelled = false;

  const tick = async () => {
    if (cancelled) return;
    try {
      const status = await getClaim(claimId);
      if (cancelled) return;
      onUpdate(status);
      if (status.status === "done" || status.status === "error") return;
    } catch (e) {
      if (cancelled) return;
      onError(e instanceof Error ? e.message : String(e));
      return;
    }
    if (Date.now() - started > timeoutMs) {
      onError("Analysis is taking longer than expected. Reload the page to check the claim state.");
      return;
    }
    window.setTimeout(tick, intervalMs);
  };

  window.setTimeout(tick, intervalMs);
  return () => {
    cancelled = true;
  };
}
