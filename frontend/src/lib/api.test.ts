import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fetchPipeline, fetchRuns, fetchTrackRecord, UnauthorizedError } from "./api";
import { getToken, setToken } from "./auth";

const originalFetch = globalThis.fetch;

beforeEach(() => {
  sessionStorage.clear();
  setToken("tok-123");
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe("fetchRuns", () => {
  it("GETs /runs with the bearer token and returns the run_ids list", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ run_ids: ["a", "b"] }) });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const result = await fetchRuns("http://api.local");

    expect(fetchMock).toHaveBeenCalledWith("http://api.local/runs", {
      headers: { Authorization: "Bearer tok-123" },
    });
    expect(result).toEqual(["a", "b"]);
  });

  it("throws on an error response", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: false, status: 500 }) as unknown as typeof fetch;

    await expect(fetchRuns("http://api.local")).rejects.toThrow();
  });

  it("throws UnauthorizedError and clears the token on a 401", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: false, status: 401 }) as unknown as typeof fetch;

    await expect(fetchRuns("http://api.local")).rejects.toThrow(UnauthorizedError);
    expect(getToken()).toBeNull();
  });
});

describe("fetchPipeline", () => {
  it("GETs /pipeline/{run_id} with the bearer token and returns the parsed body", async () => {
    const body = { run_id: "r1", step: 0, nodes: [], edges: [] };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => body });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const result = await fetchPipeline("http://api.local", "r1");

    expect(fetchMock).toHaveBeenCalledWith("http://api.local/pipeline/r1", {
      headers: { Authorization: "Bearer tok-123" },
    });
    expect(result).toEqual(body);
  });

  it("returns null on a 404 (no snapshot yet)", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: false, status: 404 }) as unknown as typeof fetch;

    expect(await fetchPipeline("http://api.local", "r1")).toBeNull();
  });

  it("throws on a non-404 error response", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: false, status: 500 }) as unknown as typeof fetch;

    await expect(fetchPipeline("http://api.local", "r1")).rejects.toThrow();
  });
});

describe("fetchTrackRecord", () => {
  it("GETs /track-record/{run_id} with the bearer token and returns the parsed body", async () => {
    const body = { run_id: "r1", history: [], incidents: [] };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => body });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const result = await fetchTrackRecord("http://api.local", "r1");

    expect(fetchMock).toHaveBeenCalledWith("http://api.local/track-record/r1", {
      headers: { Authorization: "Bearer tok-123" },
    });
    expect(result).toEqual(body);
  });
});
