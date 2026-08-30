import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { clearToken, getToken, login, logout, setToken } from "./auth";

const originalFetch = globalThis.fetch;

beforeEach(() => {
  sessionStorage.clear();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe("token storage", () => {
  it("round-trips a token through sessionStorage", () => {
    setToken("tok-123");
    expect(getToken()).toBe("tok-123");
  });

  it("returns null when no token is stored", () => {
    expect(getToken()).toBeNull();
  });

  it("clearToken removes the stored token", () => {
    setToken("tok-123");
    clearToken();
    expect(getToken()).toBeNull();
  });
});

describe("login", () => {
  it("POSTs credentials, stores the token, and returns it", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ token: "tok-123" }) });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const token = await login("http://api.local", "admin", "hunter2");

    expect(fetchMock).toHaveBeenCalledWith("http://api.local/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: "admin", password: "hunter2" }),
    });
    expect(token).toBe("tok-123");
    expect(getToken()).toBe("tok-123");
  });

  it("throws on invalid credentials and does not store a token", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: false, status: 401 }) as unknown as typeof fetch;

    await expect(login("http://api.local", "admin", "wrong")).rejects.toThrow();
    expect(getToken()).toBeNull();
  });
});

describe("logout", () => {
  it("POSTs to /auth/logout with the bearer token and clears storage", async () => {
    setToken("tok-123");
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ status: "ok" }) });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await logout("http://api.local");

    expect(fetchMock).toHaveBeenCalledWith("http://api.local/auth/logout", {
      method: "POST",
      headers: { Authorization: "Bearer tok-123" },
    });
    expect(getToken()).toBeNull();
  });

  it("clears storage even if the request fails", async () => {
    setToken("tok-123");
    globalThis.fetch = vi.fn().mockRejectedValue(new Error("network error")) as unknown as typeof fetch;

    await logout("http://api.local");

    expect(getToken()).toBeNull();
  });
});
