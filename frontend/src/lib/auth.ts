const TOKEN_KEY = "cascaid_auth_token";

export function getToken(): string | null {
  return sessionStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  sessionStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  sessionStorage.removeItem(TOKEN_KEY);
}

export async function login(apiBaseUrl: string, username: string, password: string): Promise<string> {
  const response = await fetch(`${apiBaseUrl}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!response.ok) throw new Error(`login failed: ${response.status}`);
  const body = (await response.json()) as { token: string };
  setToken(body.token);
  return body.token;
}

export async function logout(apiBaseUrl: string): Promise<void> {
  const token = getToken();
  if (token) {
    try {
      await fetch(`${apiBaseUrl}/auth/logout`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });
    } catch {
      // Best-effort: the client-side token clear below is what actually matters.
    }
  }
  clearToken();
}
