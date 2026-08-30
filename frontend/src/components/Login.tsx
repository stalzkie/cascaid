import { useState } from "react";
import { login } from "../lib/auth";

type Props = {
  apiBaseUrl: string;
  onSuccess: () => void;
};

export function Login({ apiBaseUrl, onSuccess }: Props) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await login(apiBaseUrl, username, password);
      onSuccess();
    } catch {
      setError("Invalid username or password.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main className="app login-screen">
      <form className="card login-card" onSubmit={(e) => void handleSubmit(e)}>
        <span className="eyebrow">Cascaid</span>
        <h1>Sign in</h1>
        <label htmlFor="login-username">Username</label>
        <input id="login-username" type="text" value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" />
        <label htmlFor="login-password">Password</label>
        <input
          id="login-password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          autoComplete="current-password"
        />
        {error && <p role="alert">{error}</p>}
        <button type="submit" className="btn btn-primary" disabled={submitting}>
          Log in
        </button>
      </form>
    </main>
  );
}
