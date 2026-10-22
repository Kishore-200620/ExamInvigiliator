import { useCallback, useEffect, useRef, useState } from "react";
import { checkBackendHealth } from "@/lib/api";

export type BackendState = "online" | "offline" | "checking";

/**
 * Polls the FastAPI backend health endpoint every `intervalMs` milliseconds.
 * Returns the current backend connection state and the last-known version.
 */
export function useBackendStatus(intervalMs = 8000) {
  const [state, setState] = useState<BackendState>("checking");
  const [version, setVersion] = useState<string>("");
  const timer = useRef<ReturnType<typeof setInterval> | undefined>(undefined);

  const check = useCallback(async () => {
    const result = await checkBackendHealth();
    if (result.status === "healthy") {
      setState("online");
      setVersion(result.version);
    } else {
      setState("offline");
    }
  }, []);

  useEffect(() => {
    check();
    timer.current = setInterval(check, intervalMs);
    return () => clearInterval(timer.current);
  }, [check, intervalMs]);

  return { state, version, recheck: check };
}
