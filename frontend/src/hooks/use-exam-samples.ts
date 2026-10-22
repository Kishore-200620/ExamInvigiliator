import { useEffect, useState } from "react";
import { fetchSampleVideos } from "@/lib/api";
import type { UploadedMedia } from "@/types/exam";

export function useExamSamples() {
  const [samples, setSamples] = useState<UploadedMedia[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    void fetchSampleVideos()
      .then((videos) => { if (active) setSamples(videos); })
      .catch((cause: unknown) => { if (active) setError(cause instanceof Error ? cause.message : "Could not load exam videos."); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, []);

  return { samples, loading, error };
}
