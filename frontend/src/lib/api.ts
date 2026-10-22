import type { AnalysisResult, HealthStatus, LiveLogEvent, LiveStudent, ReviewAction, ReviewResponse, UploadedMedia, Violation } from "@/types/exam";

const API_ORIGIN = ((import.meta.env["VITE_API_URL"] as string | undefined) ?? "").replace(/\/$/, "");
export function apiUrl(path: string) { return `${API_ORIGIN || ""}${path}`; }

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), { headers: { Accept: "application/json", ...init?.headers }, ...init });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? `Backend request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export async function checkBackendHealth(): Promise<HealthStatus> {
  try {
    const result = await request<{ data?: { status?: string; service?: string; version?: string } }>("/api/v1/health");
    return { status: result.data?.status === "healthy" ? "healthy" : "unhealthy", service: result.data?.service ?? "ExamInvigilatorAI", version: result.data?.version ?? "unknown", phase: 0 };
  } catch {
    return { status: "unhealthy", service: "ExamInvigilatorAI", version: "unknown", phase: 0 };
  }
}

export interface SampleVideo { id: string; name: string; path: string; size_mb?: number | null; exists: boolean }
export async function fetchSampleVideos(): Promise<UploadedMedia[]> {
  const videos = await request<SampleVideo[]>("/api/v1/video/samples");
  return videos.filter((video) => video.exists).map((video) => ({
    id: video.id, name: video.name, size: Math.round((video.size_mb ?? 0) * 1_000_000), type: "video" as const,
    url: apiUrl(`/api/v1/video/samples/${encodeURIComponent(video.id)}`), sample: true,
  }));
}

export interface ViolationSummary {
  total_violations: number;
  violations_by_rule: Record<string, number>;
  violations_by_session: Record<string, number>;
}
export function fetchViolationSummary(): Promise<ViolationSummary> {
  return request<ViolationSummary>("/api/v1/violations/summary");
}

export async function fetchPersistedViolations(limit = 100): Promise<Violation[]> {
  const response = await request<{ violations: RawViolation[] }>(`/api/v1/violations?limit=${limit}&offset=0`);
  return response.violations.map(toViolation);
}
export type HistoricalEvidence = {
  event_id: string;
  session_id: string;
  video_exists: boolean;
  metadata_exists: boolean;
  video_url: string | null;
  metadata: Record<string, unknown> | null;
  created_at: string | null;
};

export async function fetchHistoricalEvidence(
  limit = 500,
  offset = 0,
): Promise<HistoricalEvidence[]> {
  const data = await request<{
    items: HistoricalEvidence[];
    total: number;
    limit: number;
    offset: number;
  }>(`/api/v1/clips/history?limit=${limit}&offset=${offset}`);

  return Array.isArray(data.items) ? data.items : [];
}

export interface EvidenceStudentSnapshot {
  student_id?: number | string | null;
  state?: string | null;
  head_direction?: string | null;
  head_yaw?: number | null;
  movement_state?: string | null;
  left_wrist_status?: string | null;
  right_wrist_status?: string | null;
  active_violations?: string[];
}

export async function fetchEvidenceMetadata(eventId: string): Promise<{ student_snapshot?: EvidenceStudentSnapshot }> {
  return request<{ student_snapshot?: EvidenceStudentSnapshot }>(`/api/v1/violations/${encodeURIComponent(eventId)}/metadata`);
}

interface RawViolation {
  event_id: string; student_id?: number | null; rule_id?: string | null; event_type?: string | null;
  status?: string | null; severity?: string | null; start_time?: number | null; confidence?: number | null;
  reason?: string | null; session_id?: string | null; frame_start?: number | null; frame_end?: number | null;
  evidence_available?: boolean; evidence_video_url?: string | null;
  evidence_metadata_url?: string | null;
}
interface BackendLiveStudent {
  track_id: number; state: string; head_direction?: string; head_yaw?: number | null;
  movement_state?: string; left_wrist_status?: string; right_wrist_status?: string;
  violation_active?: boolean; active_violations?: string[];
}
interface BackendLiveEvent {
  event_id: string; student_id: number; rule_id?: string; event_type?: string; severity?: string;
  status?: string; confidence?: number; timestamp?: number; frame_id?: number; reason?: string;
}
interface LiveStatus {
  state: string;
  frame_id: number;
  timestamp?: number;
  fps?: number;
  processing_fps?: number;
  processing_latency_ms?: number;
  frames_processed?: number;
  total_frames?: number | null;
  students: BackendLiveStudent[];
  violations?: BackendLiveEvent[];
}
interface StartResponse { session_id: string; stream_url: string; video_url: string; processed_video_url: string; total_frames?: number | null }

function toViolation(item: RawViolation): Violation {
  const timestamp = item.start_time != null && Number.isFinite(item.start_time)
    ? new Date(item.start_time * 1000).toLocaleString()
    : "Not provided";
  return {
    id: item.event_id,
    type: item.event_type ?? item.rule_id ?? "Unknown event",
    studentId: item.student_id == null ? "Not assigned" : String(item.student_id),
    sessionId: item.session_id ?? "Not provided",
    confidence: item.confidence == null ? null : Number(item.confidence) * 100,
    detectionId: item.event_id,
    reason: item.reason ?? "No reason provided by the backend.",
    evidence: {
      frameNumber: item.frame_start == null ? "Not provided" : String(item.frame_start),
      frameEndNumber: item.frame_end == null ? undefined : String(item.frame_end),
      timestamp,
      videoUrl: item.evidence_video_url ? apiUrl(item.evidence_video_url) : "",
      status: item.status ?? "Not provided",
      mediaType: "video",
      available: Boolean(item.evidence_available && item.evidence_video_url),
      metadataUrl: item.evidence_metadata_url ? apiUrl(item.evidence_metadata_url) : undefined,
    },
  };
}

export interface LiveCallbacks {
  onUploadProgress: (progress: { percent: number; loaded: number; total: number }) => void;
  onStarted: (session: { sessionId: string; streamUrl: string; videoUrl: string; processedVideoUrl: string; totalFrames?: number }) => void;
  onProgress: (progress: {
    percent: number;
    stage: string;
    operation: string;
    frameId: number;
    videoFps: number;
    backendProcessingFps: number;
    processingLatencyMs: number;
    students: LiveStudent[];
    events: LiveLogEvent[];
  }) => void;
  onError: (message: string) => void;
}

function uploadVideoFile(file: File, signal: AbortSignal, onProgress: LiveCallbacks["onUploadProgress"]): Promise<SampleVideo> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file, file.name);
    xhr.open("POST", apiUrl("/api/v1/video/upload"));
    xhr.setRequestHeader("Accept", "application/json");
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress({
        percent: Math.min(100, Math.round((event.loaded / event.total) * 100)),
        loaded: event.loaded,
        total: event.total,
      });
    };
    xhr.onload = () => {
      let payload: SampleVideo & { detail?: string };
      try { payload = JSON.parse(xhr.responseText) as SampleVideo & { detail?: string }; }
      catch { reject(new Error(`Upload failed (${xhr.status})`)); return; }
      if (xhr.status >= 200 && xhr.status < 300) resolve(payload);
      else reject(new Error(payload.detail ?? `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("Video upload failed. Check the connection and try again."));
    xhr.onabort = () => reject(new DOMException("Upload cancelled", "AbortError"));
    const abort = () => xhr.abort();
    if (signal.aborted) { abort(); return; }
    signal.addEventListener("abort", abort, { once: true });
    xhr.onloadend = () => signal.removeEventListener("abort", abort);
    xhr.send(form);
  });
}

async function requestSessionStop(sessionId: string): Promise<void> {
  const sessionPath = `/api/v1/live/${encodeURIComponent(sessionId)}`;
  let stoppedRequest = false;
  for (let attempt = 0; attempt < 20; attempt += 1) {
    try {
      const response = await fetch(apiUrl(sessionPath), { method: "DELETE", keepalive: true });
      if (response.ok) { stoppedRequest = true; break; }
      if (response.status !== 404) return;
    } catch {
      // The start request may still be registering this client-assigned ID.
    }
    await new Promise<void>((resolve) => window.setTimeout(resolve, 250));
  }
  if (!stoppedRequest) return;

  // DELETE acknowledges the stop request; wait until the inference worker releases
  // the decoder/model before allowing the UI to treat this run as stopped.
  for (let attempt = 0; attempt < 40; attempt += 1) {
    try {
      const response = await fetch(apiUrl(`${sessionPath}/status`), { headers: { Accept: "application/json" } });
      if (!response.ok) return;
      const status = await response.json() as { state?: string };
      if (["stopped", "completed", "error"].includes((status.state ?? "").toLowerCase())) return;
    } catch {
      return;
    }
    await new Promise<void>((resolve) => window.setTimeout(resolve, 250));
  }
}

export function analyzeMediaWithLiveSession(media: UploadedMedia, callbacks: LiveCallbacks): () => Promise<void> {
  let sessionId: string | undefined;
  let cancelled = false;
  let timer: number | undefined;
  const controller = new AbortController();
  void (async () => {
    let videoId = media.id;
    if (!media.sample) {
      if (media.type !== "video" || !media.file) throw new Error("Choose an MP4 video file to analyze.");
      const uploaded = await uploadVideoFile(media.file, controller.signal, callbacks.onUploadProgress);
      videoId = uploaded.id;
    }
    if (cancelled) return;
    const requestedSessionId = `live_front_${crypto.randomUUID().replaceAll("-", "")}`;
    sessionId = requestedSessionId;
    const started = await request<StartResponse>("/api/v1/live/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: videoId, session_name: requestedSessionId }),
    });
    sessionId = started.session_id;
    const currentSessionId = sessionId;
    if (cancelled) {
      void requestSessionStop(currentSessionId);
      return;
    }
    const streamUrl = apiUrl(started.stream_url);
    const videoUrl = apiUrl(started.video_url);
    const processedVideoUrl = apiUrl(started.processed_video_url);
    callbacks.onStarted({ sessionId, streamUrl, videoUrl, processedVideoUrl, totalFrames: started.total_frames ?? undefined });

    try {
        let status: LiveStatus;
        do {
          status = await request<LiveStatus>(`/api/v1/live/${encodeURIComponent(currentSessionId)}/status`);
          const framesProcessed = status.frames_processed ?? status.frame_id;
          const totalFrames = status.total_frames ?? started.total_frames;
          const liveStudents: LiveStudent[] = status.students.map((student) => ({
            trackId: student.track_id,
            state: student.state,
            headDirection: student.head_direction ?? "UNKNOWN",
            headYaw: student.head_yaw ?? null,
            movementState: student.movement_state ?? "unknown",
            leftWristStatus: student.left_wrist_status ?? "UNKNOWN",
            rightWristStatus: student.right_wrist_status ?? "UNKNOWN",
            violationActive: Boolean(student.violation_active),
            activeViolations: student.active_violations ?? [],
          }));
          const liveEvents: LiveLogEvent[] = (status.violations ?? []).map((event) => ({
            id: event.event_id,
            studentId: event.student_id,
            type: event.event_type ?? event.rule_id ?? "Exam rule event",
            severity: event.severity ?? "medium",
            status: event.status ?? "active",
            confidence: event.confidence ?? 0,
            timestamp: event.timestamp ?? Date.now() / 1000,
            frameId: event.frame_id ?? framesProcessed,
            reason: event.reason ?? "No reason provided by the backend.",
          }));
          callbacks.onProgress({
            percent: totalFrames ? Math.min(99, Math.round((framesProcessed / totalFrames) * 100)) : 0,
            stage: status.state.toUpperCase(),
            operation: totalFrames
              ? `${framesProcessed.toLocaleString()} / ${totalFrames.toLocaleString()} frames processed · ${liveStudents.length} students tracked`
              : `${framesProcessed.toLocaleString()} frames processed · ${liveStudents.length} students tracked`,
            frameId: framesProcessed,
            videoFps: status.fps ?? 0,
            backendProcessingFps: status.processing_fps ?? 0,
            processingLatencyMs: status.processing_latency_ms ?? 0,
            students: liveStudents,
            events: liveEvents,
          });
          if (["completed", "error", "stopped"].includes(status.state.toLowerCase())) break;
          await new Promise<void>((resolve) => { timer = window.setTimeout(resolve, 700); });
        } while (!cancelled);
        if (cancelled) return;
        if (status.state.toLowerCase() !== "completed") throw new Error(`Live analysis ended with status: ${status.state}`);
        let payload: { violations: RawViolation[] };
        for (let attempt = 0; ; attempt += 1) {
          payload = await request<{ violations: RawViolation[] }>(`/api/v1/sessions/${encodeURIComponent(currentSessionId)}/violations?limit=1000`, { signal: controller.signal });
          if (cancelled || payload.violations.length === 0 || payload.violations.every((item) => item.evidence_available) || attempt >= 7) break;
          await new Promise<void>((resolve) => { timer = window.setTimeout(resolve, 500); });
        }
        if (cancelled) return;
        const violations = payload.violations.map(toViolation);
        callbacks.onComplete({ id: currentSessionId, sourceUrl: processedVideoUrl, detections: [], violations, totalFrames: status.frames_processed ?? status.frame_id, videoFps: status.fps ?? 0 });
    } catch (error) {
      if (!cancelled) callbacks.onError(error instanceof Error ? error.message : "Live analysis failed.");
    }
  })().catch((error) => {
    if (!cancelled) callbacks.onError(error instanceof Error ? error.message : "Could not start live analysis.");
  });
  return () => {
    cancelled = true;
    controller.abort();
    return sessionId ? requestSessionStop(sessionId) : Promise.resolve();
  };
}

export async function submitViolationReview(violationId: string, action: ReviewAction, reviewer?: string, notes?: string): Promise<ReviewResponse> {
  return request<ReviewResponse>(`/api/v1/violations/${encodeURIComponent(violationId)}/review`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, reviewer: reviewer ?? "unspecified", notes: notes ?? "" }),
  });
}

export async function getViolationReview(violationId: string): Promise<ReviewResponse | null> {
  try { return await request<ReviewResponse>(`/api/v1/violations/${encodeURIComponent(violationId)}/review`); }
  catch { return null; }
}
