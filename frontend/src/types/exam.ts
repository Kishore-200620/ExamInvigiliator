export type ExamEventType = string;

export interface UploadedMedia {
  id: string;
  name: string;
  size: number;
  type: "video";
  url: string;
  sample?: boolean;
  file?: File;
}

export interface Detection {
  id: string;
  label: string;
  kind: "student" | "person" | "event";
  box: { left: number; top: number; width: number; height: number };
}

export interface EvidenceFrame {
  frameNumber: string;
  frameEndNumber?: string;
  timestamp: string;
  videoUrl: string;
  mediaType: "video";
  available: boolean;
  metadataUrl?: string;
  status: string;
}

export interface Violation {
  id: string;
  type: ExamEventType;
  studentId: string;
  sessionId: string;
  reason: string;
  confidence: number | null;
  detectionId: string;
  evidence: EvidenceFrame;
}

export interface AnalysisResult {
  id: string;
  sourceUrl: string;
  detections: Detection[];
  violations: Violation[];
  totalFrames?: number;
  videoFps?: number;
}

export interface LiveStudent {
  trackId: number;
  state: string;
  headDirection: string;
  headYaw: number | null;
  movementState: string;
  leftWristStatus: string;
  rightWristStatus: string;
  violationActive: boolean;
  activeViolations: string[];
}

export interface LiveLogEvent {
  id: string;
  studentId: number;
  type: string;
  severity: string;
  status: string;
  confidence: number;
  timestamp: number;
  frameId: number;
  reason: string;
}

// ---------------------------------------------------------------------------
// Backend Health & System Info
// ---------------------------------------------------------------------------

export interface HealthStatus {
  status: "healthy" | "unhealthy";
  service: string;
  version: string;
  phase: number;
}


// ---------------------------------------------------------------------------
// Violation Review
// ---------------------------------------------------------------------------

export type ReviewAction = "CONFIRMED" | "REJECTED";

export interface ReviewResponse {
  status: string;
  violation_id: string;
  review_status: ReviewAction;
  review: Record<string, unknown>;
}
