import { useCallback, useEffect, useRef, useState, type RefObject } from "react";

import {

  AlertTriangle,

  Activity,

  Check,

  CheckCircle2,

  Circle,

  Download,

  Eye,

  FileText,

  Loader2,

  Maximize2,

  RefreshCw,

  Play,

  Printer,

  RotateCcw,

  ShieldCheck,

  ShieldX,

  Trash2,

  UploadCloud,

  Video,

  WifiOff,

  XCircle,

} from "lucide-react";

import { Button } from "@/components/ui/button";

import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";

import { Progress } from "@/components/ui/progress";

import {
  analyzeMediaWithLiveSession,
  fetchEvidenceMetadata,
  fetchPersistedViolations,
  fetchHistoricalEvidence,
  getViolationReview,
  submitViolationReview,
} from "@/lib/api";

import type { EvidenceStudentSnapshot } from "@/lib/api";

import { useBackendStatus } from "@/hooks/use-backend-status";

import { useExamSamples } from "@/hooks/use-exam-samples";

import type { AnalysisResult, LiveLogEvent, LiveStudent, ReviewAction, UploadedMedia, Violation } from "@/types/exam";

import { cn } from "@/lib/utils";



// Pipeline stage display names mapped from backend stage codes

const STAGE_LABELS: Record<string, string> = {

  UPLOADING: "Uploading video",

  STARTING: "Initializing",

  RUNNING: "Analyzing exam video",

  COMPLETED: "Complete",

  FAILED: "Failed",

  QUEUED: "Queued",

};



// Stage ordering for the progress list

const PIPELINE_STAGES = ["UPLOADING", "STARTING", "RUNNING", "COMPLETED"];



const MAX_UPLOAD_BYTES = 5 * 1024 * 1024 * 1024;

const formatBytes = (size: number) => {

  const units = ["B", "KB", "MB", "GB"];

  let value = size;

  let unit = 0;

  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }

  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;

};

const formatConfidence = (confidence: number | null) => confidence == null ? "Not provided" : `${confidence.toFixed(1)}%`;



export function DemoWorkspace() {

  const inputRef = useRef<HTMLInputElement>(null);

  const cleanupRef = useRef<(() => Promise<void>) | null>(null);

  const { state: backendState } = useBackendStatus();

  const { samples: sampleMedia, loading: samplesLoading, error: samplesError } = useExamSamples();



  const [media, setMedia] = useState<UploadedMedia | null>(null);

  const [phase, setPhase] = useState<"upload" | "processing" | "results">("upload");

  const [progress, setProgress] = useState(0);

  const [currentStage, setCurrentStage] = useState("QUEUED");

  const [currentOperation, setCurrentOperation] = useState("Waiting to start...");

  const [etaSeconds, setEtaSeconds] = useState<number | null>(null);

  const [streamUrl, setStreamUrl] = useState("");

  const [liveStudents, setLiveStudents] = useState<LiveStudent[]>([]);

  const [liveEvents, setLiveEvents] = useState<LiveLogEvent[]>([]);

  const [result, setResult] = useState<AnalysisResult | null>(null);

  const [selectedId, setSelectedId] = useState("");
  type EvidenceSource = "current" | "saved" | "uploaded";
  const [activeSource, setActiveSource] = useState<EvidenceSource>("current");

  const [dragging, setDragging] = useState(false);

  const [modalOpen, setModalOpen] = useState(false);

  const [error, setError] = useState("");

  const [pipelineError, setPipelineError] = useState("");

  const selected = result?.violations.find((item) => item.id === selectedId) ?? result?.violations[0];



  useEffect(() => () => { if (media && !media.sample) URL.revokeObjectURL(media.url); }, [media]);



  // Stop the active backend session on unmount.

  useEffect(() => () => { void cleanupRef.current?.(); }, []);



  if (result && !selected && phase === "results") {

    // Report completed but no violations found

    return (

      <section id="demo" className="section-shell">

        <div data-reveal className="section-heading"><p className="eyebrow">Interactive prototype 04</p><h2>Try the demo.</h2></div>

        <div className="mt-12 flex flex-col items-center gap-6 border border-border bg-surface/70 p-12 text-center">

          <CheckCircle2 className="h-16 w-16 text-success" />

          <h3 className="text-2xl font-semibold uppercase">No events detected</h3>

          <p className="max-w-md text-sm text-muted-foreground">The analysis completed and the backend returned no exam-rule events for this video.</p>

          <video src={result.sourceUrl} controls autoPlay muted playsInline preload="metadata" className="mt-2 aspect-video w-full max-w-4xl border border-border bg-black object-contain" aria-label="Backend-processed exam video" />

          <Button variant="outline" onClick={() => { setMedia(null); setPhase("upload"); setProgress(0); setPipelineError(""); }}><RotateCcw className="h-4 w-4" /> New analysis</Button>

        </div>

        <EvidenceLibrary liveEvents={liveEvents} />

      </section>

    );

  }



  const accept = (file?: File) => {

    if (!file) return;

    const validType = file.type === "video/mp4" || file.name.toLowerCase().endsWith(".mp4");

    if (!validType || file.size > MAX_UPLOAD_BYTES) { setError("Choose an MP4 video up to 5 GB. The backend analyzes video, not still images."); return; }

    setError("");

    setMedia({ id: crypto.randomUUID(), name: file.name, size: file.size, type: "video", url: URL.createObjectURL(file), file });

  };



  const clear = async () => { const stopActiveSession = cleanupRef.current; cleanupRef.current = null; if (stopActiveSession) { setCurrentOperation("Stopping backend analysis..."); await stopActiveSession(); } setMedia(null); setResult(null); setStreamUrl(""); setLiveStudents([]); setLiveEvents([]); setPhase("upload"); setProgress(0); setPipelineError(""); setCurrentStage("QUEUED"); setCurrentOperation("Waiting to start..."); setEtaSeconds(null); };



  const start = () => {

    if (!media) return;

    setPhase("processing");

    setProgress(0);

    setCurrentStage("STARTING");

    setCurrentOperation("Starting exam video analysis...");

    setPipelineError("");

    setStreamUrl("");

    setLiveStudents([]);

    setLiveEvents([]);



    const cleanup = analyzeMediaWithLiveSession(media, {

      onUploadProgress: (event) => {

        setCurrentStage("UPLOADING");

        setProgress(event.percent);

        setCurrentOperation(`Uploading ${formatBytes(event.loaded)} / ${formatBytes(event.total)}`);

      },

      onStarted: (session) => {

        setStreamUrl(session.streamUrl);

        setProgress(0);

        setCurrentStage("STARTING");

        setCurrentOperation(session.totalFrames

          ? `Video uploaded · ${session.totalFrames.toLocaleString()} frames queued for full analysis`

          : "Video uploaded · preparing full-video analysis");

      },

      onProgress: (event) => {

        setProgress(event.percent);

        setCurrentStage(event.stage);

        setCurrentOperation(event.operation);

        setLiveStudents(event.students);

        setLiveEvents(event.events.slice(0, 12));

        setEtaSeconds(null);

      },

      onComplete: (analysisResult: AnalysisResult) => {

        setProgress(100);

        setResult(analysisResult);

        setSelectedId(analysisResult.violations[0]?.id ?? "");

        window.setTimeout(() => setPhase("results"), 300);

      },

      onError: (errMsg: string) => {

        setPipelineError(errMsg);

        setCurrentStage("FAILED");

        setCurrentOperation(errMsg);

      },

    });



    cleanupRef.current = cleanup;

  };



  const downloadEvidence = async () => {

    if (!selected?.evidence.available || !selected.evidence.videoUrl) return;

    const anchor = document.createElement("a");

    anchor.href = selected.evidence.videoUrl;

    anchor.download = `exam-evidence-${selected.id}.mp4`;

    anchor.click();

  };



  const isBackendAvailable = backendState === "online";



  return <section id="demo" className="section-shell">

    <div data-reveal className="section-heading"><p className="eyebrow">Interactive prototype 04</p><h2>Try the demo.</h2><p>Choose an exam video to run the connected student tracking and exam-rule analysis pipeline.</p></div>

    <div className="mt-12 min-h-[620px] border border-border bg-surface/70 p-4 shadow-2xl md:p-7">

      <div className="mb-6 flex items-center justify-between border-b border-border pb-4">

        <div className="flex items-center gap-3">

          <span className={cn("h-2 w-2", isBackendAvailable ? "bg-success" : "bg-violation animate-pulse")} />

          <span className="text-[10px] uppercase tracking-[0.15em]">Exam Invigilation / Analysis console</span>

        </div>

        <span className="text-[9px] uppercase text-muted-foreground">{isBackendAvailable ? "Backend connected" : "Backend offline"}</span>

      </div>



      {phase === "upload" && <div className="grid gap-5 lg:grid-cols-[1fr_310px]">

        <div onDragEnter={(e)=>{e.preventDefault();setDragging(true)}} onDragOver={(e)=>e.preventDefault()} onDragLeave={()=>setDragging(false)} onDrop={(e)=>{e.preventDefault();setDragging(false);accept(e.dataTransfer.files[0])}} className={cn("grid min-h-[430px] place-items-center border border-dashed p-6 transition-colors", dragging ? "border-primary bg-primary/5" : "border-border bg-background/40")}>

            {!media ? <div className="text-center"><UploadCloud className="mx-auto h-10 w-10 text-primary"/><h3 className="mt-5 text-lg font-semibold uppercase">Drag & drop an MP4 here</h3><p className="my-4 text-xs text-muted-foreground">or</p><Button onClick={()=>inputRef.current?.click()}>Select video</Button><p className="mt-5 text-[10px] uppercase leading-5 text-muted-foreground">MP4 video · Maximum 5 GB · Full video analysis</p>{error && <p className="mt-3 text-xs text-violation">{error}</p>}</div> : <div className="w-full max-w-xl"><div className="relative aspect-video overflow-hidden border border-border bg-background"><video src={media.url} controls className="h-full w-full object-cover"/></div><div className="mt-4 grid grid-cols-[minmax(0,1fr)_auto] items-center gap-4 border border-border bg-background p-4"><div className="flex min-w-0 items-center gap-3"><Video className="h-5 w-5 shrink-0 text-primary"/><div className="min-w-0"><p className="truncate text-sm">{media.name}</p><p className="text-[10px] text-muted-foreground">{formatBytes(media.size)}</p></div></div><Button variant="ghost" size="icon" onClick={clear} aria-label="Remove video"><Trash2/></Button></div>

            {!isBackendAvailable && <div className="mt-4 flex items-center gap-2 border border-violation/40 bg-violation/5 px-4 py-3 text-[10px] text-violation"><WifiOff className="h-4 w-4 shrink-0"/> Backend is offline. Start FastAPI from the project root with <code className="mx-1 font-mono text-violation/80">uvicorn backend.app.main:app --reload --port 8000</code></div>}

            <Button size="lg" className="mt-5 w-full" onClick={start} disabled={!isBackendAvailable}>

              {isBackendAvailable ? <>Start analysis <Play className="h-4 w-4"/></> : <>Backend offline <WifiOff className="h-4 w-4"/></>}

            </Button>

          </div>}

          <input ref={inputRef} type="file" accept="video/mp4,.mp4" className="sr-only" onChange={(e)=>accept(e.target.files?.[0])}/>

        </div>

        <div><p className="mb-3 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">Backend video samples</p><div className="space-y-3">{sampleMedia.length ? sampleMedia.map((sample, index)=><button key={sample.id} onClick={()=>{setMedia(sample);setError("")}} className={cn("group grid w-full grid-cols-[96px_1fr] overflow-hidden border text-left transition-colors", media?.id===sample.id?"border-primary bg-primary/5":"border-border bg-background hover:border-primary/50")}><video src={sample.url} className="h-24 w-24 object-cover" muted preload="metadata"/><span className="self-center px-4"><span className="block text-[9px] uppercase text-primary">Sample {String(index+1).padStart(2,"0")}</span><span className="mt-1 block truncate text-xs uppercase leading-5">{sample.name}</span></span></button>) : <p className="border border-border bg-background p-4 text-xs text-muted-foreground">{samplesLoading ? "Loading sample videos…" : samplesError || "No sample videos are currently available."}</p>}</div><p className="mt-5 border-t border-border pt-5 text-[10px] leading-5 text-muted-foreground">MP4 video is analyzed by the FastAPI exam-monitoring pipeline. Progress and saved evidence come from the running backend.</p></div>

      </div>}



      {phase === "processing" && <ProcessingView media={media} progress={progress} currentStage={currentStage} currentOperation={currentOperation} etaSeconds={etaSeconds} streamUrl={streamUrl} liveStudents={liveStudents} liveEvents={liveEvents} error={pipelineError} onRetry={() => { setPipelineError(""); start(); }} onCancel={clear} />}

      {phase === "results" && result && selected && <ResultsView result={result} selected={selected} onSelect={(id)=>setSelectedId(id)} onEventReached={(id)=>{setSelectedId(id);setModalOpen(true);}} onReset={clear}/>}

    </div>

    <EvidenceLibrary liveEvents={liveEvents} />

    {phase === "results" && result && selected && <><EvidenceViewer selected={selected} onOpen={()=>setModalOpen(true)} onDownload={downloadEvidence}/><ViolationReport result={result} selected={selected}/></>}

    <Dialog open={modalOpen} onOpenChange={setModalOpen}><DialogContent className="max-h-[92vh] max-w-6xl border-border bg-background p-3 sm:p-4"><DialogHeader className="sr-only"><DialogTitle>Full evidence video</DialogTitle><DialogDescription>Full-size exam-rule evidence video</DialogDescription></DialogHeader>{selected && <EvidenceVideo violation={selected} className="max-h-[82vh]"/>}</DialogContent></Dialog>

  </section>;

}



// ---------------------------------------------------------------------------

// Processing View — live exam session state

// ---------------------------------------------------------------------------



function ProcessingView({

  media,

  progress,

  currentStage,

  currentOperation,

  etaSeconds,

  streamUrl,

  liveStudents,

  liveEvents,

  error,

  onRetry,

  onCancel,

}: {

  media: UploadedMedia | null;

  progress: number;

  currentStage: string;

  currentOperation: string;

  etaSeconds: number | null;

  streamUrl: string;

  liveStudents: LiveStudent[];

  liveEvents: LiveLogEvent[];

  error: string;

  onRetry: () => void;

  onCancel: () => void;

}) {

  const stageIndex = PIPELINE_STAGES.indexOf(currentStage);

  const liveStreamUrl = streamUrl;

  const [streamConnected, setStreamConnected] = useState(false);
  const [videoExpanded, setVideoExpanded] = useState(false);

  useEffect(() => {
    setStreamConnected(false);
    setVideoExpanded(false);
  }, [liveStreamUrl]);



  return <div className="animate-fade-in">

    <div className="mb-8 text-center">

      <p className="eyebrow justify-center">Processing / Live</p>

      <h3 className="mt-3 text-3xl font-semibold uppercase">{error ? "Analysis failed" : "Analyzing media"}</h3>

      <p className="mt-2 text-xs text-muted-foreground">{error ? error : currentOperation}</p>

    </div>



    {error ? (

      <div className="flex flex-col items-center gap-5 py-12">

        <XCircle className="h-16 w-16 text-violation" />

        <p className="max-w-md text-center text-sm text-muted-foreground">{error}</p>

        <div className="flex gap-3">

          <Button variant="outline" onClick={onCancel}><RotateCcw className="h-4 w-4"/> Start over</Button>

          <Button onClick={onRetry}><Play className="h-4 w-4"/> Retry</Button>

        </div>

      </div>

    ) : (

      <div className="grid gap-5 lg:grid-cols-[1.4fr_1fr]">

        <div className="relative overflow-hidden border border-border bg-black">
          <button
            type="button"
            className="group relative block aspect-video w-full cursor-zoom-in bg-black focus:outline-none focus:ring-2 focus:ring-primary focus:ring-inset"
            onClick={() => liveStreamUrl && setVideoExpanded(true)}
            disabled={!liveStreamUrl}
            aria-label={liveStreamUrl ? "Open live exam video in full view" : "Waiting for live exam video"}
          >
            {liveStreamUrl ? (
              <img
                src={liveStreamUrl}
                alt="Live backend annotated exam stream"
                onLoad={() => setStreamConnected(true)}
                onError={() => setStreamConnected(false)}
                className="h-full w-full object-contain"
              />
            ) : (
              <video
                src={media?.url}
                autoPlay
                muted
                loop
                playsInline
                className="h-full w-full object-contain"
              />
            )}

            {liveStreamUrl && (
              <div className="pointer-events-none absolute inset-0 flex items-end justify-end bg-gradient-to-t from-black/50 via-transparent to-transparent p-3 opacity-100 transition-opacity md:opacity-0 md:group-hover:opacity-100">
                <span className="flex items-center gap-2 border border-white/20 bg-black/70 px-3 py-2 text-[9px] font-semibold uppercase tracking-[0.12em] text-white backdrop-blur-md">
                  <Maximize2 className="h-3.5 w-3.5" />
                  Touch / click to enlarge
                </span>
              </div>
            )}

            <div className="absolute bottom-3 left-3 flex items-center gap-2 border border-border bg-background/80 px-3 py-2 backdrop-blur-md">
              <Loader2 className="h-3 w-3 animate-spin text-primary" />
              <span className="text-[9px] uppercase tracking-[0.1em] text-muted-foreground">
                {STAGE_LABELS[currentStage] ?? currentStage}
              </span>
            </div>

            {etaSeconds !== null && etaSeconds > 0 && (
              <div className="absolute bottom-3 right-3 bg-background/80 px-3 py-2 text-[9px] uppercase tracking-[0.1em] text-primary backdrop-blur-md">
                ETA {Math.ceil(etaSeconds)}s
              </div>
            )}
          </button>

          <div className="flex items-center justify-between gap-3 border-t border-border bg-background px-3 py-2">
            <p className="truncate text-[9px] uppercase tracking-[0.1em] text-muted-foreground">
              {streamConnected
                ? "Live annotated video"
                : liveStreamUrl
                  ? "Connecting to live video…"
                  : "Preparing video…"}
            </p>

            {liveStreamUrl && (
              <button
                type="button"
                onClick={() => setVideoExpanded(true)}
                className="shrink-0 text-[9px] font-semibold uppercase tracking-[0.1em] text-primary hover:underline"
              >
                Full view
              </button>
            )}
          </div>
        </div>

        <Dialog open={videoExpanded} onOpenChange={setVideoExpanded}>
          <DialogContent className="h-[92vh] w-[96vw] max-w-[1600px] border-border bg-black p-2 sm:p-3">
            <DialogHeader className="sr-only">
              <DialogTitle>Live exam video</DialogTitle>
              <DialogDescription>
                Full-size live annotated exam video from the backend.
              </DialogDescription>
            </DialogHeader>

            <div className="flex h-full min-h-0 items-center justify-center overflow-hidden bg-black">
              {liveStreamUrl ? (
                <img
                  src={liveStreamUrl}
                  alt="Full-size live backend annotated exam stream"
                  className="max-h-full max-w-full object-contain"
                />
              ) : (
                <video
                  src={media?.url}
                  autoPlay
                  muted
                  loop
                  playsInline
                  controls
                  className="max-h-full max-w-full object-contain"
                />
              )}
            </div>
          </DialogContent>
        </Dialog>

        <div className="border border-border bg-background p-5">

          {/* Pipeline stages with real backend status */}

          <div className="space-y-4">

            {PIPELINE_STAGES.map((stage, index) => {

              const isActive = stage === currentStage;

              const isComplete = stageIndex > index || (currentStage === "COMPLETED" || (progress >= 100));

              return (

                <div key={stage} className="flex items-center justify-between text-xs">

                  <span className={cn(isComplete || isActive ? "text-foreground" : "text-muted-foreground")}>

                    {STAGE_LABELS[stage] ?? stage}

                  </span>

                  {isComplete ? (

                    <Check className="h-4 w-4 text-success" />

                  ) : (

                    <Circle className={cn("h-3 w-3", isActive ? "animate-pulse text-primary" : "text-border")} />

                  )}

                </div>

              );

            })}

          </div>



          <Progress value={progress} className="mt-8" />

          <div className="mt-3 flex justify-between text-[10px] uppercase tracking-[0.1em]">

            <span className="truncate max-w-[180px] text-muted-foreground">{currentOperation.length > 50 ? currentOperation.slice(0, 48) + "…" : currentOperation}</span>

            <span className="text-primary">{progress}%</span>

          </div>



          {/* Live stream indicator */}

          {streamUrl && (

            <div className="mt-5 flex items-center gap-2 border-t border-border pt-4 text-[10px] text-muted-foreground">

              <span className={cn("h-2 w-2 rounded-full", streamConnected ? "animate-pulse bg-success" : "animate-pulse bg-primary")} />

              <span className="uppercase tracking-[0.1em]">{streamConnected ? "Backend annotated stream connected" : "Connecting to annotated stream…"}</span>

            </div>

          )}

        </div>

      </div>

    )}

    {!error && <div className="mt-5 grid gap-5 xl:grid-cols-2">

      <section className="border border-border bg-background p-5" aria-live="polite">

        <div className="mb-4 flex items-center justify-between">

          <div><p className="eyebrow">Live detections</p><h4 className="mt-2 text-lg font-semibold uppercase">Students in frame</h4></div>

          <span className="border border-primary/40 px-2 py-1 text-[10px] uppercase text-primary">{liveStudents.length} tracked</span>

        </div>

        {liveStudents.length ? <div className="max-h-64 space-y-2 overflow-y-auto">{liveStudents.map((student) => <article key={student.trackId} className="grid grid-cols-[1fr_auto] gap-3 border border-border px-3 py-3 text-xs">

          <div><p className="font-semibold uppercase">Student #{student.trackId} <span className="ml-2 text-[9px] font-normal text-muted-foreground">{student.state}</span></p>

            <p className="mt-2 text-[10px] leading-5 text-muted-foreground">Head {student.headDirection}{student.headYaw !== null ? ` (${student.headYaw.toFixed(1)}°)` : ""} · Movement {student.movementState}<br/>Wrists L {student.leftWristStatus} / R {student.rightWristStatus}</p>

            {student.activeViolations.length > 0 && <p className="mt-2 text-[10px] font-semibold uppercase text-violation">{student.activeViolations.join(" · ")}</p>}

          </div>

          <span className={cn("self-start border px-2 py-1 text-[9px] uppercase", student.violationActive ? "border-violation/50 text-violation" : "border-success/40 text-success")}>{student.violationActive ? "Review" : "Clear"}</span>

        </article>)}</div> : <div className="grid min-h-28 place-items-center border border-dashed border-border text-xs text-muted-foreground">Waiting for student tracks from the detector…</div>}

        <p className="mt-3 text-[9px] leading-4 text-muted-foreground">Bounding boxes, head direction, wrist markers, and rule banners are drawn into the live annotated stream.</p>

      </section>

      <section className="border border-border bg-background p-5" aria-live="polite">

        <div className="mb-4 flex items-center justify-between">

          <div><p className="eyebrow">Live rule log</p><h4 className="mt-2 text-lg font-semibold uppercase">Detection events</h4></div>

          <Activity className="h-4 w-4 text-primary" />

        </div>

        {liveEvents.length ? <div className="max-h-64 space-y-2 overflow-y-auto">{liveEvents.map((event) => <article key={event.id} className="border border-violation/30 bg-violation/5 px-3 py-3">

          <div className="flex items-start justify-between gap-3"><div><p className="text-xs font-semibold uppercase text-violation">{event.type.replaceAll("_", " ")}</p><p className="mt-1 text-[10px] text-muted-foreground">Student #{event.studentId} · Frame {event.frameId.toLocaleString()}</p></div><span className="shrink-0 text-[9px] text-muted-foreground">{new Date(event.timestamp * 1000).toLocaleTimeString()}</span></div>

          <p className="mt-2 text-[10px] leading-4 text-muted-foreground">{event.reason}</p>

        </article>)}</div> : <div className="grid min-h-28 place-items-center border border-dashed border-border text-xs text-muted-foreground">No rule events reported yet. New backend events will appear here.</div>}

        <div className="mt-4 flex justify-end"><Button size="sm" variant="outline" onClick={onCancel}><RotateCcw className="h-3 w-3"/> Stop analysis</Button></div>

      </section>

    </div>}

  </div>;

}



function EvidenceLibrary({ liveEvents }: { liveEvents: LiveLogEvent[] }) {
  const [items, setItems] = useState<Violation[]>([]);
  const [historyItems, setHistoryItems] = useState<import("@/lib/api").HistoricalEvidence[]>([]);
  const [selectedId, setSelectedId] = useState("");
  type EvidenceSource = "current" | "saved" | "uploaded";
  const [activeSource, setActiveSource] = useState<EvidenceSource>("current");
  const [selectedSavedSessionId, setSelectedSavedSessionId] = useState<string | null>(null);
  const [studentSnapshot, setStudentSnapshot] = useState<EvidenceStudentSnapshot | null>(null);
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [detailsError, setDetailsError] = useState("");
  const detailsRequestId = useRef(0);
  const refreshInFlight = useRef(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const liveEventKey = liveEvents.map((event) => event.id).join("|");

  const refresh = useCallback(async () => {
    if (refreshInFlight.current) return;
    refreshInFlight.current = true;

    try {
      const [events, history] = await Promise.all([
        fetchPersistedViolations(100),
        fetchHistoricalEvidence(500),
      ]);
      setItems(events);
      setHistoryItems(history);
      setError("");
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not load saved evidence from the backend."
      );
    } finally {
      refreshInFlight.current = false;
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 5000);
    const onVisible = () => {
      if (document.visibilityState === "visible") void refresh();
    };

    window.addEventListener("focus", onVisible);
    document.addEventListener("visibilitychange", onVisible);

    return () => {
      window.clearInterval(timer);
      window.removeEventListener("focus", onVisible);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [refresh]);

  // Existing persisted violations take precedence over disk-only history.
  const knownEventIds = new Set(items.map((event) => event.id));

  const liveOnlyItems: Violation[] = liveEvents
    .filter((event) => {
      if (knownEventIds.has(event.id)) return false;
      knownEventIds.add(event.id);
      return true;
    })
    .map((event) => ({
      id: event.id,
      type: event.type,
      studentId: String(event.studentId),
      sessionId: "Current session",
      reason: event.reason,
      confidence: event.confidence == null ? null : Number(event.confidence) * 100,
      detectionId: event.id,
      evidence: {
        frameNumber: String(event.frameId),
        timestamp: Number.isFinite(event.timestamp)
          ? new Date(event.timestamp * 1000).toLocaleString()
          : "Just detected",
        videoUrl: "",
        mediaType: "video" as const,
        available: false,
        status: "Saving evidence",
      },
    }));

  const historicalOnlyItems: Violation[] = historyItems
    .filter((history) => !knownEventIds.has(history.event_id))
    .map((history) => {
      const metadata = history.metadata ?? {};
      const readString = (value: unknown, fallback: string) =>
        typeof value === "string" || typeof value === "number"
          ? String(value)
          : fallback;

      const type = readString(
        metadata.event_type ?? metadata.rule_id,
        "HISTORICAL_EVIDENCE"
      );

      const videoUrl = history.video_url
      ? `${(import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "")}${history.video_url}`
      : "";

      return {
        id: history.event_id,
        type,
        studentId: readString(metadata.student_id, "Unknown"),
        sessionId: history.session_id,
        reason: readString(metadata.reason, "Previously saved evidence"),
        confidence: null,
        detectionId: history.event_id,
        evidence: {
          frameNumber: readString(metadata.frame_number, "—"),
          timestamp: history.created_at
            ? new Date(history.created_at).toLocaleString()
            : "Historical record",
          videoUrl,
          mediaType: "video" as const,
          available: history.video_exists && Boolean(videoUrl),
          status: history.video_exists ? "Clip saved" : "No clip available",
          metadataUrl: history.metadata_exists ? "historical" : "",
        },
      } as Violation;
    });

  const currentSessionItems = [
  ...liveOnlyItems,
  ...items.filter((event) =>
    liveEvents.some((liveEvent) => liveEvent.id === event.id)
  ),
];


const savedEvidenceItems = [
  ...items.filter(
    (event) => !liveEvents.some((liveEvent) => liveEvent.id === event.id)
  ),
  ...historicalOnlyItems,
];

// Sort events newest to oldest.
const timestampValue = (event: Violation) => {
  const value = Date.parse(event.evidence.timestamp);
  return Number.isFinite(value) ? value : 0;
};

savedEvidenceItems.sort(
  (a, b) => timestampValue(b) - timestampValue(a)
);

// Build one folder for each saved session.
const savedFolders = Array.from(
  savedEvidenceItems.reduce((folders, event) => {
    const sessionId = event.sessionId || "unknown-session";
    const folder = folders.get(sessionId) ?? [];
    folder.push(event);
    folders.set(sessionId, folder);
    return folders;
  }, new Map<string, Violation[]>())
).map(([sessionId, events]) => ({
  sessionId,
  events,
  latestTimestamp: Math.max(...events.map(timestampValue)),
})).sort((a, b) => b.latestTimestamp - a.latestTimestamp);



const visibleItems =
  activeSource === "current"
    ? currentSessionItems
    : activeSource === "saved"
      ? selectedSavedSessionId
        ? savedFolders.find(
            (folder) => folder.sessionId === selectedSavedSessionId
          )?.events ?? []
        : []
      : [];

const selected = visibleItems.find((event) => event.id === selectedId);

const showSavedFolders =
  activeSource === "saved" && selectedSavedSessionId === null;

  useEffect(() => {
    if (liveEventKey) void refresh();
  }, [liveEventKey, refresh]);

  const openEvidence = async (event: Violation) => {
    const requestId = ++detailsRequestId.current;
    setSelectedId(event.id);
    setStudentSnapshot(null);
    setDetailsError("");

    // Disk-only historical entries may not have a database metadata endpoint.
    const isHistoricalOnly = historyItems.some(
      (history) =>
        history.event_id === event.id &&
        !items.some((saved) => saved.id === event.id)
    );

    if (isHistoricalOnly) {
      setDetailsLoading(false);
      return;
    }

    setDetailsLoading(true);
    try {
      const metadata = await fetchEvidenceMetadata(event.id);
      if (requestId === detailsRequestId.current) {
        setStudentSnapshot(metadata.student_snapshot ?? null);
      }
    } catch (err) {
      if (requestId === detailsRequestId.current) {
        setDetailsError(
          err instanceof Error
            ? err.message
            : "Could not load saved detection details."
        );
      }
    } finally {
      if (requestId === detailsRequestId.current) setDetailsLoading(false);
    }
  };

  const friendly = (value: string | null | undefined, fallback = "UNKNOWN") =>
    value ? value.replaceAll("_", " ").toUpperCase() : fallback;

  return (
    <section id="evidence-library" className="mt-16 border border-border bg-surface/60 p-4 md:p-6">
      <div className="mb-5 flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="eyebrow">Live detections + saved records</p>
          <h3 className="mt-2 text-2xl font-semibold uppercase">Evidence library</h3>
          <p className="mt-2 max-w-2xl text-xs leading-5 text-muted-foreground">
            New detections appear as soon as the backend reports them. Saved
            records and historical evidence remain available across sessions.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => void refresh()} disabled={loading}>
          <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} />
          Refresh
        </Button>
      </div>
      <div className="mb-4 flex flex-wrap gap-2" role="tablist" aria-label="Evidence source">
  {([
    ["current", "Current Session"],
    ["saved", "Saved Evidence"],
    ["uploaded", "Uploaded Exam Video"],
  ] as const).map(([source, label]) => (
    <Button
      key={source}
      type="button"
      size="sm"
      variant={activeSource === source ? "default" : "outline"}
      role="tab"
      aria-selected={activeSource === source}
      
      onClick={() => {
        setActiveSource(source);
        setSelectedId("");
        setSelectedSavedSessionId(null);
        setStudentSnapshot(null);
        setDetailsError("");
      }}

    >
      {label}
    </Button>
  ))}
</div>

{showSavedFolders && (
  <div className="space-y-2">
    <div className="mb-3 flex items-center justify-between">
      <h4 className="text-xs font-semibold uppercase tracking-wider">
        Exam session folders
      </h4>
      <span className="text-xs text-muted-foreground">
        {savedFolders.length} folders
      </span>
    </div>

    {savedFolders.length === 0 ? (
      <p className="border border-dashed border-border p-5 text-xs text-muted-foreground">
        No saved evidence folders found yet.
      </p>
    ) : (
      savedFolders.map((folder) => (
        <button
          key={folder.sessionId}
          type="button"
          onClick={() => {
            setSelectedSavedSessionId(folder.sessionId);
            setSelectedId("");
            setStudentSnapshot(null);
            setDetailsError("");
          }}
          className="flex w-full items-center justify-between gap-4 border border-border bg-background p-4 text-left transition-colors hover:border-primary"
        >
          <span className="min-w-0">
            <span className="block truncate text-sm font-semibold">
              📁 {folder.sessionId}
            </span>
            <span className="mt-1 block text-xs text-muted-foreground">
              {folder.events.length} evidence event(s)
            </span>
          </span>
          <span className="shrink-0 text-xs text-muted-foreground">
            {folder.latestTimestamp
              ? new Date(folder.latestTimestamp).toLocaleString()
              : "Date unavailable"}
            {" →"}
          </span>
        </button>
      ))
    )}
  </div>
)}

{activeSource === "saved" && selectedSavedSessionId && (
  <Button
    variant="outline"
    size="sm"
    onClick={() => {
      setSelectedSavedSessionId(null);
      setSelectedId("");
      setStudentSnapshot(null);
      setDetailsError("");
    }}
  >
    ← Back to folders
  </Button>
)}

      {error ? (
        <p className="border border-violation/40 bg-violation/5 p-4 text-xs text-violation">
          {error}
        </p>
      ) : loading && !visibleItems.length ? (
        <p className="border border-border p-5 text-xs text-muted-foreground">
          Loading saved events and evidence…
        </p>
      ) : !visibleItems.length ? (
        <p className="border border-dashed border-border p-5 text-xs text-muted-foreground">
          No malpractice has been detected or saved yet. New backend detections
          will appear here automatically.
        </p>
      ) : (
        <div className="grid gap-4 lg:grid-cols-[minmax(260px,.75fr)_minmax(0,1.25fr)]">
          <div className="max-h-[440px] space-y-2 overflow-y-auto pr-1">
            {visibleItems.map((event) => (
              <article
                key={event.id}
                className={cn(
                  "border p-3 transition-colors",
                  selected?.id === event.id
                    ? "border-primary bg-primary/5"
                    : "border-border bg-background"
                )}
              >
                <div className="flex items-start justify-between gap-3">
                  <span className="text-xs font-semibold uppercase">
                    Student #{event.studentId}
                  </span>
                  <span
                    className={cn(
                      "shrink-0 text-[9px] uppercase",
                      event.evidence.available
                        ? "text-success"
                        : "text-muted-foreground"
                    )}
                  >
                    {event.evidence.available
                      ? "Clip saved"
                      : event.evidence.status === "Saving evidence"
                        ? "Saving evidence"
                        : "No clip yet"}
                  </span>
                </div>
                <p className="mt-1 text-[10px] text-muted-foreground">
                  {event.evidence.timestamp} · {event.type.replaceAll("_", " ")}
                </p>
                <Button
                  className="mt-3"
                  size="sm"
                  variant={selected?.id === event.id ? "default" : "outline"}
                  disabled={!event.evidence.metadataUrl}
                  onClick={() => void openEvidence(event)}
                >
                  <Eye className="h-3.5 w-3.5" /> View evidence
                </Button>
              </article>
            ))}
          </div>

          {selected && (
            <div className="min-w-0 border border-border bg-background">
              <div className="border-b border-border p-4">
                <p className="text-sm font-semibold uppercase">
                  Student #{studentSnapshot?.student_id ?? selected.studentId}{" "}
                  {studentSnapshot?.state
                    ? friendly(studentSnapshot.state, "").toLowerCase()
                    : "details"}
                </p>
                <p className="mt-2 text-xs leading-6 text-muted-foreground">
                  Head {friendly(studentSnapshot?.head_direction)} (
                  {typeof studentSnapshot?.head_yaw === "number"
                    ? studentSnapshot.head_yaw.toFixed(1)
                    : "0.0"}°) · Movement{" "}
                  {friendly(studentSnapshot?.movement_state, "UNAVAILABLE").toLowerCase()}
                  <br />
                  Wrists L {friendly(studentSnapshot?.left_wrist_status)} / R{" "}
                  {friendly(studentSnapshot?.right_wrist_status)}
                </p>
                {detailsLoading && (
                  <p className="mt-2 text-[10px] text-muted-foreground">
                    Loading saved detection details…
                  </p>
                )}
                {detailsError && (
                  <p className="mt-2 text-[10px] text-violation">{detailsError}</p>
                )}
                {studentSnapshot && (
                  <p className="mt-2 text-[10px] font-semibold uppercase leading-5 text-violation">
                    {studentSnapshot.active_violations?.length
                      ? studentSnapshot.active_violations.join(" · ")
                      : selected.type}
                  </p>
                )}
                {!detailsLoading && !detailsError && !studentSnapshot && (
                  <p className="mt-2 text-[10px] text-muted-foreground">
                    Detailed tracking data was not saved for this record. Event:{" "}
                    {selected.type.replaceAll("_", " ")}.
                  </p>
                )}
                <p className="mt-2 text-[9px] text-muted-foreground">
                  Event {selected.id} · Frame {selected.evidence.frameNumber} ·{" "}
                  {selected.evidence.timestamp}
                </p>
              </div>

              <div className="grid min-h-[230px] place-items-center bg-black">
                {selected.evidence.available && selected.evidence.videoUrl ? (
                  <video
                    key={`${selected.id}:${selected.evidence.videoUrl}`}
                    src={selected.evidence.videoUrl}
                    controls
                    playsInline
                    preload="metadata"
                    onError={(event) => {
                      const video = event.currentTarget;
                      console.error("Evidence video failed to load", {
                        src: video.currentSrc || video.src,
                        errorCode: video.error?.code,
                        errorMessage: video.error?.message,
                      });
                    }}
                    className="max-h-[430px] w-full object-contain"
                    aria-label={`Evidence clip for ${selected.type}`}
                  />
                ) : (
                  <p className="px-6 py-12 text-center text-xs text-muted-foreground">
                    This event is saved, but its evidence clip is not available yet.
                  </p>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}


// ---------------------------------------------------------------------------

// Results View

// ---------------------------------------------------------------------------



function ResultsView({ result, selected, onSelect, onEventReached, onReset }: { result: AnalysisResult; selected: Violation; onSelect: (id: string) => void; onEventReached: (id: string) => void; onReset: () => void }) {

  const videoRef = useRef<HTMLVideoElement>(null);

  const lastTriggeredEvent = useRef("");

  const [playhead, setPlayhead] = useState(0);

  const [duration, setDuration] = useState(0);

  const eventSeconds = useCallback((violation: Violation) => {

    const frame = Number(violation.evidence.frameNumber);

    return Number.isFinite(frame) && result.videoFps && result.videoFps > 0 ? frame / result.videoFps : null;

  }, [result.videoFps]);

  useEffect(() => {

    const activeEvent = result.violations.find((violation) => {

      const start = eventSeconds(violation);

      if (start == null) return false;

      const endFrame = Number(violation.evidence.frameEndNumber);

      const end = Number.isFinite(endFrame) && result.videoFps ? endFrame / result.videoFps : start + 2;

      return playhead >= start && playhead <= Math.max(start + 0.5, end);

    });

    if (!activeEvent) {

      lastTriggeredEvent.current = "";

      return;

    }

    if (activeEvent.id !== lastTriggeredEvent.current) {

      lastTriggeredEvent.current = activeEvent.id;

      onSelect(activeEvent.id);

      if (activeEvent.evidence.available) {

        videoRef.current?.pause();

        onEventReached(activeEvent.id);

      }

    }

  }, [eventSeconds, onEventReached, onSelect, playhead, result.violations, result.videoFps]);



  const seekToEvent = (violation: Violation) => {

    const time = eventSeconds(violation);

    onSelect(violation.id);

    if (time != null && videoRef.current) {

      videoRef.current.currentTime = time;

      void videoRef.current.play();

    }

  };



  return <div className="animate-fade-in">

    <div className="mb-6 grid grid-cols-[minmax(0,1fr)_auto] items-center gap-4">

      <div>

        <p className="eyebrow">Analysis complete</p>

        <h3 className="mt-2 text-2xl font-semibold uppercase md:text-4xl">Exam event results</h3>

        {result.totalFrames !== undefined && (

          <p className="mt-1 text-[10px] text-muted-foreground">

            {result.totalFrames} frames processed · {result.violations.length} persisted events

          </p>

        )}

      </div>

      <Button variant="outline" size="sm" onClick={onReset}><RotateCcw/> New analysis</Button>

    </div>

    <div className="mb-5 inline-flex items-center gap-2 border border-violation/50 bg-violation/10 px-3 py-2 text-[10px] font-semibold uppercase text-violation"><AlertTriangle className="h-4 w-4"/>{result.violations.length} exam-rule events</div>

    <div className="grid gap-5 xl:grid-cols-[minmax(0,1.55fr)_minmax(300px,.65fr)]">

      <div>

        <SessionVideo result={result} selected={selected} videoRef={videoRef} playhead={playhead} duration={duration} onPlayheadChange={setPlayhead} onDurationChange={setDuration} onSeekToEvent={seekToEvent}/>

        {result.violations.length > 1 && <div className="mt-3 grid grid-cols-4 gap-2">{result.violations.map((violation,index)=><button key={violation.id} onClick={()=>onSelect(violation.id)} className={cn("relative aspect-video overflow-hidden border",selected.id===violation.id?"border-primary":"border-border opacity-60 hover:opacity-100")}><EvidenceMedia violation={violation} preview/><span className="absolute bottom-1 right-1 bg-background/80 px-1 text-[8px]">0{index+1}</span></button>)}</div>}

      </div>

      <div className="border border-border bg-background p-4">

        <p className="mb-4 text-[10px] font-semibold uppercase tracking-[.14em] text-muted-foreground">Exam-rule events</p>

        <div className="space-y-2">{result.violations.map((violation)=><button key={violation.id} onClick={()=>seekToEvent(violation)} className={cn("w-full border p-4 text-left transition-colors",selected.id===violation.id?"border-violation bg-violation/5":"border-border hover:border-muted-foreground")}><div className="flex items-start gap-3"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-violation"/><div><p className="text-xs font-semibold uppercase">{violation.type}</p><p className="mt-2 text-[10px] leading-5 text-muted-foreground">Student #{violation.studentId} · Confidence {formatConfidence(violation.confidence)}<br/>Frame {violation.evidence.frameNumber} · {violation.evidence.timestamp}</p></div></div></button>)}</div>

      </div>

    </div>

  </div>;

}



// ---------------------------------------------------------------------------

// Detection Image (annotated scene overlay)

// ---------------------------------------------------------------------------



function SessionVideo({ result, selected, videoRef, playhead, duration, onPlayheadChange, onDurationChange, onSeekToEvent }: {

  result: AnalysisResult;

  selected: Violation;

  videoRef: RefObject<HTMLVideoElement | null>;

  playhead: number;

  duration: number;

  onPlayheadChange: (time: number) => void;

  onDurationChange: (duration: number) => void;

  onSeekToEvent: (violation: Violation) => void;

}) {

  const eventTime = (violation: Violation) => {

    const frame = Number(violation.evidence.frameNumber);

    return Number.isFinite(frame) && result.videoFps && result.videoFps > 0 ? frame / result.videoFps : null;

  };



  return <div>

    <div className="relative aspect-[16/10] overflow-hidden border border-border bg-background">

      <video ref={videoRef} src={result.sourceUrl} controls autoPlay muted playsInline preload="metadata" onTimeUpdate={(event) => onPlayheadChange(event.currentTarget.currentTime)} onLoadedMetadata={(event) => onDurationChange(event.currentTarget.duration)} className="h-full w-full object-contain" aria-label="Backend-processed annotated exam video"/>

      {result.detections.map((detection)=><div key={detection.id} style={{left:`${detection.box.left}%`,top:`${detection.box.top}%`,width:`${detection.box.width}%`,height:`${detection.box.height}%`}} className={cn("absolute border transition-all duration-300",detection.kind==="event"?"border-violation":"border-primary",detection.id===selected.detectionId?"z-20 border-2 shadow-[0_0_18px_color-mix(in_oklab,var(--violation)_55%,transparent)]":"opacity-70")}><span className={cn("absolute -top-5 left-[-1px] whitespace-nowrap px-1.5 py-1 text-[8px] font-bold uppercase",detection.kind==="event"?"bg-violation text-violation-foreground":"bg-primary text-primary-foreground")}>{detection.label}</span></div>)}

      <div className="absolute bottom-3 left-3 border border-border bg-background/70 px-3 py-2 backdrop-blur-md">

        <p className="text-[8px] uppercase text-muted-foreground">Report {result.id} / Frame {selected.evidence.frameNumber} · {playhead.toFixed(1)}s</p>

      </div>

    </div>

    <div className="relative mt-2 h-8 border-y border-border bg-background" aria-label="Malpractice event timeline">

      <div className="absolute inset-x-2 top-1/2 h-px -translate-y-1/2 bg-muted-foreground/40" />

      {result.violations.map((violation) => {

        const time = eventTime(violation);

        if (time == null) return null;

        const position = duration > 0 ? Math.min(100, Math.max(0, time / duration * 100)) : 0;

        return <button key={violation.id} type="button" title={`${violation.type.replaceAll("_", " ")} · ${time.toFixed(1)}s`} onClick={() => onSeekToEvent(violation)} style={{ left: `${position}%` }} className="absolute top-1/2 z-10 h-4 w-2 -translate-x-1/2 -translate-y-1/2 rounded-sm bg-violation ring-2 ring-background" aria-label={`Seek to ${violation.type} at ${time.toFixed(1)} seconds`} />;

      })}

      {duration > 0 && <div className="absolute inset-x-2 -bottom-5 flex justify-between text-[8px] text-muted-foreground"><span>{playhead.toFixed(1)}s</span><span>{duration.toFixed(1)}s</span></div>}

    </div>

    {result.violations.length > 0 && <p className="mt-6 text-[9px] uppercase tracking-wide text-muted-foreground">Red timeline markers are malpractice events. Playback opens the matching evidence clip when it reaches an event.</p>}

  </div>;

}



// ---------------------------------------------------------------------------

// Evidence Video

// ---------------------------------------------------------------------------



function EvidenceVideo({ violation, className }: { violation: Violation; className?: string }) {

  return <div className={cn("relative aspect-[4/3] overflow-hidden bg-background", className)}>

    <EvidenceMedia violation={violation}/>

    <div className="absolute bottom-3 left-3 bg-background/80 px-3 py-2 text-[8px] uppercase backdrop-blur-md">Frame {violation.evidence.frameNumber} / {violation.evidence.timestamp}</div>

  </div>;

}



function EvidenceMedia({ violation, preview = false }: { violation: Violation; preview?: boolean }) {

  return violation.evidence.available && violation.evidence.videoUrl

    ? <video src={violation.evidence.videoUrl} controls={!preview} muted={preview} autoPlay={!preview} loop={!preview} playsInline className="h-full w-full object-contain" aria-label={`Evidence for ${violation.type}`} />

    : <div className="grid h-full w-full place-items-center bg-background p-4 text-center text-xs text-muted-foreground">Evidence video is not available for this event.</div>;

}



// ---------------------------------------------------------------------------

// Evidence Viewer with violation review buttons

// ---------------------------------------------------------------------------



function EvidenceViewer({ selected, onOpen, onDownload }: { selected: Violation; onOpen: () => void; onDownload: () => void }) {

  const [reviewState, setReviewState] = useState<Record<string, "CONFIRMED" | "REJECTED" | "pending">>({});



  useEffect(() => {

    let active = true;

    setReviewState((state) => ({ ...state, [selected.id]: "pending" }));

    void getViolationReview(selected.id).then((review) => {

      if (active) setReviewState((state) => {

        const next = { ...state };

        if (review) next[selected.id] = review.review_status;

        else delete next[selected.id];

        return next;

      });

    });

    return () => { active = false; };

  }, [selected.id]);



  const handleReview = async (action: ReviewAction) => {

    try {

      setReviewState((s) => ({ ...s, [selected.id]: "pending" }));

      await submitViolationReview(selected.id, action);

      setReviewState((s) => ({ ...s, [selected.id]: action }));

    } catch {

      setReviewState((s) => { const next = { ...s }; delete next[selected.id]; return next; });

    }

  };



  const currentReview = reviewState[selected.id];

  const rows = [

    ["Violation", selected.type],

    ["Student track", selected.studentId],

    ["Confidence", formatConfidence(selected.confidence)],

    ["Reason", selected.reason],

    ["Session", selected.sessionId],

    ["Frame number", selected.evidence.frameNumber],

    ["Timestamp", selected.evidence.timestamp],

    ["Status", currentReview === "CONFIRMED" ? "CONFIRMED" : currentReview === "REJECTED" ? "REJECTED" : selected.evidence.status],

  ];



  return <section data-reveal className="mt-24">

    <div className="section-heading"><p className="eyebrow">Evidence review 07</p><h2>Evidence video.</h2><p>Evidence saved by the backend when available for the selected event.</p></div>

    <div className="mt-10 grid border border-border bg-surface lg:grid-cols-[1.3fr_.7fr]">

      <EvidenceVideo violation={selected}/>

      <div className="border-t border-border p-6 lg:border-l lg:border-t-0">

        {rows.map(([label, value]) => <div key={label} className="grid grid-cols-2 border-b border-border py-3 text-xs"><span className="text-muted-foreground">{label}</span><span className={label === "Status" ? (value === "CONFIRMED" ? "text-success" : value === "REJECTED" ? "text-violation" : "text-muted-foreground") : "text-foreground"}>{value}</span></div>)}



        {/* Officer Review Buttons */}

        <div className="mt-5 border-t border-border pt-4">

          <p className="mb-3 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">Officer review</p>

          <div className="flex gap-2">

            <Button size="sm" variant={currentReview === "CONFIRMED" ? "default" : "outline"} className={cn("flex-1", currentReview === "CONFIRMED" && "bg-success text-black hover:bg-success/90")} onClick={() => handleReview("CONFIRMED")} disabled={currentReview === "pending"}>

              <ShieldCheck className="h-3 w-3" /> Confirm

            </Button>

            <Button size="sm" variant={currentReview === "REJECTED" ? "destructive" : "outline"} className="flex-1" onClick={() => handleReview("REJECTED")} disabled={currentReview === "pending"}>

              <ShieldX className="h-3 w-3" /> Reject

            </Button>

          </div>

        </div>



        <div className="mt-5 flex flex-col gap-3 sm:flex-row lg:flex-col">

          <Button onClick={onOpen} disabled={!selected.evidence.available}><Maximize2/> View evidence</Button>

          <Button variant="outline" onClick={onDownload} disabled={!selected.evidence.available}><Download/> Download evidence</Button>

          {selected.evidence.metadataUrl && <Button variant="ghost" asChild><a href={selected.evidence.metadataUrl} target="_blank" rel="noreferrer"><FileText/> View evidence metadata</a></Button>}

        </div>

      </div>

    </div>

  </section>;

}



// ---------------------------------------------------------------------------

// Event report export and print

// ---------------------------------------------------------------------------



function ViolationReport({ result, selected }: { result: AnalysisResult; selected: Violation }) {

  const rows = [

    ["Exam-rule event", selected.type],

    ["Student track", selected.studentId],

    ["Session", selected.sessionId],

    ["Reason", selected.reason],

    ["Confidence", formatConfidence(selected.confidence)],

    ["Frame number", selected.evidence.frameNumber],

    ["Timestamp", selected.evidence.timestamp],

    ["Status", selected.evidence.status],

  ];



  const exportTextReport = () => {

    const content = `EXAM INVIGILATION — EVENT REPORT\n\nSession: ${result.id}\n${rows.map(([a, b]) => `${a}: ${b}`).join("\n")}\n\nValues shown are returned by the exam-monitoring backend.`;

    const blob = new Blob([content], { type: "text/plain" });

    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");

    a.href = url;

    a.download = `exam-event-${selected.id}.txt`;

    a.click();

    URL.revokeObjectURL(url);

  };



  return <section data-reveal className="mt-24">

    <div className="section-heading"><p className="eyebrow">Report 08</p><h2>Event report.</h2><p>Structured event details returned by the exam-monitoring backend.</p></div>

    <article id="print-report" className="mt-10 border border-border bg-report text-report-foreground">

      <header className="flex flex-col justify-between gap-6 border-b border-report-border p-6 sm:flex-row sm:items-end">

        <div><p className="text-xs font-bold uppercase tracking-[.16em]">Exam <span className="text-primary">AI</span></p><h3 className="mt-3 text-3xl font-semibold uppercase">Event report</h3></div>

        <div className="text-left text-[9px] uppercase leading-5 text-report-muted sm:text-right">Session ID / {result.id}<br/>Backend event data</div>

      </header>

      <div className="grid lg:grid-cols-[1fr_1fr]">

        <div className="p-5"><EvidenceVideo violation={selected}/></div>

        <div className="border-t border-report-border p-6 lg:border-l lg:border-t-0">{rows.map(([label, value]) => <div key={label} className="grid grid-cols-2 border-b border-report-border py-3 text-xs"><span className="text-report-muted">{label}</span><span className={label === "Status" ? "font-semibold text-success" : "font-medium"}>{value}</span></div>)}</div>

      </div>

      <footer className="flex flex-col justify-between gap-5 border-t border-report-border p-6 sm:flex-row sm:items-center">

        <p className="max-w-xl text-[9px] leading-4 text-report-muted">This report contains event data returned by the exam-monitoring backend and may be reviewed by an authorized officer.</p>

        <div className="no-print flex gap-3">

          <Button variant="outline" onClick={exportTextReport}>

            <FileText className="h-4 w-4"/> Export report

          </Button>

          <Button onClick={() => window.print()}><Printer/> Print report</Button>

        </div>

      </footer>

    </article>

  </section>;

}
