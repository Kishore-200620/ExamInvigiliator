import { Activity, ArrowDown, BrainCircuit, Camera, CheckCheck, ClipboardCheck, FileOutput, ScanLine, ShieldCheck, Upload, Users, Video } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useExamSamples } from "@/hooks/use-exam-samples";

export function DetectionShowcase() {
  const { samples, loading } = useExamSamples();
  const sample = samples[0];

  return <section id="detection" className="relative h-[360vh] bg-background">
    <div data-detection-stage className="sticky top-0 h-screen overflow-hidden">
      {sample ? <video data-detection-image src={sample.url} controls preload="none" className="absolute inset-0 h-full w-full object-cover" aria-label={`Exam-session sample video: ${sample.name}`} /> : <div data-detection-image className="absolute inset-0 bg-[radial-gradient(ellipse_at_48%_48%,color-mix(in_oklab,var(--primary)_18%,transparent),transparent_48%),linear-gradient(135deg,var(--surface),var(--background)_75%)]" />}
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(90deg,color-mix(in_oklab,var(--background)_30%,transparent),color-mix(in_oklab,var(--background)_5%,transparent)_55%,color-mix(in_oklab,var(--background)_65%,transparent)),linear-gradient(0deg,var(--background),transparent_30%,transparent_70%,var(--background))]" />
      <div className="absolute left-5 top-24 z-10 md:left-10 xl:left-14"><p className="eyebrow">Sequence 02 / Exam session</p><h2 className="mt-3 text-3xl font-semibold uppercase md:text-5xl">Student activity,<br />made reviewable.</h2></div>
      <div data-scan-line className="absolute left-[20%] top-[14%] z-10 h-px w-[60%] bg-primary shadow-[0_0_18px_var(--primary)]" />
      <div className="absolute right-5 top-[24%] z-20 w-[min(330px,43vw)] space-y-4 md:right-10 xl:right-14">
        <div data-detection-label className="hud-panel"><span className="eyebrow">Video input</span><p className="mt-2 truncate text-lg uppercase">{sample?.name ?? (loading ? "Loading exam videos" : "No sample video available")}</p><div className="mt-4 flex justify-between border-t border-border pt-3 text-[10px] uppercase text-muted-foreground"><span>Source</span><span className="text-primary">Backend sample library</span></div></div>
        <div data-detection-label data-student-tracking className="hud-panel"><span className="eyebrow">Analysis pipeline</span><p className="mt-2 text-lg uppercase">Student tracking</p><div className="mt-4 flex justify-between border-t border-border pt-3 text-[10px] uppercase text-muted-foreground"><span>Exam-rule evaluation</span><span className="text-primary">Configured</span></div></div>
        <div data-rule-summary className="border border-primary/50 bg-primary/10 p-5 backdrop-blur-md"><p className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.18em] text-primary"><ShieldCheck className="h-4 w-4" /> Evidence workflow</p><p className="mt-3 text-2xl font-semibold uppercase md:text-3xl">Detect.<br />Review.<br />Record.</p></div>
      </div>
      <div className="absolute bottom-8 left-5 z-20 flex items-center gap-4 text-[9px] uppercase tracking-[0.14em] text-muted-foreground md:left-10 xl:left-14"><span className="h-1.5 w-1.5 bg-success" /> Exam monitoring <span>/</span> MP4 input <span>/</span> SQLite records</div>
    </div>
  </section>;
}

const steps = [
  { icon: Upload, title: "Choose exam video", text: "Select a backend sample or upload an MP4 exam recording." },
  { icon: Users, title: "Track students", text: "The live pipeline detects and tracks people across video frames." },
  { icon: BrainCircuit, title: "Evaluate exam rules", text: "Behavior signals are checked against the configured rule engine." },
  { icon: FileOutput, title: "Review evidence", text: "Inspect persisted events and available evidence, then record a review." },
];

export function HowItWorks() {
  return <section id="how" className="section-shell overflow-hidden"><div data-reveal className="section-heading text-center"><p className="eyebrow justify-center">System workflow</p><h2>How it works</h2><p>From an exam-session video to tracked students, rule events, and reviewable evidence.</p></div><div className="relative mt-16 grid gap-9 md:grid-cols-4 md:gap-4"><div className="absolute left-[12.5%] right-[12.5%] top-8 hidden h-px bg-border md:block" />{steps.map((step, index) => <div data-step key={step.title} className="relative grid grid-cols-[64px_1fr] gap-5 md:block md:text-center"><div className="relative z-10 grid h-16 w-16 place-items-center rounded-full border border-primary/60 bg-background text-primary shadow-[0_0_30px_color-mix(in_oklab,var(--primary)_12%,transparent)]"><step.icon className="h-6 w-6" /><span className="absolute -top-3 right-0 grid h-5 w-5 place-items-center rounded-full bg-primary text-[9px] font-bold text-primary-foreground">{index + 1}</span></div><div className="md:mt-7"><p className="text-[9px] uppercase tracking-[0.18em] text-primary">Step 0{index + 1}</p><h3 className="mt-2 text-base font-semibold uppercase">{step.title}</h3><p className="mx-auto mt-3 max-w-[190px] text-xs leading-5 text-muted-foreground">{step.text}</p></div></div>)}</div></section>;
}

const tech = [
  [Video, "Video ingestion"], [ScanLine, "Person detection"], [Users, "Multi-student tracking"], [Activity, "Head and movement signals"], [ShieldCheck, "Temporal exam rules"], [ClipboardCheck, "Review decisions"],
] as const;

export function Technology() {
  const capabilities = ["MP4 session video", "Person detection", "Student track state", "Behavior-rule evaluation", "SQLite event storage", "Evidence retrieval"];
  return <section id="technology" className="section-shell border-y border-border/70 bg-surface"><div data-reveal className="section-heading"><p className="eyebrow">Technology 09</p><h2>Built for exam-session<br />monitoring.</h2></div><div className="mt-14 grid gap-px overflow-hidden border border-border bg-border md:grid-cols-6">{capabilities.map((item, i) => <div key={item} className="bg-background p-5"><span className="text-[9px] text-primary">0{i + 1}</span><p className="mt-8 text-xs font-semibold uppercase leading-5">{item}</p><ArrowDown className="mt-4 h-3 w-3 text-muted-foreground md:rotate-[-90deg]" /></div>)}</div><div className="mt-12 grid gap-px bg-border sm:grid-cols-2 lg:grid-cols-3">{tech.map(([Icon, label]) => <div key={label} data-reveal className="flex items-center gap-4 bg-surface p-7"><Icon className="h-5 w-5 text-primary" /><span className="text-sm uppercase">{label}</span></div>)}</div></section>;
}

const useCases = [
  { icon: Camera, title: "Exam-session video", text: "Start live analysis from the backend sample library or an uploaded MP4." },
  { icon: Users, title: "Student activity", text: "Follow available student tracks, head direction, and movement signals." },
  { icon: CheckCheck, title: "Evidence review", text: "Inspect saved event evidence and persist confirm or reject decisions." },
];

export function UseCases() {
  const { samples } = useExamSamples();
  return <section id="use-cases" className="section-shell"><div data-reveal className="section-heading"><p className="eyebrow">Workflow 10</p><h2>One invigilation system.<br />A reviewable session.</h2></div><div className="mt-14 grid gap-5 lg:grid-cols-3">{useCases.map(({ icon: Icon, title, text }, index) => {
    const sample = samples[index % Math.max(samples.length, 1)];
    return <article data-reveal key={title} className="group relative aspect-[4/5] overflow-hidden border border-border">
      {sample ? <video src={sample.url} muted preload="metadata" className="h-full w-full object-cover transition-transform duration-700 group-hover:scale-105" aria-label={`Exam video sample ${sample.name}`} /> : <div className="h-full w-full bg-[radial-gradient(ellipse_at_50%_35%,color-mix(in_oklab,var(--primary)_18%,transparent),transparent_45%),linear-gradient(145deg,var(--surface),var(--background))]" />}
      <div className="absolute inset-0 bg-gradient-to-t from-background via-background/35 to-transparent" /><div className="absolute inset-x-0 bottom-0 p-7"><Icon className="mb-5 h-6 w-6 text-primary" /><h3 className="text-xl font-semibold uppercase">{title}</h3><p className="mt-3 max-w-xs text-sm leading-6 text-muted-foreground">{text}</p></div>
    </article>;
  })}</div></section>;
}

export function ImpactAndCta() {
  const metrics = [
    ["Live", "Student tracking during session processing"],
    ["Rule engine", "Temporal exam-behavior evaluation"],
    ["SQLite", "Persisted event and evidence records"],
    ["Officer review", "Confirm or reject saved events"],
  ];
  return <><section className="relative min-h-[720px] overflow-hidden"><div className="absolute inset-0 bg-[radial-gradient(ellipse_at_55%_42%,color-mix(in_oklab,var(--primary)_18%,transparent),transparent_48%),linear-gradient(135deg,var(--surface),var(--background))]"/><div className="absolute inset-0 bg-[linear-gradient(0deg,var(--background),transparent_40%,color-mix(in_oklab,var(--background)_45%,transparent)),linear-gradient(90deg,var(--background),transparent)]"/><div className="section-shell relative flex min-h-[720px] flex-col justify-end"><p className="eyebrow">Backend capabilities</p><h2 className="max-w-4xl text-4xl font-semibold uppercase md:text-7xl">Support fair exams.<br />Make events reviewable.</h2><div className="mt-12 grid border-y border-border bg-background/55 backdrop-blur-md sm:grid-cols-2 lg:grid-cols-4">{metrics.map(([value,label]) => <div key={value} className="border-border p-6 sm:border-r"><p className="text-3xl font-semibold uppercase">{value}</p><p className="mt-2 max-w-[180px] text-xs leading-5 text-muted-foreground">{label}</p></div>)}</div></div></section><section className="relative grid min-h-[80vh] place-items-center overflow-hidden text-center"><div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,color-mix(in_oklab,var(--primary)_13%,transparent),transparent_54%),var(--background)]"/><div className="absolute inset-0 bg-[radial-gradient(circle_at_center,transparent,var(--background)_72%)]"/><div data-reveal className="relative px-5"><p className="eyebrow justify-center">Exam Invigilation AI</p><h2 className="mt-5 text-5xl font-semibold uppercase leading-[0.9] md:text-8xl">See the session<br />with context.</h2><p className="mt-6 text-sm text-muted-foreground">Analyze exam video, inspect detected events, and review evidence.</p><Button size="lg" className="mt-8" onClick={() => document.querySelector("#demo")?.scrollIntoView({behavior:"smooth"})}>Open the analysis console →</Button></div></section><footer className="border-t border-border px-5 py-8 text-[10px] uppercase tracking-[0.12em] text-muted-foreground md:px-10"><div className="mx-auto flex max-w-[1440px] flex-col justify-between gap-4 sm:flex-row"><span>Exam Invigilation AI / Monitoring system</span><span>Computer vision for exam integrity</span></div></footer></>;
}
