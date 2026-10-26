import { useEffect, useState } from "react";
import { ArrowDown, Menu, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { useBackendStatus } from "@/hooks/use-backend-status";
import { fetchViolationSummary } from "@/lib/api";

const nav: ReadonlyArray<readonly [string, string]> = [
  ["Features", "#how"], ["Demo", "#demo"], ["Technology", "#technology"], ["Use cases", "#use-cases"],
];

export function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const { state: backendState, version } = useBackendStatus();
  useEffect(() => {
    const update = () => setScrolled(window.scrollY > 36);
    update();
    window.addEventListener("scroll", update, { passive: true });
    return () => window.removeEventListener("scroll", update);
  }, []);
  const go = (href: string) => {
    setOpen(false);
    document.querySelector(href)?.scrollIntoView({ behavior: "smooth" });
  };

  const statusColor = backendState === "online" ? "bg-success" : backendState === "offline" ? "bg-violation" : "bg-white-foreground animate-pulse";
  const statusLabel = backendState === "online" ? `Backend online${version ? ` · v${version}` : ""}` : backendState === "offline" ? "Backend offline — start FastAPI server" : "Connecting...";

  return (
    <header className={`fixed inset-x-0 top-0 z-50 transition-all duration-500 ${scrolled ? "border-b border-border/70 bg-background/80 backdrop-blur-xl" : "bg-transparent"}`}>
      <div className="mx-auto grid h-18 max-w-[1440px] grid-cols-[minmax(0,1fr)_auto] items-center px-5 md:flex md:justify-between md:px-10 xl:px-14">
        <button onClick={() => go("#home")} className="flex min-w-0 items-center gap-3" aria-label="Exam Invigilation AI home">
          <span className="grid h-8 w-8 shrink-0 place-items-center border border-primary/50 bg-primary/10 text-[10px] font-bold text-primary">EI</span>
          <span className="truncate text-sm font-bold uppercase tracking-[0.16em]">Exam Invigilation <span className="text-primary">AI</span></span>
        </button>
        <nav className="hidden items-center gap-8 md:flex" aria-label="Main navigation">
          {nav.map(([label, href]) => <button key={href} onClick={() => go(href)} className="text-xs text-white transition-colors hover:text-black">{label}</button>)}
        </nav>
        <div className="hidden items-center gap-4 md:flex">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center gap-2 cursor-default" aria-label={statusLabel}>
                  <span className={`h-2 w-2 shrink-0 rounded-full ${statusColor} shadow-[0_0_6px_currentColor]`} />
                  <span className="text-[9px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">{backendState === "online" ? "Live" : backendState === "offline" ? "Offline" : "..."}</span>
                </div>
              </TooltipTrigger>
              <TooltipContent side="bottom" className="bg-background border-border text-xs"><p>{statusLabel}</p></TooltipContent>
            </Tooltip>
          </TooltipProvider>
          <Button onClick={() => go("#demo")} size="sm">Get started</Button>
        </div>
        <Button variant="ghost" size="icon" className="md:hidden" onClick={() => setOpen((value) => !value)} aria-label="Toggle menu">{open ? <X /> : <Menu />}</Button>
      </div>
      {open && <div className="border-t border-border bg-background/95 px-5 py-5 backdrop-blur-xl md:hidden">
        <div className="flex items-center gap-2 pb-4 text-xs text-muted-foreground"><span className={`h-2 w-2 rounded-full ${statusColor}`}/>{statusLabel}</div>
        {nav.map(([label, href]) => <button key={href} onClick={() => go(href)} className="block w-full border-b border-border py-4 text-left text-sm">{label}</button>)}<Button onClick={() => go("#demo")} className="mt-5 w-full">Get started</Button>
      </div>}
    </header>
  );
}

export function Hero() {
  const [metrics, setMetrics] = useState([
    { value: "—", label: "Stored events" },
    { value: "—", label: "Observed rules" },
    { value: "—", label: "Sessions represented" },
  ]);
  useEffect(() => {
    let active = true;
    void fetchViolationSummary().then((summary) => {
      if (!active) return;
      setMetrics([
        { value: String(summary.total_violations), label: "Stored events" },
        { value: String(Object.keys(summary.violations_by_rule).length), label: "Observed rules" },
        { value: String(Object.keys(summary.violations_by_session).length), label: "Sessions represented" },
      ]);
    }).catch(() => undefined);
    return () => { active = false; };
  }, []);
  return (
    <section id="home" className="relative min-h-[760px] overflow-hidden md:min-h-screen">
      {/* ── Hero background video ── */}
      <div data-hero-image className="absolute inset-0 scale-[1.02] overflow-hidden">
        {/* Gradient fallback — sits behind the video, visible only while video loads or if it fails */}
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_68%_42%,color-mix(in_oklab,var(--primary)_18%,transparent),transparent_45%),linear-gradient(135deg,var(--surface),var(--background)_70%)]" />
        
      </div>
      <div className="absolute inset-0 bg-[linear-gradient(90deg,var(--background)_0%,color-mix(in_oklab,var(--background)_88%,transparent)_32%,color-mix(in_oklab,var(--background)_20%,transparent)_78%),linear-gradient(0deg,var(--background)_0%,transparent_42%)]" />
      <div className="relative mx-auto flex min-h-[760px] max-w-[1440px] items-center px-5 pb-28 pt-28 md:min-h-screen md:px-10 xl:px-14">
        <div className="max-w-3xl">
          <p data-hero-reveal className="mb-5 flex items-center gap-3 text-[11px] font-semibold uppercase tracking-[0.24em] text-primary"><span className="h-px w-8 bg-primary" /> Exam Invigilation / Vision system 01</p>
          <h1 data-hero-reveal className="max-w-[760px] text-[clamp(3.3rem,7.1vw,7.6rem)] font-semibold uppercase leading-[0.82] text-foreground">Intelligent<br />exam<br /><span className="text-outline">integrity</span><br />monitoring</h1>
          <p data-hero-reveal className="mt-7 max-w-xl text-sm leading-7 text-muted-foreground md:text-base">Analyze exam-session video with student tracking, head-direction and behavior signals, exam-rule event detection, and reviewable evidence.</p>
          <div data-hero-reveal className="mt-8 flex flex-col gap-3 sm:flex-row">
            <Button size="lg" onClick={() => document.querySelector("#demo")?.scrollIntoView({ behavior: "smooth" })}>Try demo <span aria-hidden>→</span></Button>
            <Button size="lg" variant="outline" onClick={() => document.querySelector("#detection")?.scrollIntoView({ behavior: "smooth" })}>Watch detection</Button>
          </div>
          <div data-hero-reveal className="mt-12 grid max-w-2xl grid-cols-3 border-y border-border/80 py-5">
            {metrics.map((metric) => <div key={metric.label} className="border-r border-border px-3 first:pl-0 last:border-0 md:px-6"><p className="text-xl font-semibold md:text-3xl">{metric.value}</p><p className="mt-1 text-[9px] uppercase leading-4 tracking-[0.08em] text-muted-foreground md:text-[10px]">{metric.label}</p></div>)}
          </div>
          <p className="mt-3 text-[9px] uppercase tracking-[0.12em] text-subtle">Live counts from the connected exam-monitoring database</p>
        </div>
      </div>
      <button onClick={() => document.querySelector("#detection")?.scrollIntoView({ behavior: "smooth" })} className="absolute bottom-7 left-1/2 flex -translate-x-1/2 flex-col items-center gap-2 text-[9px] uppercase tracking-[0.18em] text-muted-foreground"><span>Scroll to explore</span><ArrowDown className="h-4 w-4 animate-bounce" /></button>
    </section>
  );
}
