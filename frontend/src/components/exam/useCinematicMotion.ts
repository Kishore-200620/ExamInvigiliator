import { useEffect } from "react";
import Lenis from "lenis";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

export function useCinematicMotion() {
  useEffect(() => {
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduceMotion) return;

    const lenis = new Lenis({ duration: 1.05, smoothWheel: true });
    const tick = (time: number) => lenis.raf(time * 1000);
    gsap.ticker.add(tick);
    gsap.ticker.lagSmoothing(0);
    lenis.on("scroll", ScrollTrigger.update);

    const ctx = gsap.context(() => {
      gsap.from("[data-hero-reveal]", { y: 28, opacity: 0, duration: 1, stagger: 0.12, ease: "power3.out", delay: 0.15 });
      gsap.to("[data-hero-image]", { scale: 1.08, ease: "none", scrollTrigger: { trigger: "#home", start: "top top", end: "bottom top", scrub: 1.2 } });
      gsap.utils.toArray<HTMLElement>("[data-reveal]").forEach((el) => {
        gsap.from(el, { y: 36, opacity: 0, duration: 0.8, ease: "power3.out", scrollTrigger: { trigger: el, start: "top 88%", once: true } });
      });
      const timeline = gsap.timeline({ scrollTrigger: { trigger: "#detection", start: "top top", end: "+=260%", scrub: 1, pin: "[data-detection-stage]" } });
      timeline.to("[data-detection-image]", { scale: 1.09, duration: 3, ease: "none" })
        .fromTo("[data-scan-line]", { yPercent: -20, opacity: 0 }, { yPercent: 660, opacity: 1, duration: 2 }, 0.4)
        .from("[data-student-tracking]", { clipPath: "inset(0 100% 100% 0)", opacity: 0, duration: 0.8 }, 1)
        .from("[data-detection-label]", { x: 20, opacity: 0, stagger: 0.2, duration: 0.6 }, 2.1)
        .from("[data-rule-summary]", { scale: 0.94, opacity: 0, duration: 0.8 }, 3);
      gsap.from("[data-step]", { y: 30, opacity: 0, stagger: 0.14, duration: 0.7, scrollTrigger: { trigger: "#how", start: "top 72%", once: true } });
    });

    return () => {
      ctx.revert();
      lenis.destroy();
      gsap.ticker.remove(tick);
    };
  }, []);
}
