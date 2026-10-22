import { Navbar, Hero } from "./SiteChrome";
import { DetectionShowcase, HowItWorks, ImpactAndCta, Technology, UseCases } from "./StorySections";
import { DemoWorkspace } from "./DemoWorkspace";
import { useCinematicMotion } from "./useCinematicMotion";

export function ExamInvigilationPage() {
  useCinematicMotion();
  return <main className="overflow-clip bg-background text-foreground"><Navbar/><Hero/><DetectionShowcase/><HowItWorks/><DemoWorkspace/><Technology/><UseCases/><ImpactAndCta/></main>;
}
