import { createFileRoute } from "@tanstack/react-router";
import { ExamInvigilationPage } from "@/components/exam/ExamInvigilationPage";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Exam Invigilation AI — Exam Session Monitoring" },
      { name: "description", content: "Monitor exam-session video with student tracking, exam-rule events, and reviewable evidence." },
      { property: "og:title", content: "Exam Invigilation AI — Exam Session Monitoring" },
      { property: "og:description", content: "Analyze exam videos, inspect detected events, and review persisted evidence." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: ExamInvigilationPage,
});
