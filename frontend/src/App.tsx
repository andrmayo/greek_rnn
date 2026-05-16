import { useState } from "react";

import { ModelSelector } from "./components/ModelSelector";
import { PredictForm } from "./components/PredictForm";
import { PredictKForm } from "./components/PredictKForm";
import { RankForm } from "./components/RankForm";
import { FileUploadForm } from "./components/FileUploadForm";
import { Widget } from "./components/Widget";

type Tab = "predict" | "predict-k" | "rank" | "batch";

const TABS: { id: Tab; label: string }[] = [
  { id: "predict", label: "Predict" },
  { id: "predict-k", label: "Top K" },
  { id: "rank", label: "Rank" },
  { id: "batch", label: "Batch" },
];

function Header() {
  return (
    <header className="h-[65px] bg-[#343a40] flex items-center px-6 shrink-0">
      <span className="font-['Noto_Serif'] text-white text-xl italic">
        Greek Lacuna Reconstruction
      </span>
    </header>
  );
}

function Sidebar({
  side,
  children,
}: {
  side: "left" | "right";
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(true);
  const border =
    side === "left"
      ? "border-r border-[var(--color-border)]"
      : "border-l border-[var(--color-border)]";

  return (
    <div
      className="relative flex shrink-0 transition-all duration-300"
      style={{ width: open ? "280px" : "0" }}
    >
      <div className={`w-[280px] h-full overflow-y-auto bg-white ${border}`}>
        {children}
      </div>
      <button
        onClick={() => setOpen(!open)}
        className={`absolute ${side === "left" ? "-right-6" : "-left-6"} top-4 text-[var(--color-text-secondary)] hover:text-[var(--color-text)] text-sm`}
      >
        {side === "left" ? (open ? "<" : ">") : open ? ">" : "<"}
      </button>
    </div>
  );
}

function TabBar({
  active,
  onChange,
}: {
  active: Tab;
  onChange: (tab: Tab) => void;
}) {
  return (
    <div className="flex border-b border-[var(--color-border)]">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          className={`px-5 py-3 font-['Noto_Sans'] text-sm transition-colors ${
            active === tab.id
              ? "border-b-2 border-[var(--color-brand)] text-[var(--color-brand)]"
              : "text-[var(--color-text-secondary)] hover:text-[var(--color-text)]"
          }`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

function ContentBody() {
  const [activeTab, setActiveTab] = useState<Tab>("predict");
  return (
    <section className="flex-1 overflow-y-auto bg-white flex flex-col">
      <TabBar active={activeTab} onChange={setActiveTab} />
      <div className="p-6">
        {activeTab === "predict" && <PredictForm />}
        {activeTab === "predict-k" && <PredictKForm />}
        {activeTab === "rank" && <RankForm />}
        {activeTab === "batch" && <FileUploadForm />}
      </div>
    </section>
  );
}

function App() {
  return (
    <div className="flex flex-col h-screen">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar side="left">
          <Widget title="Model">
            <p className="p-2 text-sm text-[var(--color-text-secondary)]">
              <ModelSelector />
            </p>
          </Widget>
        </Sidebar>
        <ContentBody />
        <Sidebar side="right">
          <Widget title="History">
            <p className="p-2 text-sm text-[var(--color-text-secondary)]">
              Results history coming soon
            </p>
          </Widget>
        </Sidebar>
      </div>
    </div>
  );
}

export default App;
