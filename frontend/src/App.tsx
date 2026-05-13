import { useState } from "react";

function Header() {
  return (
    <header className="h-[65px] bg-[#343a40] flex items-center px-6 shrink-0">
      <span className="font-['Noto_Serif'] text-white text-xl italic">
        Lacuna RNN
      </span>
    </header>
  );
}

function Sidebar({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(true);
  return (
    <div
      className="relative flex shrink-0 transition-all duration-300"
      style={{ width: open ? "280px" : "0" }}
    >
      <div className="w-[280px] h-full overflow-y-auto bg-white border-r border-[var(--color-border)]">
        {children}
      </div>
      <button
        onClick={() => setOpen(!open)}
        className="absolute -right-6 top-4 text-[var(--color-text-secondary)] hover:text-[var(--color-text)] text-sm"
      >
        {open ? "<" : ">"}
      </button>
    </div>
  );
}

function ContentBody({ children }: { children: React.ReactNode }) {
  return (
    <section className="flex-1 overflow-y-auto p-6 bg-white">
      {children}
    </section>
  );
}

function App() {
  return (
    <div className="flex flex-col h-screen">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar>
          <p className="p-4 text-sm text-[var(--color-text-secondary)]">
            Left sidebar
          </p>
        </Sidebar>
        <ContentBody>
          <p className="text-[var(--color-text-secondary)]">Content area</p>
        </ContentBody>
        <Sidebar>
          <p className="p-4 text-sm text-[var(--color-text-secondary)]">
            Right sidebar
          </p>
        </Sidebar>
      </div>
    </div>
  );
}

export default App;
