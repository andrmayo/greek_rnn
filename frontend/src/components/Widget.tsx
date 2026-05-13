import { useState } from "react";

import type { ReactNode } from "react";

interface WidgetProps {
  title: string;
  children: ReactNode;
  defaultOpen?: boolean;
}

export function Widget({ title, children, defaultOpen = true }: WidgetProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="border-b border-[var(--color-border)] px-2 pb-2">
      <div className="sticky top-0 bg-white py-3">
        <button
          onClick={() => setOpen(!open)}
          className="flex items-center gap-2 w-full text-left"
        >
          <span className="text-sm font-['Noto_Sans'] uppercase tracking-widest text-[var(--color-text-secondary)]">
            {open ? "▾" : "▸"}
          </span>
          <span className="text-sm font-['Noto_Sans'] uppercase tracking-widest text-[var(--color-text-secondary)]">
            {title}
          </span>
        </button>
      </div>
      {open && <div className="overflow-y-auto">{children}</div>}
    </div>
  );
}
