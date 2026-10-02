"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Customer Intelligence" },
  { href: "/risk", label: "Customer Risk" },
  { href: "/models", label: "Model Lab" },
  { href: "/experiment", label: "Experiment Lab" },
  { href: "/uplift", label: "Uplift Lab" },
  { href: "/campaign", label: "Campaign Simulator" },
];

export function Nav() {
  const path = usePathname();
  return (
    <nav className="flex gap-1 overflow-x-auto md:flex-col">
      {LINKS.map((l) => {
        const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
        return (
          <Link
            key={l.href}
            href={l.href}
            className={`whitespace-nowrap rounded-lg px-3 py-2 text-sm ${
              active ? "bg-surface-1 font-medium text-ink shadow-sm" : "text-ink-2 hover:text-ink"
            }`}
          >
            {l.label}
          </Link>
        );
      })}
    </nav>
  );
}
