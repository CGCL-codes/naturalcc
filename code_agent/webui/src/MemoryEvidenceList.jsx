import React from "react";


export function MemoryEvidenceList({ evidence = [] }) {
  if (!evidence.length) {
    return <p className="memory-empty-copy">No evidence is attached.</p>;
  }
  return (
    <details className="memory-evidence" open={evidence.length <= 2}>
      <summary>Why this was proposed · {evidence.length} evidence item{evidence.length === 1 ? "" : "s"}</summary>
      <ol>
        {evidence.map((item) => (
          <li key={item.ref}>
            <div>
              <strong>{item.label}</strong>
              <span>{item.source_locator} · {item.verification_label}</span>
            </div>
            <blockquote>{item.preview}</blockquote>
          </li>
        ))}
      </ol>
    </details>
  );
}
