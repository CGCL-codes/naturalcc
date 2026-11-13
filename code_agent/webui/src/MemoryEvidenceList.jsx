import React from "react";


// [maint-note:9e1f7e51dcbe]
// 维护导航：MemoryEvidenceList；下列为本函数源码中的接口与路径索引。
// 输入 { evidence = [] }：类型由调用方与函数体约束。
// 分支条件：!evidence.length。
// 返回路径：<p className="memory-empty-copy">No evidence is attached.</p>。
// 返回路径：( <details className="memory-evidence" open={evidence.length <= 2}> <summary>Why this was proposed · {evidence.length} evidence item{evidence.length === 1 ? "" : "s"}</summary> <ol> {evidence.map((item) => ( <li key={item.ref}> <div> <stron。
// 直接调用：evidence.map。
// [/maint-note:9e1f7e51dcbe]
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
