import React, { useEffect, useState } from "react";
import { Check, Clock3, Pencil, ShieldAlert, X } from "lucide-react";
import { MemoryEvidenceList } from "./MemoryEvidenceList.jsx";


const SCOPE_OPTIONS = [
  ["run", "Current task"],
  ["thread", "Current conversation"],
  ["project", "Current project"],
  ["user", "All projects"]
];
const KIND_OPTIONS = [
  ["user_preference", "User preference"],
  ["project_constraint", "Project constraint"],
  ["architecture_decision", "Architecture decision"],
  ["verified_fact", "Verified fact"],
  ["repository_convention", "Repository convention"],
  ["workflow", "Workflow"],
  ["failure_pattern", "Failure pattern"],
  ["successful_approach", "Successful approach"],
  ["important_artifact", "Important artifact"],
  ["task_summary", "Task archive"]
];


export function MemoryProposalCard({ proposal, busy = false, onAction }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState({});
  useEffect(() => {
    setDraft({
      subject: proposal.title || "",
      canonical_content: proposal.summary || "",
      scope: proposal.scope || "project",
      kind: proposal.kind || "project_constraint",
      expires_at: proposal.expires_at || ""
    });
    setEditing(false);
  }, [proposal]);

  const can = (action) => proposal.allowed_actions?.includes(action);
  const invoke = async (action, changes = null) => {
    await onAction(proposal, action, changes);
    if (action === "edit") setEditing(false);
  };

  return (
    <article className="memory-proposal-card">
      <header>
        <div>
          <small>{proposal.operation_label}</small>
          <h5>{proposal.title}</h5>
        </div>
        <span className={`memory-status ${proposal.status}`}>{proposal.status.replaceAll("_", " ")}</span>
      </header>
      <div className="memory-badges">
        <span>{proposal.kind_label}</span>
        <span>{proposal.scope_label}</span>
        <span>{proposal.confidence_label}</span>
        <span>{proposal.verification_label}</span>
      </div>

      {editing ? (
        <div className="memory-edit-form">
          <label>Title<input value={draft.subject} onChange={(event) => setDraft({ ...draft, subject: event.target.value })} /></label>
          <label>What to remember<textarea rows={4} value={draft.canonical_content} onChange={(event) => setDraft({ ...draft, canonical_content: event.target.value })} /></label>
          <div>
            <label>Scope<select value={draft.scope} onChange={(event) => setDraft({ ...draft, scope: event.target.value })}>{SCOPE_OPTIONS.map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select></label>
            <label>Type<select value={draft.kind} onChange={(event) => setDraft({ ...draft, kind: event.target.value })}>{KIND_OPTIONS.map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select></label>
          </div>
          <label>Expires at (optional)<input type="datetime-local" value={draft.expires_at} onChange={(event) => setDraft({ ...draft, expires_at: event.target.value })} /></label>
        </div>
      ) : (
        <p className="memory-proposal-summary">{proposal.summary}</p>
      )}

      {proposal.target_memory && (
        <details className="memory-diff">
          <summary>Compare with current memory</summary>
          <p><strong>Before:</strong> {proposal.target_memory.content}</p>
          <p><strong>Proposed:</strong> {proposal.summary}</p>
        </details>
      )}
      {(proposal.warnings?.length > 0 || proposal.conflicts?.length > 0) && (
        <div className="memory-warnings">
          <ShieldAlert size={15} />
          <div>
            {[...(proposal.warnings || []), ...(proposal.conflicts || [])].map((item) => <p key={item}>{item}</p>)}
          </div>
        </div>
      )}
      <MemoryEvidenceList evidence={proposal.evidence} />
      <p className="memory-impact"><strong>Impact:</strong> {proposal.impact}</p>

      {proposal.allowed_actions?.length > 0 && (
        <footer>
          {editing ? (
            <>
              <button type="button" disabled={busy || !draft.subject.trim() || !draft.canonical_content.trim()} className="memory-approve" onClick={() => invoke("edit", draft)}><Check size={13} />Save review</button>
              <button type="button" disabled={busy} onClick={() => setEditing(false)}><X size={13} />Cancel</button>
            </>
          ) : (
            <>
              {can("approve") && <button type="button" disabled={busy} className="memory-approve" onClick={() => invoke("approve")}><Check size={13} />Accept and remember</button>}
              {can("edit") && <button type="button" disabled={busy} onClick={() => setEditing(true)}><Pencil size={13} />Edit</button>}
              {can("defer") && proposal.status !== "deferred" && <button type="button" disabled={busy} onClick={() => invoke("defer")}><Clock3 size={13} />Later</button>}
              {can("reject") && <button type="button" disabled={busy} className="memory-reject" onClick={() => invoke("reject")}><X size={13} />Reject</button>}
            </>
          )}
        </footer>
      )}
    </article>
  );
}
