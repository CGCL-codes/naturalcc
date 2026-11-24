import React, { useId } from "react";
import { MAX_USER_ANSWER_CHARS, validUserAnswer } from "./agentInteraction.js";
import { formatToolCall } from "./agentUi.js";

const outcomes = [
  { id: "completed", label: "Confirmed completed" },
  { id: "failed", label: "Confirmed failed" },
  { id: "unknown", label: "Still unknown" }
];

export function AgentInteractionForm({ interaction, draft, busy, error, onChange, onSubmit }) {
  const fieldId = useId();
  const { request, kind } = interaction;
  const isInput = kind === "input";
  const valid = isInput
    ? validUserAnswer(request, draft.value)
    : outcomes.some((outcome) => outcome.id === draft.value);
  const options = isInput ? request.options || [] : outcomes;

  return (
    <form
      className="agent-interaction"
      aria-labelledby={`${fieldId}-title`}
      onSubmit={(event) => {
        event.preventDefault();
        if (!busy && valid) onSubmit();
      }}
    >
      <h4 id={`${fieldId}-title`}>{isInput ? "Input requested" : "Check tool outcome"}</h4>
      {isInput ? (
        <p className="agent-interaction-question" id={`${fieldId}-question`}>{request.question}</p>
      ) : (
        <>
          <pre className="approval-tool-details">{formatToolCall(request.tool_call)}</pre>
          <p className="field-help">Verify what happened before recording an outcome.</p>
        </>
      )}
      {isInput && request.answer_type === "confirm" ? (
        <fieldset className="agent-interaction-confirm" disabled={busy} aria-describedby={`${fieldId}-question`}>
          <legend>Your answer</legend>
          {[[true, "Yes"], [false, "No"]].map(([choice, label]) => (
            <label key={label}>
              <input
                type="radio"
                name={fieldId}
                checked={draft.value === choice}
                onChange={() => onChange({ value: choice })}
              />
              {label}
            </label>
          ))}
        </fieldset>
      ) : (
        <div>
          <label className="field-label" htmlFor={fieldId}>{isInput ? "Your answer" : "Observed outcome"}</label>
          {isInput && request.answer_type === "text" ? (
            <textarea
              id={fieldId}
              className="text-input"
              rows={3}
              maxLength={MAX_USER_ANSWER_CHARS}
              aria-describedby={`${fieldId}-question`}
              disabled={busy}
              value={draft.value || ""}
              onChange={(event) => onChange({ value: event.target.value })}
            />
          ) : (
            <select
              id={fieldId}
              className="text-input"
              aria-describedby={isInput ? `${fieldId}-question` : undefined}
              disabled={busy}
              value={draft.value || ""}
              onChange={(event) => onChange({ value: event.target.value })}
            >
              <option value="" disabled>{isInput ? "Select an answer" : "Select an outcome"}</option>
              {options.map(({ id, label }) => <option key={id} value={id}>{label}</option>)}
            </select>
          )}
        </div>
      )}
      {!isInput && (
        <div>
          <label className="field-label" htmlFor={`${fieldId}-note`}>Note (optional)</label>
          <textarea
            id={`${fieldId}-note`}
            className="text-input"
            rows={2}
            maxLength={2000}
            disabled={busy}
            value={draft.note}
            onChange={(event) => onChange({ note: event.target.value })}
          />
        </div>
      )}
      {error && <p className="agent-interaction-error" role="alert">{error}</p>}
      <button type="submit" className="primary-button compact-button" disabled={busy || !valid}>
        {isInput ? "Send answer" : "Record outcome and continue"}
      </button>
    </form>
  );
}
