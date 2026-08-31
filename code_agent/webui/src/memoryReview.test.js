import assert from "node:assert/strict";
import test from "node:test";

import {
  buildMemoryProposalSelection,
  groupMemoryReviews,
  normalizeMemoryReview,
  toggleEvidenceSelection
} from "./memoryReview.js";


test("normal review DTO drops internal model protocol fields", () => {
  const review = normalizeMemoryReview({
    proposal_id: "p1",
    title: "Keep API stable",
    evidence: [{ ref: "m1", preview: "Keep it stable" }],
    proposal_json: { secret: "machine protocol" },
    analysis_json: { chain: "internal" },
    raw_proposal: { sensitive: true }
  });

  assert.equal(review.title, "Keep API stable");
  assert.equal(review.evidence[0].preview, "Keep it stable");
  assert.equal(Object.hasOwn(review, "proposal_json"), false);
  assert.equal(Object.hasOwn(review, "analysis_json"), false);
  assert.equal(Object.hasOwn(review, "raw_proposal"), false);
});


test("message evidence selection is unique and reversible", () => {
  assert.deepEqual(toggleEvidenceSelection([], "m1"), ["m1"]);
  assert.deepEqual(toggleEvidenceSelection(["m1", "m2"], "m1"), ["m2"]);
  assert.deepEqual(
    buildMemoryProposalSelection("t1", "project", ["m1", "m1", "m2"]),
    {
      thread_id: "t1",
      project_id: "project",
      evidence_refs: [
        { type: "conversation_message", source_id: "m1" },
        { type: "conversation_message", source_id: "m2" }
      ]
    }
  );
});


test("reviews are grouped by governance state", () => {
  const groups = groupMemoryReviews([
    { status: "review_ready" },
    { status: "deferred" },
    { status: "applied" }
  ]);
  assert.equal(groups.reviewReady.length, 1);
  assert.equal(groups.deferred.length, 1);
  assert.equal(groups.history.length, 1);
});
