import React from "react";
import { groupMemoryReviews } from "./memoryReview.js";
import { MemoryProposalCard } from "./MemoryProposalCard.jsx";


// [maint-note:5787526d764d]
// 维护导航：MemoryReviewPanel；下列为本函数源码中的接口与路径索引。
// 输入 { proposals = [], busy = false, error = "", onAction }：类型由调用方与函数体约束。
// 返回路径：( <div className="memory-review-panel"> <div className="memory-review-heading"> <div> <h4>Memory review</h4> <p>Model suggestions are converted into readable cards before anything is remembered.</p> </div> <span>{groups.reviewReady.length} 。
// 直接调用：groupMemoryReviews。
// 直接调用：groups.reviewReady.map。
// 直接调用：groups.deferred.map。
// [/maint-note:5787526d764d]
export function MemoryReviewPanel({ proposals = [], busy = false, error = "", onAction }) {
  const groups = groupMemoryReviews(proposals);
  return (
    <div className="memory-review-panel">
      <div className="memory-review-heading">
        <div>
          <h4>Memory review</h4>
          <p>Model suggestions are converted into readable cards before anything is remembered.</p>
        </div>
        <span>{groups.reviewReady.length} pending</span>
      </div>
      {error && <div className="error-box memory-review-error" role="alert">{error}</div>}
      {groups.reviewReady.map((proposal) => <MemoryProposalCard proposal={proposal} busy={busy} onAction={onAction} key={proposal.proposal_id} />)}
      {groups.deferred.length > 0 && <h5 className="memory-group-title">Deferred</h5>}
      {groups.deferred.map((proposal) => <MemoryProposalCard proposal={proposal} busy={busy} onAction={onAction} key={proposal.proposal_id} />)}
      {!groups.reviewReady.length && !groups.deferred.length && <p className="memory-empty-copy">No memory suggestions need review.</p>}
    </div>
  );
}
