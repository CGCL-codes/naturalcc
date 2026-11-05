import type { ToolSchema } from '../types.js'
import { MAX_APPROVAL_REASON_CHARS } from '../approval.js'
import { MAX_PATCH_LENGTH } from '../toolUtils.js'

export const applyPatchSchema: ToolSchema = {
  name: 'apply_patch',
  description: 'Apply a precise, reviewable patch inside the workspace; updates and new files require approval. The patch uses a Codex envelope with Add File, Update File, and Delete File directives. Prefer a bare @@ update hunk header; numeric @@ -old[,count] +new[,count] @@ headers are also accepted.',
  readOnly: false,
  requiresApproval: true,
  repeatKind: 'workspaceMutation',
  parameters: {
    type: 'object',
    properties: {
      patch: {
        type: 'string',
        minLength: 1,
        maxLength: MAX_PATCH_LENGTH,
        description: 'Use a Codex envelope with *** Begin Patch and *** End Patch, and Add File, Update File, or Delete File directives. For Update File, prefer a bare @@ hunk header; numeric @@ -old[,count] +new[,count] @@ is also accepted. Every hunk body line must start with a space, +, or -, except the literal \\ No newline at end of file marker. Minimal example: *** Begin Patch\\n*** Update File: file.txt\\n@@\\n-old\\n+new\\n*** End Patch',
      },
      reason: {
        type: 'string',
        minLength: 1,
        maxLength: MAX_APPROVAL_REASON_CHARS,
        description: '用一句简洁、具体、面向用户的话说明为什么必须应用这个 patch，不得编造结果',
      },
    },
    required: ['patch', 'reason'],
    additionalProperties: false,
  },
}
