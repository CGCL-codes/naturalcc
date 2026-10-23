# SWE-bench Multilingual task: jekyll__jekyll-8771

Resolve the GitHub issue below in the repository workspace. Follow existing conventions and add or update tests as appropriate. Do not use the reference patch; the official evaluator validates the patch in the repository's pinned environment.

## Issue

Incremental regeneration ignores changes to data files
## My Environment

<!--
  Replace the values in the Version(s) column with the ones in your build. If you're not
  using `github-pages`, just replace it with "No".
-->

| Software         | Version(s) |
| ---------------- | ---------- |
| Operating System |     MacOS High Sierra  10.13.6     |
| `jekyll`         |  4.0.0.pre.alpha1     |
| `github-pages`   | No     |

---

## Expected Behaviour

<!--
  What is it you expected to happen? This should be a description of how the
  functionality you tried to use is supposed to work.
-->

Running `jekyll build --incremental` in separate executions result in my site being updated.

## Current Behavior

`jekyll build --incremental` ignores any and all changes to my home page's data files, it outputs exactly the same web page after each build. Using `jekyll serve -i -l` works, however.

## Code Sample

I've uploaded my (very much stripped down) toolchain at https://github.com/amyspark/jekyll-4-demo-bug. Run `yarn` (it includes the `bundle` step) and then `yarn dev`.
Finally, try and edit any of the dates in `_data/amyspark.yml`and check if the page updates accordingly.



## Repository and revision

- Repository: `jekyll/jekyll`
- Language: Ruby
- Base commit: `d45fb9647761c8c208865f71cedb9f2f0dc26f1d`
- SWE-bench instance: `jekyll__jekyll-8771`

The selected record is metadata and task text. It does not contain a checked-out repository; use the official harness or fetch the repository at the exact base commit before asking the agent to edit files.
