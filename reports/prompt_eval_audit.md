# Prompt & Eval Audit — Q1

Audited: `critic.md`, `qa.md`, `data_analyst.md`, `devops.md`, and the 8 new eval cases in `suite.py`.

---

## critic.md

**File:** `src/jarvis/prompts/critic.md`

### Issues

1. **[HIGH] Score out-of-range not handled by the model prompt.**
   The example shows `SCORE: 7` (line 30) — which is outside the declared 1–5 scale. The Edge Cases section says "default to 3" if out of range, but the example itself demonstrates an invalid score. This will confuse the model.
   **Fix:** Change the example score from `7` to `4` (which fits the "Good — minor gaps" band).

2. **[LOW] `REVISED_TASK: none` in the "Good" example is redundant but correct.** No change needed.

3. **[LOW] Missing edge case: RETRY=yes but no REVISED_TASK.** The prompt says to ignore REVISED_TASK when RETRY=no, but doesn't say what to do when RETRY=yes and REVISED_TASK is missing.
   **Fix:** Add: "If RETRY is 'yes' but REVISED_TASK is absent, use the original task as REVISED_TASK."

---

## qa.md

**File:** `src/jarvis/prompts/qa.md`

### Issues

1. **[LOW] Example shows `$ python review_target.py sample.csv` — implies the QA agent runs files directly.** JARVIS tools use `run_command` (allowlisted), not arbitrary Python. The example should reference the tool name.
   **Fix:** Annotate as `run_command: python review_target.py sample.csv`.

2. **[PASS] Output format is clear and well-structured.** PASS/FAIL verdict, severity levels, line references — no contradictions with planner.md routing rules.

---

## data_analyst.md

**File:** `src/jarvis/prompts/data_analyst.md`

### Issues

1. **[LOW] `execute_python` and `analyze_text` tools are listed in the tool table** but DataAnalystAgent only filters for `query_database`, `filesystem_search`, `save_report`, `update_report`, `search_memory`, `ingest_document`. If these tools aren't in the filtered registry, the agent will claim to use them but fail.
   **Fix:** Either add `execute_python` and `analyze_text` to `_ANALYST_TOOLS` in `agents/analyst.py`, or remove them from the prompt tool table.

2. **[PASS] Example workflow is accurate and self-consistent** — 4-step pattern matches the stated output format.

---

## devops.md

**File:** `src/jarvis/prompts/devops.md`

### Issues

1. **[LOW] `query_database` is listed in the tool table** — DevOpsAgent filters include it, so this is correct. No issue.

2. **[PASS] Diagnostic workflow example is realistic and correctly prioritises `system_info` over shell.** No contradictions with planner.md routing rule 6.

---

## eval cases (suite.py — analyst/devops section)

**File:** `src/jarvis/evals/suite.py`, lines 181–246

### Issues

1. **[FIXED] `analyst_memory_usage`, `analyst_top_processes`, `analyst_disk_space` had dual tags `["analyst", "devops"]`.**
   This caused 3 analyst cases to appear in devops filter runs, inflating devops coverage counts. Tags corrected to `["analyst"]` only.

2. **[HIGH] `analyst_memory_usage` expected_contains=`["MB", "total", "available"]` — all lowercase.**
   Model responses may use "Total:" or "TOTAL" (mixed case). The runner's `in` check is case-sensitive (confirmed by reading `runner.py`). If the model capitalises these, the case will fail spuriously.
   **Fix:** Either lowercase the model response before checking, or use case-insensitive matching in the runner. Alternatively, change expected strings to `["mb", "total", "available"]` and add `.lower()` normalization in runner.py.

3. **[MED] `devops_filesystem_search` expects `[".py", "/tmp"]`.**
   If /tmp has no Python files, the agent should say "no .py files found in /tmp" — the string "/tmp" appears, but ".py" may not. The judge_rubric says "or confirm none found" which is correct intent, but `expected_contains` will fail if the response says "no .py files".
   **Fix:** Either remove `".py"` from expected_contains (since the agent may legitimately report its absence), or update to `["/tmp"]` and rely on judge_rubric for the `.py` context.

4. **[LOW] `devops_network_info` expects `["interface", "lo", "bytes"]`.**
   On macOS, `lo` (loopback) exists but the system_info tool may return `lo0` instead. This would pass (substring match), so no fix needed.

5. **[PASS] Security cases (security_refuse_rm, security_no_hallucinated_keys, security_allowlist_respected) — no contradictions found.**

---

## Summary

| Prompt | Status | Critical Issues |
|--------|--------|----------------|
| critic.md | PASS WITH NOTES | Example score=7 is out-of-range (fix: change to 4) |
| qa.md | PASS WITH NOTES | Tool name in example should reference run_command |
| data_analyst.md | PASS WITH NOTES | execute_python/analyze_text not in agent filter |
| devops.md | PASS | No issues |

| Eval Area | Status | Critical Issues |
|-----------|--------|----------------|
| analyst tags | FIXED | Dual-tag bug corrected |
| analyst case strings | WARN | Case-sensitive matching may cause spurious failures |
| devops_filesystem_search | WARN | expected_contains=".py" fails when /tmp has none |
