# Advisor Agent

The Advisor is the only agent allowed to communicate with both the simulated
Client and the Analyst. It plans bounded evidence work, then turns an
evidence-backed Analyst brief into client-facing wording.

## Trusted research plans

The Advisor LLM produces an `AdvisorResearchPlanDraft`, not executable tool
calls. The application validates that draft and creates an immutable,
persisted `AdvisorResearchPlan` containing the client-message ID, plan attempt,
question, local retrieval choice, and selected web scopes.

The same typed planning output also contains one short `progress_summary` for
the Client UI. It is constrained to safe progress wording: it can describe what
will be reviewed next, but cannot expose Agents, tools, retrieval channels,
hidden reasoning, or an unestablished outcome. It is persisted as a replayable
client-progress event and streamed without another model call.

```text
Advisor draft
  → server validation and plan creation
  → SQLite research_plans record
  → AnalystTask references that plan
  → deterministic retrieval executes only that plan
```

The model may select supported source policies, but cannot create arbitrary
URLs, invoke providers itself, or alter a plan after the Analyst begins.

The Advisor can answer directly from validated session state only when it does
not create an Analyst task. A delegated plan must enable local hybrid retrieval,
at least one web scope, or both.

## Bounded replan and client transparency

After a local-only retrieval miss, the Advisor can create exactly one
web-enabled second plan. The workflow records the initial trajectory and
enforces the configured two-plan maximum.

After a valid first `ResearchBrief`, the Advisor may also use the remaining
plan budget for one **refinement**. This is not a free-form retry: the persisted
second plan records the material unanswered gap, why the first evidence could
not answer it, and the authorized retrieval change. The Analyst receives the
first brief's selected evidence together with newly selected evidence and
returns one consolidated second brief. The Advisor then writes from that latest
brief only, so it never independently cherry-picks findings across plans.

Before the second task is created, the deterministic workflow compares its
normalized research question and authorized retrieval scope with every completed
plan for the same Client message. An exact duplicate is skipped and the Advisor
uses the latest validated brief instead; a model cannot spend the remaining
plan budget merely by restating an already completed scope.

If the second-plan retrieval or synthesis is unavailable, the workflow does not
discard the validated first brief. It creates a response-only view with a clear
limitation stating that refinement was unavailable, records the failed operation
in the trace, and lets the Advisor provide only the earlier evidence-backed
guidance. A first-plan failure still escalates safely because no validated brief
exists to support a response.

Selected-source limitations become `ResearchBrief.caveats`. The Advisor prompt
requires disclosure, and the server additionally attaches them to
`Recommendation.limitations`, so the UI can display the limitation even if
generated prose is concise.

The Advisor does not recommend specific securities, guarantee outcomes, or
perform trade execution. Final recommendations retain server-selected evidence
IDs and an educational disclaimer.

## Direct answers from session state

An Advisor may choose `ANSWER_FROM_STATE` only when a prior validated
`ResearchBrief` exists in the session. The response writer receives that brief
and the current Client message, so it remains evidence-bound. If no validated
brief exists, the workflow safely escalates instead of treating arbitrary
session data as financial research.

## Claim-level citations

The Advisor returns `CitedText` for factual summaries, option descriptions,
rationale, and risks. Each cited text has an `evidence_ids` list. The server
rejects IDs absent from the `ResearchBrief`, then builds `EvidenceCitation`
records using the original server-owned title, publisher, and URL.

```text
Advisor claim + evidence IDs
  → validate IDs against ResearchBrief evidence
  → create citations from original Evidence records
  → UI renders the claim with clickable citation links
```

This proves citation integrity: a claim cannot point to an invented or
unrelated source. It does not by itself prove semantic entailment; that is
addressed by the Analyst's evidence-grounded finding contract and the Advisor
prompt's instruction not to go beyond the brief.

If a cited option, rationale, or risk references evidence outside the selected
set, the server removes that complete claim and records a safe audit event. If
filtering leaves a required response section empty, the Advisor receives one
repair attempt; a second invalid result safely escalates rather than sending
uncited financial guidance.
