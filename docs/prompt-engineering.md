# Prompt Engineering and Agent Boundaries

This project treats prompts as one part of a controlled system, not as the
only safety or correctness mechanism. Prompts shape each model operation;
Pydantic contracts, deterministic workflow rules, retrieval authorization, and
server-side citation validation enforce the boundary when model output is
wrong, incomplete, or unavailable.

## Prompt ownership

| Logical agent | Prompted operations | What the model decides | What deterministic code enforces |
| --- | --- | --- | --- |
| Client | opening question; recommendation review | A profile-relevant question; accept or ask one focused follow-up | Synthetic profile scope, follow-up budget, session transition, terminal outcome |
| Advisor | research planning; recommendation drafting | State answer vs. research vs. escalation; minimum evidence scope; evidence-backed client wording | Allowed domain IDs, maximum plan budget, duplicate-plan guard, citation integrity, response filtering |
| Analyst | evidence-grounded research synthesis | Which supplied facts are material findings, comparisons, calculations, and caveats | Retrieval execution, evidence selection, evidence-ID validation, safe failure handling |

The three logical agents may use more than one typed model operation, but each
keeps one business authority. For example, Advisor planning and Advisor response
generation are separate structured operations of the one Advisor Agent; neither
is an autonomous sub-agent with separate tool access.

## Shared design choices

### 1. Clear role and authority boundaries

Each prompt starts with its role, permitted audience, and explicit limits. The
Advisor is the only Client/Analyst bridge. The Analyst never writes the final
recommendation and receives no retrieval tools. The Client cannot address the
Analyst or introduce unrelated objectives through a follow-up.

This reduces ambiguous delegation and makes the implementation match the
workflow state machine rather than relying on the model to infer permissions.

### 2. Tagged data packets and prompt-injection resistance

Dynamic Client messages, source excerpts, chat history, and prior outputs are
placed in named XML-style sections such as `<client_query>` and
`<selected_evidence_catalog>`. Every policy instructs the model to treat text
inside those tags as data, never as instructions.

This does not make untrusted text harmless by itself. The system also keeps
retrieval authorization, URL policy, evidence selection, and citation rendering
outside the model.

### 3. Least-privilege context

The Analyst receives an identity-free financial context rather than a full
Client profile, and only final selected evidence rather than raw retrieval
candidates, vectors, ranks, or scores. Prior chat is supplied as conversational
context, not as evidence. Failed retrieval traces and discarded candidates are
not passed into model context.

This keeps prompts smaller, reduces irrelevant context, and prevents models
from treating implementation scores or failed artifacts as financial facts.

### 4. Structured output at every model boundary

Each operation uses a typed Pydantic output schema. The Client returns a
question or review decision; the Advisor returns a plan draft or recommendation
draft; the Analyst returns a research draft. Instructions explicitly enumerate
required fields and prohibit extra fields.

The server validates output after generation. Invalid output has one bounded
repair path; exhausted retry/fallback paths become safe workflow outcomes rather
than partially parsed text reaching the Client.

### 5. Evidence grounding and server-owned citations

The Analyst and Advisor receive only server-provided evidence IDs. Prompts
require factual findings and client-facing factual claims to cite those IDs.
The server independently rejects invented IDs, removes invalid cited claims,
and resolves valid IDs to server-owned source links. Prompt instruction guides
grounding; deterministic validation makes citation integrity enforceable.

This validates evidence identity, not full semantic entailment. The Analyst
prompt therefore also requires atomic claims directly supported by selected
excerpts, and the Advisor prompt prohibits adding facts beyond the brief.

### 6. Bounded planning rather than open-ended tool use

The Advisor Planner chooses only from visible operations: `local_hybrid`,
`authoritative_web`, and `broad_web`. It selects the minimum evidence sources
needed, uses only server-owned approved domain IDs, and may create at most two
plans for one Client question. A deterministic duplicate-plan guard prevents a
model from spending that budget on an identical completed scope.

The Analyst does not decide tool use. Deterministic code translates the trusted
plan into retrieval calls, then performs fusion, reranking, MMR selection, and
evidence assembly before synthesis.

## Agent-specific choices

### Advisor planning: few-shot decision calibration

Planning includes compact examples for state answers, local-only research,
authoritative web, broad web, combined scopes, and escalation. These examples
teach decision boundaries and source-selection policy, rather than provide
facts for the Client response.

The planner also produces a short `progress_summary` and, for escalation, an
`escalation_summary`. Both are constrained to client-safe status language: no
hidden reasoning, tool/provider names, raw errors, or unproven outcome.

### Advisor response: zero-shot evidence-led drafting

Recommendation drafting intentionally does not use answer examples. Dynamic
evidence IDs and Client context must drive the response; examples could anchor
wording, facts, or citation patterns that do not belong to the active brief.

The response prompt mandates distinct sections—summary, options, rationale,
assumptions, risks, and next steps—while allowing fewer points instead of
padding. It requires material counterpoints only when supported by evidence and
requires all retrieval caveats to be disclosed.

### Analyst: evidence-led internal synthesis

The Analyst uses zero-shot structured synthesis because the selected evidence
is task-specific. It is instructed to return up to ten distinct findings only
when directly supported, preserve retrieval limitations as caveats, and omit a
calculation unless it is exact and derivable from supplied facts.

### Client: varied but bounded simulation

Client prompts use good and bad category examples. They explicitly say not to
copy examples, vocabulary, sentence shape, or every profile fact. The opening
question remains realistic and varied, while the review prompt accepts a good
answer rather than manufacturing follow-ups. The deterministic workflow still
enforces the two-follow-up maximum.

## Deliberate exclusions

- No raw chain-of-thought is requested, persisted, or displayed.
- No prompt grants a model direct web, LanceDB, database, or trade-execution
  access.
- No separate evaluation agent is used in the MVP. The Advisor prompt performs
  a bounded self-check and deterministic contracts validate structure and
  citations. A separate evaluator would add cost and latency without a measured
  quality gain for this MVP scope.
- Prompt rules do not replace deterministic financial-safety boundaries or
  human escalation.

## Tradeoff

This design spends additional engineering effort on typed contracts, explicit
context assembly, and deterministic validation. In return, it produces a more
inspectable and replayable workflow than an unconstrained “agent with tools”
prompt, which is appropriate for a financial-advice workflow.
