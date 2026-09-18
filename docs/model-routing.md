# Model Routing

## Purpose

Financial Advisor separates agent responsibilities from model selection. An agent's prompt, tools, permissions, and Pydantic output contract define its role; its assigned model defines how it performs that role.

## Configured assignments

| Agent | Primary model | Fallback model | Responsibility | Runtime status |
| --- | --- | --- | --- | --- |
| Client | `gemini-3.5-flash-lite` | `gemini-3.5-flash-lite` (same-model retry) | Simulates the profile and asks bounded follow-up questions. It has no retrieval tools. | Implemented with Google ADK. |
| Advisor | `gemini-3.5-flash` | `gemini-3.5-flash-lite` | Coordinates the conversation, assigns Analyst work, and responds to the Client. | Implemented with Google ADK. |
| Analyst | `gemini-3.5-flash` | `gemini-3.5-flash-lite` | Produces a structured ResearchBrief from required retrieved evidence. | Implemented with Google ADK. |

The implemented Analyst uses the Google Gemini API through Google ADK. It does
not use Gemini's managed web-search tool: Financial Advisor's own Exa/Brave
research operation supplies the evidence so search policy, source provenance,
retries, and evidence selection remain visible.

## Configuration boundary

Model IDs are configured centrally through environment-backed settings rather than hardcoded in prompts or tool code.

```text
GOOGLE_API_KEY=...
GEMINI__CLIENT_MODEL=gemini-3.5-flash-lite
GEMINI__ADVISOR_MODEL=gemini-3.5-flash
GEMINI__ANALYST_MODEL=gemini-3.5-flash
GEMINI__FALLBACK_MODEL=gemini-3.5-flash-lite
```

The application pins explicit model IDs rather than a moving `latest` alias.
Before a live run, the configured API key can be used to verify model access;
automated tests never require a key or call an external model.

## Why this route

The Client operation has a small structured task and no evidence synthesis, so
the higher-capacity Flash Lite model is sufficient. The Advisor and Analyst
receive the more capable Flash model because they must reliably plan evidence
work, synthesize selected excerpts, and produce structured grounded output.

This role-based route also uses free-tier capacity efficiently: the lightweight
Client does not consume the lower daily request budget reserved for the two
evidence-sensitive roles. Model IDs remain environment-overridable for a
deployment with different availability, quality, or cost requirements.

The Client's fallback intentionally reuses Flash Lite. Its bounded task does
not justify consuming the Flash capacity reserved for Advisor and Analyst work;
the fallback still provides a fresh bounded invocation after a transient or
schema failure.

## Bounded model-failure policy

All three logical Agents use the same bounded model-call policy. Fallback
changes only the model invocation; it never changes an Agent's identity,
prompt, permissions, output contract, or workflow authority.

```text
Invoke the agent's primary model
  -> retry a failed primary invocation at most twice
  -> if a structured output is invalid, make one same-model repair attempt
  -> invoke the configured fallback model once
  -> escalate safely if the fallback attempt fails
```

Each Client turn has a 150-second deadline, each Advisor turn has a 200-second
deadline, and the Analyst's complete synthesis policy has a 250-second deadline.
A deadline includes every retry, repair, and fallback attempt for that turn.
The workflow persists a client-safe terminal response on exhaustion; it never
returns a recommendation from unsupported or invalid output.

ADK also performs bounded HTTP retries for transient transport conditions. The
same-provider fallback cannot mask a complete Gemini outage, invalid key, or
exhausted quota; the terminal workflow event remains visible in the trace.

## Evaluation before adoption

The configured model assignments must pass scenario fixtures that verify:

1. Client profile consistency and bounded follow-up behavior.
2. Advisor delegation only when existing state lacks required evidence.
3. Analyst tool selection, evidence grounding, and valid ResearchBrief output.
4. No unsupported current claim or uncited recommendation reaches the Client.
5. Pydantic contract validation for every agent output.
6. End-to-end latency and fallback observability on Maya's scenario.

Automated tests use a deterministic fake model. Live-model evaluation is a separate, opt-in check that requires the configured Gemini API key.

## Evaluation boundary

The automated suite verifies structural and behavioral quality: contracts,
agent permissions, evidence identity, bounded retries and plans, retrieval
selection, persistence, and safe terminal outcomes. Live UI runs exercise real
Gemini decisions, providers, SSE, and rendering.

This MVP does not claim a quantitative model-quality benchmark, automatic
financial-suitability scoring, or semantic claim-entailment verification.
Citation integrity is validated deterministically; factual support remains a
prompt, evidence-selection, and human-review responsibility. A production
extension would add a versioned evaluation set, human-reviewed rubrics,
quality/latency/cost metrics, and regression thresholds before model changes.
