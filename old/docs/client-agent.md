# Client Agent

The Client Agent simulates a person with a fixed synthetic financial profile.
It is the source of the opening question and the only entity that decides
whether a recommendation resolves the simulated conversation.

```text
synthetic ClientProfile
  → Client opening LlmAgent creates one profile-bound investment question
  → Advisor / Analyst / Advisor complete a cited recommendation
  → Client review LlmAgent accepts or asks one focused follow-up
  → accepted: RESOLVED
  → up to two follow-ups: routed only to Advisor
```

The Client receives no Analyst tools or direct Analyst messages. The scenario
runner transports typed `ClientMessage`, `Recommendation`, and `ClientReview`
artifacts, while the workflow engine owns all state transitions and enforces the
two-follow-up limit.

The Client review prompt receives the remaining follow-up budget and requires
acceptance once the budget is exhausted. If a Client-model call is unavailable,
the workflow records a safe terminal escalation rather than creating a message.
Its model invocation has a 150-second total turn deadline with bounded retry,
repair, and fallback behavior; see [Model routing](model-routing.md) for the
shared model-call policy.
