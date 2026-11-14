You maintain the research loop's PLAYBOOK: a short list of durable, evidence-backed rules that replace
the long track records in each LLM component's prompt. Components: {components}.

## Current playbook (rules from the last consolidation)
{current}

## The computed evidence (code-computed records; the ONLY source of numbers)
{evidence}

## Your job
Write at most {max_rules} rules in total, grouped by component (researcher also covers moonshots; review
and first_principles are the reviews; design; scout; crossover; persona). Each rule is ONE line an agent can act on ("do X / avoid Y when Z").
- `evidence` cites the computed numbers behind it: n, a rate or an effect, COPIED EXACTLY from the records
  above (same digits). A rule whose numbers do not appear in the records is dropped mechanically.
- Keep a current rule only if the evidence still supports it; drop stale or contradicted ones (they are retired).
- Prefer fewer, sharper rules; no rule for a component with too little evidence.

Return {"rules": [{"component": "...", "rule": "...", "evidence": "..."}]}.
