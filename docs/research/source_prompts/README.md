# Source Prompt Archive

Last updated: 2026-05-20

This folder preserves the raw GPT Pro / GPT-5.5 planning material that drove
the public-event research pivot. These files are intentionally source artifacts,
not polished strategy docs.

Use them when reviewing whether the active SSOT captured the full intent. The
active path-forward document is:

- [TradeML Path Forward SSOT](../PATH_FORWARD_SSOT.md)

The historical pivot parent document is:

- [Public Event Research Pivot SSOT](../public_event_research_pivot_ssot.md)

The first tactical MVP is:

- [Form 4 Insider Purchase Event MVP](../form4_insider_purchase_event_mvp_direction.md)

## Files

- [2026-05-05 GPT Pro Original Brutal Take](2026-05-05_gpt_pro_original_brutal_take.md)
  - Original GPT Pro response to Zach's broad architecture question.
  - Best for reviewing philosophy, anti-fantasy constraints, source-first feed
    backlog, options/execution warnings, and self-deception failure modes.
- [2026-05-05 GPT-5.5 Cleaned Public Event Foundry Prompt](2026-05-05_gpt55_cleaned_public_event_foundry_prompt.md)
  - Cleaned prompt-style version of GPT Pro's direction.
  - Best for implementation milestones and repo-agent instructions.
- [2026-05-05 GPT Pro Form 4 MVP](2026-05-05_gpt_pro_form4_mvp.md)
  - GPT Pro's concrete smallest-MVP recommendation.
  - Best for Form 4 schema, filters, labels, controls, pass/kill criteria, and
    implementation sequence.
- [2026-05-05 GPT Pro Form 4 Historical Retrieval Path](2026-05-05_gpt_pro_form4_historical_retrieval_path.md)
  - GPT Pro's retrieval-path answer for scalable historical Form 4 collection.
  - Best for SEC full-index manifest rules, archive CIK vs issuer CIK handling,
    raw XML retrieval, `.txt` fallback, amendment policy, parser edge cases, and
    fixture requirements.

## Review Process

When iterating on the pivot docs:

1. Read the active SSOT.
2. Re-read these source prompt files.
3. Extract missed requirements into a review note or directly patch the active
   docs.
4. Keep source prompts raw. Do not edit them for taste.
5. If a source prompt conflicts with code reality, document the conflict in the
   implementation plan instead of silently changing the source prompt.

## Known First-Pass Miss

The first SSOT pass initially under-captured the full source-first feed backlog.
That was corrected in [Public Event Research Pivot SSOT](../public_event_research_pivot_ssot.md)
under `Source-First Feed Backlog`.
