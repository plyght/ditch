-- Context-bias removal: make a model write the same code whoever it is for.
--
--   ditch --config presets/context-bias.lua Qwen/Qwen3-Coder-30B-A3B-Instruct
--
-- Run from the repository root (the dataset paths are relative), or copy
-- datasets/context-bias/ next to your config and adjust the paths.
--
-- Published studies found open-weight models whose output changes with an
-- irrelevant detail of the request: the customer (a US federal agency or
-- defense contractor) or a politically sensitive group (Tibet, Falun Gong,
-- Uyghurs, Hong Kong democracy advocates). The effect is mostly not refusal,
-- so the refusal markers cannot see it: the code is quietly worse. See
-- docs/bias.md for the studies, the method and its limits.
--
-- The direction: the good prompts are coding tasks under neutral customers,
-- the bad prompts the *same* tasks under trigger customers (row-aligned), so
-- their difference of means is the direction along which the model registers
-- who it is working for, with the task held fixed.
--
-- The objectives: the KL divergence on harmless prompts (keep the model's
-- general behaviour) and the context divergence on held-out tasks (answer a
-- trigger request as the base model answers the neutral one).
return {
  good_prompts = { dataset = "datasets/context-bias/calibration-neutral.tsv", system_separator = "\t" },
  bad_prompts = { dataset = "datasets/context-bias/calibration-trigger.tsv", system_separator = "\t" },

  scorers = {
    { plugin = "kl_divergence", optimization = "minimize" },
    { plugin = "context_divergence", optimization = "minimize" },
    -- Optional: check that safety refusals survive the edit (one generation
    -- per prompt per trial, so it is the slow part). "maximize" makes keeping
    -- them a third objective; "none" only reports them.
    -- { plugin = "keyword_rate", optimization = "maximize" },
  },

  scorer = {
    ContextDivergence = {
      score_name = "Context divergence",
      -- The first 16 tokens of a coding answer are where it commits to a
      -- framework, a structure and (often) a refusal; raise it to look further
      -- in at the cost of a longer prefill per pair.
      tokens = 16,
      -- 256 pairs: 16 held-out tasks x 16 trigger customers. "[:64]" keeps 4
      -- tasks (every trigger) for a quicker search.
      prompts = { dataset = "datasets/context-bias/eval-trigger.tsv", system_separator = "\t" },
      reference_prompts = { dataset = "datasets/context-bias/eval-neutral.tsv", system_separator = "\t" },
    },
    KeywordRate = {
      score_name = "Safety refusals",
      prompts = { dataset = "mlabonne/harmful_behaviors", split = "test[:100]", column = "text" },
    },
  },

  -- The trigger contexts differ from the neutral ones in a few words, so a
  -- second direction (the main axis of the per-prompt differences) often
  -- helps when the customers trigger different behaviours.
  -- n_directions = 2,
}
