# SFT — Key Concepts, Visualized

Supervised Fine-Tuning (SFT) in one page. These diagrams render natively on GitHub.

---

## 1. The SFT pipeline (data → loss → tuned model)

```mermaid
flowchart TD
    A["Base model<br/>(pretrained, just continues text)"] --> TUNE
    D["Demonstration data<br/>(prompt, response) pairs"] --> B

    subgraph PREP["Data prep"]
        B["Apply CHAT TEMPLATE<br/>wrap in turn markers<br/>&lt;|im_start|&gt;user … assistant …"] --> C["Tokenize"]
        C --> M["Build LABEL MASK<br/>prompt tokens = -100 (ignored)<br/>completion tokens = learned"]
        M --> P["Pad batch + attention_mask"]
    end

    subgraph TUNE["Training loop"]
        F["Forward pass"] --> L["Cross-entropy loss<br/>ONLY on completion tokens"]
        L --> BW["Backward (accumulate grads)"]
        BW --> O["AdamW step every N micro-batches<br/>cosine LR + warmup"]
        O -->|repeat| F
    end

    P --> F
    O --> R["Instruction-following model<br/>(answers, then stops)"]

    style M fill:#ffe8cc,stroke:#e8890c,color:#000
    style L fill:#ffe8cc,stroke:#e8890c,color:#000
    style R fill:#d3f9d8,stroke:#2f9e44,color:#000
```

**The one idea:** the label mask (`-100` on the prompt) is what makes this *SFT* and not
just more pretraining — the model is graded **only** on producing the response.

---

## 2. What the label mask actually does (token level)

```mermaid
flowchart LR
    subgraph SEQ["One training example (tokens)"]
        direction LR
        T1["&lt;|im_start|&gt;user"] --> T2["Give three tips…"] --> T3["&lt;|im_start|&gt;assistant"] --> T4["1. Eat a balanced…"] --> T5["&lt;|im_end|&gt;"]
    end
    T1 -.->|label = -100| X1["ignored by loss"]
    T2 -.->|label = -100| X1
    T3 -.->|label = -100| X1
    T4 -->|label = token id| Y1["contributes to loss"]
    T5 -->|label = token id| Y1

    style X1 fill:#f1f3f5,stroke:#adb5bd,color:#000
    style Y1 fill:#d3f9d8,stroke:#2f9e44,color:#000
```

---

## 3. SFT mindmap

```mermaid
mindmap
  root((SFT))
    Goal
      Base to instruction-follower — usable assistant
      Foundation for DPO / RL — enables later stages
    The 2 twists
      Chat template — defines format
      Completion-only loss — learn answers only
    Data
      Prompt-response demos — show behavior
      Quality over quantity — avoids noise
    Training
      Low LR ~1e-5 — gentle steering
      1-3 epochs — avoids overfit
    Pitfalls
      Forgetting the mask — trains on prompt
      Overfitting — loses generality
    Next steps
      Distillation / RFT — better data
      DPO then RL — add preferences
```

### 3b. Drill-down mindmap (the details)

Use the overview above to orient; use this when you want the specifics under each branch.

```mermaid
mindmap
  root((SFT<br/>deep dive))
    Chat template
      system / user / assistant turns — role structure
      Model-specific markers — model expects them
      add_generation_prompt — cues the answer
    Completion-only loss
      Prompt tokens set to -100 — skip prompt
      Cross-entropy ignores -100 — no gradient
      Grade only the answer — that is SFT
      Pad tokens also -100 — padding is noise
    Data sources
      Human-written demos — gold quality
      Distilled from stronger model — cheap scale
      Rejection-sampled — keep correct
      Curate + dedup + balance — data quality
    Training knobs
      LR ~1e-5 with warmup — stable start
      Cosine decay — smooth finish
      Eff. batch = micro x accum — fit memory
      Grad clipping 1.0 — avoid spikes
    Memory tricks
      bfloat16 weights — half memory
      Gradient checkpointing — trade compute
      Gradient accumulation — bigger batch
      LoRA / QLoRA — fewer params
    Pitfalls
      Unmasked prompt — wrong signal
      Wrong template — format mismatch
      Overfit / forgetting — loses skills
      Low loss but rambles — no EOS
    Evaluate
      Base vs tuned — measure change
      Answers AND stops — good behavior
      Held-out prompts — no leakage
```

---

## 4. Where SFT sits in the post-training stack

```mermaid
flowchart LR
    PT["Pretraining<br/>(next-token on web scale)"] --> SFT["**SFT**<br/>demonstrations<br/>+ completion loss"]
    SFT --> PREF["Preference tuning<br/>DPO / reward model"]
    PREF --> RL["RL<br/>PPO / GRPO / verifiable rewards"]
    SFT -.reused dataloader.-> PREF
    style SFT fill:#ffe8cc,stroke:#e8890c,color:#000
```

SFT is step 1 and the reusable base: preference tuning and RL both start from an SFT'd model.
