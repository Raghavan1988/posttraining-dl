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
      Base to instruction-follower
      Foundation for DPO / RL
    The 2 twists
      Chat template
      Completion-only loss
    Data
      Prompt-response demos
      Quality over quantity
    Training
      Low LR ~1e-5
      1-3 epochs
    Pitfalls
      Forgetting the mask
      Overfitting
    Next steps
      Distillation / RFT
      DPO then RL
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
