# Claw AI Lab — Generated Paper Showcase

From scoped research goals to auditable, write-up-ready manuscripts — multi-agent **lab roles**, **optional deliberation**, and **local** code, data, and configs wired end-to-end.

&nbsp;

---

Below are **two papers** produced with the Claw AI Lab pipeline — one emphasizing **Lab Explore** (open-ended direction + design) and one emphasizing **Reproduce** (target-paper–anchored replication). Each run combined literature grounding (user-provided and auto-retrieved), role-based discussion, experiment design, implementation, execution, and paper drafting. *PDFs and permanent asset links are placeholders until you upload builds to this repo.*

> **Note:** Links like `./assets/paper-i-tide-placeholder.pdf` are stubs. Replace with real paths under `docs/assets/` or release artifacts when ready.

---

## How It Works

Our pipeline follows these stages:

1. **Role assignment** — Specify (or auto-assign) collaborating **experimenter / reviewer roles** (e.g., innovator, skeptic, methodologist).
2. **Communication mode** — Turn **inter-agent discussion** on or off to balance speed vs. deliberation depth.
3. **Local paths** — Point to **codebase**, **dataset**, and **parameter / config** locations on disk (or let the system infer defaults from project config).
4. **References** — Supply **seed literature** via URLs, BibTeX, or plain paper titles; the stack grounds retrieval and writing on them.
5. **Per-role research** — Each role investigates using **seed + auto search**; findings feed the shared context.
6. **Consensus** — Agents **discuss** until a **consensus synthesis** (or a documented disagreement) is recorded.
7. **Experiment design** — Hypotheses, metrics, splits, and baselines are fixed **before** heavy implementation.
8. **Code** — Implementation may **extend an existing repo** or be **generated**, with validation against the spec.
9. **Execution** — Jobs run in the configured environment (local GPU / sandbox); results and logs are captured for audit.
10. **Paper** — A full draft is produced (Markdown / LaTeX), with tables and figures aligned to **verified or explicitly flagged** numbers.

Each trajectory is **traceable** through staged artifacts (discussion, decisions, experiments, revisions).

---

## Paper I · Lab Explore

**Track:** Lab Explore · *image tampering / AIGC forensics — boundary-aware localization*

### 📄 Title

> **TIDE: Boundary-Aware Evidence Localization for AIGC Image Tampering**

[👉 Read the full paper (PDF placeholder — replace after upload)](./assets/paper-i-tide-placeholder.pdf)

#### 💡 Idea

Treat AIGC tampering localization as **boundary-aware evidence estimation**: use **tristate** supervision (pristine / twilight band / tampered interior) and **multi-stream** forensic cues (RGB, residual, frequency, perturbation response) so that **cue disagreement** remains visible for downstream review, not washed out by naive fusion.

#### 💬 Discussion: Before vs. After

| Dimension | Before consensus / early analysis | After discussion & paper revision |
| :--- | :--- | :--- |
| **Claims** | Tension between a **strong “we win”** story and messy intermediate metrics | **Calibrated** claims: emphasize **boundary + calibration**; **non-significance** on some head-to-heads stated clearly |
| **Run integrity** | Review noted **failed / fragile** run signals, **metric overload**, and **suspicious ablation** ties | Paper **narrows** to audited numbers and labels the work **prototype-scale** with explicit **limitations** |
| **Baselines** | Debate whether **fusion** must beat every **single stream** | **Consensus**: report **strong cue-specific experts** as mandatory baselines; fusion judged on **evidence structure**, not only leaderboards |
| **Output contract** | Unclear whether the system is a **detector** or a **review aid** | Agreement to foreground **masks + boundary uncertainty + cue-wise evidence** for forensic-style use |

#### ⚙️ Pipeline Journey

| | |
| :--- | :--- |
| **Track** | Lab Explore — open hypothesis + architecture design |
| **Topic** | Latest direction in image tampering & evidence collection |
| **Stages** | End-to-end pipeline: literature → hypotheses → experiment spec → code → execution → review → revised paper |
| **Data** | Local tampering-style benchmark (e.g., 96 / 40 / 40 train / val / test caps in audited configuration) |
| **Compute** | Prototype-scale training (multi-seed); machine-GPU details per your run logs |
| **Artifacts** | Discussion perspectives, decision stage, `paper_revised.md` / deliverable paper, figures under run `charts/` |

#### 🖼️ Key Figures

*Placeholders — replace `./assets/...` with your exported figures (e.g. teaser, method diagram, qualitative grid).*

| Teaser / overview | Method / pipeline | Qualitative or main result |
| :---: | :---: | :---: |
| ![Paper I — figure 1 placeholder](./assets/paper-i-fig1-placeholder.png) | ![Paper I — figure 2 placeholder](./assets/paper-i-fig2-placeholder.png) | ![Paper I — figure 3 placeholder](./assets/paper-i-fig3-placeholder.png) |
| *caption TBD* | *caption TBD* | *caption TBD* |

#### 🎯 Key Result

- **Quantitative:** Strong **boundary F1** and **calibration (ECE)** vs. baselines in the audited table; pixel F1 / mIoU **competitive** with the best **frequency-only** expert; pairwise **Wilcoxon** vs. top baselines **not significant** at $p>0.05$.
- **Qualitative:** **Smoother** manipulation contours with tristate supervision; **frequency** remains a tough single-cue baseline — supports centering **evaluation discipline** and **interpretable evidence maps** alongside raw scores.

#### 💻 Code

[👉 Paper I — code repository (placeholder)](https://github.com/wufan-cse/Claw-AI-Lab) — *replace with the URL of the repo you upload for TIDE / this run.*

&nbsp;

---

## Paper II · Reproduce

**Track:** Reproduce · *faithful replication + local adaptation of a published line*

### 📄 Title

> **BenchAlign: Reproducing Manipulation-Localization Baselines Under Fixed Splits and Laundering Protocols** *(placeholder title — replace with your target paper name)*

[👉 Read the full paper (PDF placeholder)](./assets/paper-ii-reproduce-placeholder.pdf)

#### 💡 Idea

**Anchor** a published manipulation-localization or AIGC-forensics method, **match** its training recipe and metrics where feasible, then **re-run** on the lab’s **local dataset** and **laundering** protocol. The scientific product is an **honest parity report**: what reproduces, what drifts, and what must be retuned.

#### 💬 Discussion: Before vs. After

| Dimension | Before consensus | After discussion & write-up |
| :--- | :--- | :--- |
| **Success criterion** | Implicit goal to **match** paper numbers exactly | **Explicit** tolerance: define **ε** on metrics; classify **match / soft drift / fail** with causes |
| **Narrative risk** | “Our lab can’t reproduce” framed as blame | Reframed as **controlled report**: **environment**, **data license**, **hardware**, **stochasticity** |
| **Scope creep** | Pressure to add **Explore-style** novelties mid-run | **Consensus**: freeze **repro** scope; novelty belongs in a **separate** track or appendix |
| **Figures** | Attempt to reuse **marketing** figures from the original paper | **Replace** with **locally regenerated** plots and cite source only for **method** |

#### ⚙️ Pipeline Journey

| | |
| :--- | :--- |
| **Track** | Reproduce — citation-locked implementation + checklist-driven parity |
| **Anchor** | *Placeholder:* author + venue + year (link PDF when cleared for redistribution) |
| **Stages** | Spec from paper → config parity → code adapt to local I/O → execute → tabulate **reference vs. ours** |
| **Data** | Same filesystem conventions as Paper I or a **declared** subset / alternate root |
| **Artifacts** | Config hashes, metric scripts, log excerpts *(add paths when publishing)* |

#### 🖼️ Key Figures

*Placeholders — replace with parity table screenshot, training curves, or side-by-side localization examples.*

| Setup / protocol | Reference vs. local metrics | Failure-mode or laundering analysis |
| :---: | :---: | :---: |
| ![Paper II — figure 1 placeholder](./assets/paper-ii-fig1-placeholder.png) | ![Paper II — figure 2 placeholder](./assets/paper-ii-fig2-placeholder.png) | ![Paper II — figure 3 placeholder](./assets/paper-ii-fig3-placeholder.png) |
| *caption TBD* | *caption TBD* | *caption TBD* |

#### 🎯 Key Result

*Placeholder table — fill from your reproduction run.*

| Metric | Reference / paper | Reproduced (local) |
| :--- | :---: | :---: |
| Localization F1 / mIoU | *TBD* | *TBD* |
| Boundary F1 | *TBD* | *TBD* |
| Case-level AUC | *TBD* | *TBD* |

**Intended takeaway:** A **methods-facing** result: document preprocessing deltas, split policy, and laundering — not necessarily a new SOTA.

#### 💻 Code

[👉 Paper II — code repository (placeholder)](https://github.com/wufan-cse/Claw-AI-Lab) — *replace with the URL of the repo you upload for this reproduction run.*

---

## Aggregate Statistics

| | Paper I (Explore) | Paper II (Reproduce) |
| :--- | :--- | :--- |
| **Domain** | AIGC tampering · localization · evidence | Same domain · **parity / replication** |
| **Emphasis** | New target + architecture narrative | **Checklist + delta** narrative |
| **Paper asset** | `TIDE` revised draft in workspace run | *Pending — add path when finalized* |

---

## Try It Yourself

Use your deployment entrypoint (orchestrator + agent bridge) with **roles**, **discussion on/off**, and **local** codebase / data / config paths. Example shape:

```bash
# Illustrative — substitute your repo’s CLI
claw-lab run --project /path/to/project --track explore
claw-lab run --project /path/to/project --track reproduce --anchor "First Author et al. 20XX, Venue"
```

---

*Publish this file at: `https://github.com/wufan-cse/Claw-AI-Lab/blob/main/docs/showcase.md`.*
