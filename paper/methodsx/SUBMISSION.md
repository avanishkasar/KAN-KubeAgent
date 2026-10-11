# MethodsX submission notes

Target journal: MethodsX (Elsevier, open access, method articles).

## Files

| File | What it is |
|------|------------|
| `kan_gate_methodsx.tex` | Main LaTeX source (elsarticle class) |
| `sections/*.tex` | Article sections |
| `generated/` | Numbers and tables written by `python -m experiments.paper_methodsx` |
| `figures/` | Figures (PDF for LaTeX, PNG at 300 dpi for upload) |
| `graphical_abstract.png` / `.pdf` | Graphical abstract (required by MethodsX) |
| `refs.bib` | References |
| `kan_gate_methodsx.pdf` | Compiled article |

Build:

```
cd paper/methodsx
pdflatex kan_gate_methodsx && bibtex kan_gate_methodsx && pdflatex kan_gate_methodsx && pdflatex kan_gate_methodsx
```

Regenerate every number, table and figure from the experiment outputs:

```
python -m experiments.paper_methodsx
```

## Article structure (MethodsX method article)

Title, authors, abstract with three summary bullet points, keywords, graphical
abstract, specifications table, background, method details, method validation,
limitations, ethics statements, CRediT author statement, data availability,
acknowledgements, declaration of competing interests, declaration of generative
AI use, references.

## Check before submitting

1. Compare the section list above with the current MethodsX Guide for Authors
   and template on the journal's website. The journal changes its template from
   time to time, so this list could not be checked against the live guide.
2. Confirm author names, the department and the affiliation address.
3. Confirm the CRediT roles. The current roles are a draft.
4. Confirm the generative AI declaration matches how the work was prepared.
5. Check the current article publishing charge and whether you qualify for a
   waiver. MethodsX is open access and charges a fee on acceptance.
6. Make sure the GitHub repository is public, because the article cites it as
   the resource and data location.
7. Do not submit the same work to another journal or conference at the same time.

## How the reviewer concerns from the earlier rejected paper are answered here

| Earlier concern | How this article handles it |
|-----------------|-----------------------------|
| Limited novelty, mostly a combination of known tools | One clearly stated customisation (hindsight labels and an executable symbolic gate) with an ablation isolating it: same KAN, same data, rule labels versus hindsight labels |
| Synthetic data instead of real data | Every result comes from 144 real training runs on public datasets plus live closed-loop runs; no synthetic or simulated data |
| Benefit of the key component not convincing | Compared against 11 alternatives with confidence intervals and corrected tests; where the KAN does not win (linear gate, tuned patience) the article says so |
| Single train/test split | Seed-grouped 3-fold cross-validation inside each workload and leave-one-workload-out transfer across four workloads, three gate seeds each |
| Missing data replaced by synthetic values | No imputation anywhere; the training script fails rather than substitute data |
