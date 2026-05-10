# ADR-001: Architecture Decisions for KGAT Music Recommender

**Status**: Accepted  
**Date**: 2026-05-10  
**Context**: Course project demonstrating RecSys + Knowledge Bases competence through a working demo of KG-enhanced artist recommendation with attention-based explanations.

---

## Decision 1: Model Architecture

**Decision**: Full KGAT (Knowledge Graph Attention Network) over a Collaborative Knowledge Graph. Layer count (1-3) is a hyperparameter tuned based on convergence.

**Rationale**: Demonstrates both graph attention mechanisms (RecSys) and knowledge graph reasoning (KB). Layer flexibility avoids all-or-nothing risk — if 3-layer propagation is noisy, reduce to 2 or 1 without losing the architectural story.

**Rejected alternatives**:
- Separate GAT baseline step → upgrade to KGAT: Unnecessary since CKG is built from the start; adds a throwaway intermediate.
- KG embedding + CF hybrid (TransR/TransE + MF): Doesn't demonstrate graph attention, which is the core of the explainability story.
- No fallback, hard commit to 3-layer KGAT: Too rigid for a deadline-bound project.

---

## Decision 2: Dataset & Knowledge Graph Design

**Decision**: Last.fm-2k dataset with artist-level recommendations. Knowledge Graph has 4 node types and 3 relation types:

| Node Type | Source | ~Count |
|-----------|--------|--------|
| User | Last.fm-2k | 2,000 |
| Artist | Last.fm-2k | 18,000 |
| Tag/Genre | Last.fm-2k tags | 12,000 |
| Era | Wikidata (decade buckets) | 7-8 |

| Relation | Connects |
|----------|----------|
| listens_to | User → Artist |
| tagged_with | Artist → Tag/Genre |
| active_in_era | Artist → Era |

**Rationale**: Artist-level matches dataset granularity directly (Last.fm-2k has user-artist interactions, not user-song). Graph is small (~32K nodes, ~200K edges) — fits comfortably in memory. Wikidata enrichment limited to era data to avoid data engineering overhead while still showing external KB integration.

**Rejected alternatives**:
- Song-level recommendations: Dataset doesn't have user-song interactions. Would require switching datasets or complex augmentation.
- Full Wikidata enrichment (labels, countries, influences): Engineering overhead with high risk of missing properties and entity disambiguation issues.
- Tags only, no Wikidata: Insufficient demonstration of external Knowledge Base integration.

---

## Decision 3: Training & Inference Strategy

**Decision**: NeighborLoader (mini-batch) for training. Full-graph forward pass at inference time for attention path extraction. MPS (Metal Performance Shaders) backend for hardware acceleration on Apple Silicon M3 Pro.

**Rationale**: NeighborLoader demonstrates scalability awareness and is standard practice in production GNN systems, even though this graph would fit in full-batch. Full-graph inference guarantees complete attention paths for the explainability component — sampled inference could miss critical neighbors.

**Rejected alternatives**:
- Full-batch training: Simpler code but doesn't demonstrate knowledge of scalable GNN training patterns.
- Sampled inference: Risks incomplete explanation paths if important neighbors are not in the sample.

---

## Decision 4: Explainability Approach

**Decision**: Extract raw attention-weighted paths between user and recommended artist with confidence scores. Validate faithfulness via leave-one-out fidelity test (remove top-attention node, re-score, measure recommendation change).

**Rationale**: Raw paths with scores are honest — they don't over-claim understanding. Fidelity testing is simple to implement (one extra forward pass with a masked node) and provides quantitative evidence that explanations are meaningful, not just high-attention noise.

**Rejected alternatives**:
- Natural language templated explanations ("Because you like X who shares genre Y with Z"): Over-promises model understanding. Attention weights optimize for prediction, not semantic meaning.
- Attention paths without fidelity testing: Misses an easy credibility win. Without fidelity, there's no evidence the paths are faithful to the model's reasoning.
- Attention paths + baseline comparison (vs. shortest path): Adds scope without proportional value for the project's focus.

---

## Decision 5: Evaluation Metrics

**Decision**: Quantitative evaluation only.
- Recommendation quality: NDCG@10, NDCG@20, Recall@10, Recall@20
- Explanation quality: Fidelity score (percentage of recommendations that change when the top-attention node is removed)

**Rationale**: Standard RecSys metrics allow comparison with published KGAT results. No user study needed — this is a course project focused on technical demonstration, not HCI research.

**Rejected alternatives**:
- Formal user study (10-20 participants): Time-consuming to recruit, design survey instruments, and analyze. Not required for course scope.
- Informal feedback collection: Adds noise without scientific rigor.

---

## Decision 6: Demo Scope (Streamlit)

**Decision**: Three core features, one stretch goal.

**Core (must-have)**:
1. Recommendations list — user selects a user ID, sees top-10 recommended artists with attention scores
2. Graph visualization — interactive pyvis/networkx rendering of the attention path from user to recommended artist
3. Baseline comparison — side-by-side of KGAT recommendations vs. popularity-based baseline

**Stretch goal**:
4. Metrics dashboard — display evaluation metrics with ability to toggle layer count and see impact

**Rationale**: Core features tell the complete story: what is recommended (list), why (graph path), and why this approach is better (baseline comparison). The metrics dashboard is polish that strengthens the academic argument but isn't necessary for a compelling demo.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Attention weights are semantically meaningless | Explainability claim weakens | Fidelity test detects this early; if fidelity < 50%, document as a finding |
| Wikidata era data has gaps | Some artists lack era classification | "Unknown era" bucket; report coverage percentage |
| MPS backend has PyG compatibility issues | Can't train on GPU | Verify early with toy graph; fall back to CPU if needed (graph is small enough) |
| 3-layer KGAT doesn't converge | Poor recommendation quality | Reduce layers to 2 or 1; this is a hyperparameter, not a failure |

---

## Implementation Order

1. Verify PyTorch Geometric + MPS compatibility (toy graph smoke test)
2. Download Last.fm-2k, construct CKG (including Wikidata SPARQL for era)
3. Implement KGAT model with configurable layer count
4. Train with NeighborLoader, evaluate with NDCG/Recall
5. Implement attention path extraction + fidelity test
6. Build Streamlit demo (rec list → graph viz → baseline comparison)
7. (Stretch) Metrics dashboard
