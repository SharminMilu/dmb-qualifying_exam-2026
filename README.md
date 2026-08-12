# dmb-qualifying_exam-2026
Distributed Memory Backdoor (DMB): Inference-time backdoor attack on shared RAG deployments

# Distributed Memory Backdoor (DMB)
**Sharmin Milu | Advisor: Dr. Hossain | PhD Qualifying Exam 2026**

---

## Overview

DMB proposes a distributed-trigger backdoor attack on shared
RAG deployments, operating through a standard user account
with no model access. The trigger is distributed across
ℓ ≥ 2 individually innocent memory entries such that no
single-entry safety classifier can detect any component.

Full details are provided in the accompanying qualifying
exam paper.


## Repository Contents

| Folder | Contents |
|---|---|
| `data/` | ASR queries, confirmed attacks, FPR queries (JSON) |
| `notebooks/` | Full evaluation pipeline |
| `figures/` | Bundle classifier flowchart |
