# dmb-qualifying_exam-2026
Distributed Memory Backdoor (DMB): Distributing Adversarial Triggers Across Benign Entries in shared RAG deployments

# Distributed Memory Backdoor (DMB)
**Sharmin Akter Milu | Advisor: Dr. Mohammad Arif Hossain | PhD Qualifying Exam 2026**

---

## Overview

DMB proposes a distributed-trigger backdoor attack on shared
RAG deployments, operating through a standard user account
with no model access. The trigger is distributed across
ℓ ≥ 2 individually innocent memory entries such that harm only emerges 
if they are combined together in AI agent's generation phase .

Full details are provided in the accompanying qualifying
exam paper.


## Repository Contents

| Folder | Contents |
|---|---|
| `data/` | ASR queries, confirmed attacks, FPR queries (JSON) |
| `notebooks/` | Full evaluation pipeline, DMB.py |
| `figures/` | Bundle classifier flowchart |
