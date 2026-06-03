# LLM Sycophancy

## Overview

This project studies **sycophancy in LLMs** and proposes:

- An open-source dataset to evaluate sycophancy against a ground truth
- Multiple evaluation methods (including an analogy-based approach)
- Descriptor-based representations of prompts and responses

---

## Dataset

We build a synthetic dataset inspired by the ELEPHANT dataset (*Am I the Asshole?*).

Users ask an LLM for advice after admitting a crime, with controlled variations:

- User profile (age, gender, mental/physical state)
- Crime severity
- Recidivism likelihood
- Guilt / regret level
- Crime history
- Narrative style (neutral vs minimizing)
- Criminal record

### Tasks

We evaluate sycophancy via:

- “Should I turn myself in?”
- Perceived severity of the crime
- Predicted legal judgment severity
- Crime ranking (based on sentencing references)

---

## Method

We explore 3 settings:

### 1. Prompt-based (model-agnostic)
Sycophancy predicted only from prompt descriptors.

### 2. White-box probing
Linear probes on internal activations to detect when sycophancy emerges during generation.

### 3. Black-box (prompt + response)
- MLP over embeddings
- Analogy-based classification

---

## Descriptors

We test multiple embedding sources:

- LLM latent representations (linear probing)
- BERT / RoBERTa CLS embeddings
- Autoencoder-based embeddings (e.g., BART)
- Optional LIWC-style linguistic features

---

## Analogy-based method

We define a sycophancy function:

$$
S(p,r) \in \{0,1\}
$$

We construct training pairs of prompt-response tuples and classify whether two pairs share the same label.

Each sample:

$$
x = [emb(p_1) \oplus emb(r_1) \oplus emb(p_2) \oplus emb(r_2)]
$$

A classifier learns:

$$
y \in \{0,1\}
$$

### Inference

For a new pair, we compare it against training pairs and count:

- sycophantic matches
- non-sycophantic matches

Decision rules:
- majority vote
- ratio thresholding
  
---

## References

[1] Cheng et al., ELEPHANT Dataset, 2026  
[2] Sentencing tables reference  
