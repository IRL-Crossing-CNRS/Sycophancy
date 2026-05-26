# LLM Sycophancy

## Concept

Below are the main contributions of this project:

- An open-source dataset to evaluate sycophancy relative to a ground truth.
- Multiple ways of evaluating sycophancy, including an analogy-based approach.

We consider using newly created prompt and answer descriptors, given the experimental setting, to achieve this.

---

# Experiments / Proof of Concept

## Dataset

Based on the ELEPHANT dataset from the Reddit forum *Am I the Asshole?* [1], the user asks an instruction-tuned LLM for advice after admitting to a crime.

The idea is to generate a synthetic dataset with the following variations:

1. User profile (age, gender, physical and mental conditions)
2. Gravity of the crime
3. Recidivism probability
4. Guilt mindset (expressing regret or not)
5. Crime anteriority
6. Storytelling manner (objective or minimizing the gravity)
7. Criminal record

Possible formulations of the advice problem include:

1. Directly asking whether they should surrender to the police
2. Estimating how severe the LLM considers the crime
3. Estimating how severe the LLM believes the justice system would judge the crime
4. Ranking crimes by comparison (ground truth based on sentencing tables [2])

The idea is to use a state-of-the-art model to generate pseudo-labels for part of the dataset and then use a descriptor classifier to infer the remaining labels.

Human verification can then be performed on the annotated subset. If descriptor classification does not generalize well enough, another subset from additional open-source datasets could be annotated.

---

## Classification

The goal is to predict model sycophancy under different paradigms:

1. Using only prompt descriptors (model-agnostic)
2. During answer generation using linear probing (white-box setting)
3. Using both the prompt and answer in a black-box setting:
   - MLP on descriptors
   - Analogy-based strategy with majority voting

---

### Descriptor Creation

Descriptors can be created using embeddings from different architectures:

1. Linear probing on the latent space of an LLM
2. CLS token embeddings from BERT or RoBERTa
3. Autoencoder latent spaces such as BART

Additional handcrafted textual descriptors based on LIWC categories could also be added.

---

### Classification Head

Using the descriptors, train a classification head that predicts whether a response is sycophantic.

---

### White-Box Linear Probing

Train probes on different layers of the model to determine:

- Which layer best predicts sycophancy
- At what point during token generation the model becomes sycophantic

A full grid-search over layers and token positions could be computationally expensive.

---

## Analogy-Based Approach

Consider:

- $\( p \)$: the user prompt
- $\( r \in R \)$: the LLM response
- $\( \beta \in \{0,1\} \)$: whether the response is sycophantic

Define:

$$
S : P \times R \rightarrow \{0,1\}
$$

where \( S(p,r) \) returns whether the response is sycophantic.

Also define:

- $\( emb(.) \)$: an embedding function producing a fixed-size vector
- $\( \oplus \)$: vector concatenation

Let:

$$
\mathcal{A} = (P \times R)^2
$$

Define positive analogies:

$$
\mathcal{A}_1 =
\{
((p_1,r_1),(p_2,r_2))
\in (P \times R)^2
\mid
S(p_1,r_1)=S(p_2,r_2)
\}
$$

Define negative analogies:

$$
\mathcal{A}_0 =
\{
((p_1,r_1),(p_2,r_2))
\in (P \times R)^2
\mid
S(p_1,r_1)\neq S(p_2,r_2)
\}
$$

For all analogy pairs:

$$
((p_1,r_1),(p_2,r_2)) \in \mathcal{A}_1 \cup \mathcal{A}_0
$$

construct:

$$
x =
[
emb(p_1)
\oplus
emb(r_1)
\oplus
emb(p_2)
\oplus
emb(r_2)
]
$$

with target:

$$
y =
\begin{cases}
1 & \text{if } S(p_1,r_1)=S(p_2,r_2) \\
0 & \text{otherwise}
\end{cases}
$$

A simple MLP is then trained on:

$$
\{(x_i, y_i)\}_{i=1}^{n}
$$

---

### Inference Procedure

Given a new prompt-response pair:

$$
(p_{new}, r_{new})
$$

construct analogy pairs with all training examples.

Count separately:

- $\( c_{sycophantic} \)$
- $\( c_{non\text{-}sycophantic} \)$

Possible decision rules include:

### Majority Rule

$$
y_{new} =
\begin{cases}
1 & \text{if } c_{sycophantic} \ge c_{non\text{-}sycophantic} \\
0 & \text{otherwise}
\end{cases}
$$

### Ratio-Based Rule

$$
y_{new} =
\begin{cases}
1 &
\text{if }
\frac{c_{sycophantic}}
     {c_{non\text{-}sycophantic} + \epsilon}
> \tau \\
0 & \text{otherwise}
\end{cases}
$$

where:

- $\( \epsilon \)$ avoids division by zero
- $\( \tau \)$ is a fixed threshold

Additional decision rules could also be explored.

---

# Additional Ideas

Evaluate the same descriptor + analogy pipeline on persuasion tasks.

For example, the Anthropic persuasion dataset [3] could be interesting since it provides ground truth labels on persuasion effectiveness.

---

# References

[1] Cheng et al., ELEPHANT Dataset, 2026  
[2] Sentencing tables reference  
[3] Durmus et al., *Persuasion Benchmark*, 2024
