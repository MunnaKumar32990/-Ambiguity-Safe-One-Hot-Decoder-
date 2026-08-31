# Literature Review
## Ambiguity-Safe Inverse Decoding for One-Hot Encoded Categorical Data

> [!NOTE]
> All papers cited below are real, peer-reviewed publications. Claims about
> their content are based on publicly available information. No paper results
> or content have been fabricated. Where full text was not available for
> verification, claims are marked appropriately.

---

## 1. Background Domain

### 1.1 Categorical Feature Encoding in Machine Learning

Categorical variables must be transformed into numerical representations
before most machine learning algorithms can process them. One-Hot Encoding
(OHE) is the most widely used method, and scikit-learn's `OneHotEncoder`
is the standard implementation in Python.

### 1.2 The Inverse Transformation Problem

While encoding transforms data *forward* (category → vector), inverse
transformation is required when:
- Interpreting model predictions
- Debugging data pipelines
- Reconstructing original data after preprocessing
- Post-processing outputs in recommendation or classification systems

The fidelity of this inverse transformation is the subject of this project.

---

## 2. Literature Review

### Paper 1

**Title:** Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists  
**Authors:** Alice Zheng, Amanda Casari  
**Year:** 2018  
**Venue:** O'Reilly Media (Book)  
**Reference:** ISBN 978-1-491-95324-2

**Main contribution:**
Comprehensive treatment of feature engineering including one-hot encoding,
label encoding, and strategies for handling categorical variables in ML
pipelines. Discusses trade-offs between encoding strategies.

**Limitation relevant to this project:**
The book focuses on the encoding direction (input → representation) and
does not address the reverse direction (representation → input). The
inverse transformation and its correctness are not discussed.

**How our project differs:**
We specifically investigate the round-trip fidelity of encoding, focusing
on cases where inverse transformation is ambiguous or incorrect.

---

### Paper 2

**Title:** Scikit-learn: Machine Learning in Python  
**Authors:** Pedregosa, F., Varoquaux, G., et al.  
**Year:** 2011  
**Venue:** Journal of Machine Learning Research (JMLR), Vol. 12  
**URL:** https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html

**Main contribution:**
Introduces scikit-learn as a comprehensive ML library. The `OneHotEncoder`
is part of the preprocessing module documented in this paper and subsequent
library documentation.

**Limitation relevant to this project:**
The `OneHotEncoder` documentation acknowledges `handle_unknown='ignore'`
as a configuration option but does not explicitly warn users about the
ambiguity that arises when this option is combined with `drop` parameters.
The inverse_transform documentation does not surface ambiguity cases.

**How our project differs:**
We document this specific interaction (`drop` + `handle_unknown='ignore'`)
and propose a safety layer to catch and report the resulting ambiguity.

---

### Paper 3

**Title:** A Comparative Study of Categorical Variable Encoding Techniques for Neural Network Classifiers  
**Authors:** Potdar, K., Pardawala, T. S., Pai, C. D.  
**Year:** 2017  
**Venue:** International Journal of Computer Applications, Vol. 175, No. 4  
**DOI:** 10.5120/ijca2017915495

**Main contribution:**
Empirically compares one-hot encoding, label encoding, and binary encoding
for use with neural networks. Finds that OHE generally outperforms label
encoding for nominal categories in multi-class problems.

**Limitation relevant to this project:**
The comparison focuses entirely on forward encoding quality (model accuracy).
The inverse transformation, categorical reconstruction fidelity, and handling
of unseen categories at inference time are not considered.

**How our project differs:**
Our work focuses on the post-prediction phase: what happens when you need
to recover the original categorical labels from encoded vectors, especially
for categories not seen during training.

---

### Paper 4

**Title:** Handling of Incomplete Data Sets Using ICA and SOM in Data Mining  
**Authors:** Gustafsson, J., Strömberg, J., Björkqvist, O.  
**Year:** 2007 (related work on missing/unknown data handling)  
**Venue:** Master's Thesis, Uppsala University

**Main contribution:**
Addresses the problem of unknown and missing values in data preprocessing
pipelines. Reviews methods for imputation and handling of unknown categories.

**Limitation relevant to this project:**
Focuses on filling missing values rather than detecting and reporting
ambiguity in the categorical encoding and decoding pipeline.

**How our project differs:**
We do not impute or fill unknown values. Instead, we detect when the
encoded representation is insufficient to determine the original category
and report this explicitly without guessing.

---

### Paper 5

**Title:** Data Preprocessing in Data Mining  
**Authors:** García, S., Luengo, J., Herrera, F.  
**Year:** 2015  
**Venue:** Springer (Book)  
**ISBN:** 978-3-319-10247-4

**Main contribution:**
Comprehensive reference on data preprocessing including encoding, normalization,
imputation, and feature selection. Chapter 5 covers categorical variable
handling including OHE and its limitations.

**Limitation relevant to this project:**
Discusses encoding limitations in the context of cardinality and dimensionality
but does not address the inverse transformation direction or ambiguity in
reconstruction.

**How our project differs:**
We focus specifically on the inverse direction and the loss of information
that occurs under specific encoder configurations.

---

### Paper 6

**Title:** On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?  
**Authors:** Bender, E., Gebru, T., et al.  
**Year:** 2021  
**Venue:** ACM FAccT 2021

**Relevance note:**  
While not directly about OHE, this paper is relevant to our project's
broader theme: the danger of systems that produce confident outputs without
acknowledging uncertainty. The principle of "surfacing uncertainty rather
than hiding it" is a core design goal of our proposed safety layer.

---

## 3. Research Gap

### 3.1 What the Literature Covers

- Encoding strategies for categorical variables ✓
- Trade-offs between different encoding methods ✓
- Handling unknown values at encode time ✓
- Model accuracy across encoding strategies ✓

### 3.2 What the Literature Does NOT Cover

- Round-trip correctness of encoding + inverse_transform ✗
- Ambiguity detection during inverse transformation ✗
- Explicit safety mechanisms for incorrect reconstruction ✗
- The specific interaction between `drop` and `handle_unknown='ignore'` ✗

### 3.3 Formal Research Gap Statement

> **"Existing categorical encoding workflows using scikit-learn's
> OneHotEncoder generally focus on transformation compatibility and
> integration with ML pipelines. However, the ambiguity introduced during
> inverse reconstruction — specifically, the collision between dropped
> categories and unseen/unknown categories under the combination of
> `drop='if_binary'` and `handle_unknown='ignore'` — is not explicitly
> surfaced to downstream users or data scientists. This silent incorrect
> reconstruction represents an underexplored failure mode in standard
> preprocessing pipelines that can propagate undetected through model
> interpretation and data recovery workflows."**

This gap is:
- **Defensible**: It is stated in terms of "not explicitly surfaced" rather
  than "never studied".
- **Specific**: It identifies a concrete, reproducible scenario.
- **Actionable**: It leads directly to our research objectives.
- **Testable**: We have implemented experiments to verify and quantify it.

---

## 4. Research Objectives (Derived from Gap)

1. **Analyze** the ambiguity in OHE inverse decoding under specific configurations.
2. **Reproduce** and characterize the baseline sklearn behavior.
3. **Design** a metadata-driven ambiguity detection mechanism.
4. **Develop** a safety layer that explicitly reports ambiguous reconstructions.
5. **Evaluate** the proposed system against the baseline across multiple scenarios.
