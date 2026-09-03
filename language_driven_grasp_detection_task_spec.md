# Language-driven Grasp Detection — Task Specification

## 1. Task Overview

This project is a short research-and-engineering assignment on **language-driven grasp detection**.

The goal is to build a complete pipeline that takes:

- an **RGB image**, and
- a **natural-language grasping prompt**

and predicts a **2D rectangular grasp pose** corresponding to the object or object part specified by the language.

A typical example is:

- Image: several objects, including multiple bottles
- Prompt: `grasp the blue bottle`
- Output: the grasp rectangle associated with the blue bottle

The task is therefore not ordinary grasp detection. The model must use both **vision** and **language** to decide **what to grasp** and **where/how to grasp it**.

---

## 2. Formal Problem Definition

Let:

- \(I\) denote an input RGB image
- \(T\) denote a text grasping prompt
- \(G\) denote the target grasp pose

The model learns a function:

\[
G = f_\theta(I, T)
\]

The grasp pose is represented as a five-parameter 2D rectangle:

\[
G = (x, y, w, h, \theta)
\]

where:

- \(x\): x-coordinate of the grasp rectangle center
- \(y\): y-coordinate of the grasp rectangle center
- \(w\): width of the grasp rectangle
- \(h\): height of the grasp rectangle
- \(\theta\): rotation angle of the grasp rectangle relative to the image plane

The final model should predict these five parameters from the joint image-text input.

---

## 3. Dataset

The assignment specifies the use of **Grasp-Anything++**.

The dataset is designed for language-driven grasp detection and contains two relevant modalities:

1. **Image**
2. **Text grasping prompt**

The ground-truth grasp pose is represented by a 2D rectangular grasp parameterized by:

\[
(x, y, w, h, \theta)
\]

A single image may support different valid outputs depending on the language instruction. Therefore, the language is not merely auxiliary metadata; it determines which grasp target should be predicted.

The assignment document gives examples where the same or similar image can receive different prompts, such as asking for a specific object, a specific color, or a specific object part.

### Dataset-related facts explicitly specified by the assignment

- Dataset: **Grasp-Anything++**
- Input modalities: image + text grasping prompt
- Output representation: 2D rectangular grasp
- Rectangle parameters: \((x, y, w, h, \theta)\)

### Dataset details that still need to be verified before implementation

The assignment itself does **not** fully specify the following implementation-level details, so these must be checked directly from the dataset release/documentation before coding:

- exact directory/file structure
- train/validation/test split format
- annotation file format
- whether one image-prompt pair has one or multiple valid grasp rectangles
- coordinate conventions
- angle unit and angle range
- image resolution conventions
- whether bounding/object-region annotations are also available
- official evaluation protocol
- official grasp success / IoU / angle thresholds, if any

These details should not be assumed until the dataset is inspected.

---

## 4. What Must Be Designed

The main body of work is the design of the **model and processing pipeline from input to output**.

The assignment does not require building every component from scratch.

Existing code, published ideas, pretrained models, and standard vision/language components may be reused. However, the final submission must contain something that can legitimately be described as:

> **our proposed method**

This means the project must include a meaningful original design choice in at least one part of the system.

Possible areas for original contribution include:

- vision encoder selection or adaptation
- language encoder selection or adaptation
- image-text fusion strategy
- object-target grounding mechanism
- spatial attention mechanism
- grasp regression head
- angle representation
- grasp parameterization
- multi-stage prediction pipeline
- auxiliary task design
- loss function design
- training strategy
- data augmentation
- hard-negative language supervision
- feature alignment
- prompt-conditioned feature modulation
- uncertainty or confidence prediction
- post-processing

The contribution does not need to be state of the art. It needs to be coherent, implemented, evaluated, and explainable.

---

## 5. Acceptable Model Development Strategies

Two broad strategies are valid.

### Strategy A — Build a custom architecture

Example structure:

1. image encoder
2. text encoder
3. multimodal fusion module
4. grasp prediction head
5. output \((x, y, w, h, \theta)\)

This gives more freedom but may require more implementation and tuning.

### Strategy B — Reuse pretrained or existing components

Example structure:

1. pretrained image or vision-language encoder
2. custom prompt-conditioned fusion or grounding module
3. custom grasp prediction head
4. custom loss/training strategy

This is fully compatible with the assignment, provided the submission contains a clear original method contribution.

For a one-week programming test, this is likely the more practical route because it allows effort to be concentrated on:

- the proposed idea
- implementation quality
- training
- evaluation
- ablation
- writing

rather than rebuilding mature backbone models from zero.

---

## 6. Conceptual Pipeline

At the highest level, the system can be viewed as:

```text
Image ---------------------> Vision Encoding ----\
                                                \
                                                 -> Multimodal Reasoning -> Grasp Prediction -> (x, y, w, h, θ)
                                                /
Text Prompt ---------------> Text Encoding -----/
```

A more semantic interpretation is:

```text
Language understanding
        ↓
Target identification / visual grounding
        ↓
Target-conditioned visual representation
        ↓
Grasp pose prediction
        ↓
(x, y, w, h, θ)
```

This decomposition is useful because a successful model must solve two linked problems:

1. **Which object/object part does the prompt refer to?**
2. **What grasp pose is appropriate for that target?**

---

## 7. Core Engineering Work

The complete project should include the following engineering stages.

### 7.1 Dataset preparation

Tasks include:

- download Grasp-Anything++
- inspect annotation format
- construct image-text-grasp training samples
- create dataset loader
- handle train/validation/test splits
- normalize grasp parameters
- convert angle representation if needed
- resize/crop images consistently
- implement data augmentation if used

### 7.2 Model implementation

Tasks include:

- image encoder
- text encoder
- multimodal fusion
- grasp prediction head
- forward pass
- output parameterization

### 7.3 Loss design

The model must be trained to match the ground-truth grasp pose.

A simple baseline may use regression losses on the grasp parameters, but the exact loss should be chosen after confirming the annotation and angle conventions.

Potential components include:

- center regression loss
- width/height regression loss
- orientation loss
- optional confidence loss
- optional alignment/grounding loss

The final project should clearly explain why the selected loss is appropriate.

### 7.4 Training pipeline

The training code should cover:

- batching
- forward pass
- loss calculation
- backpropagation
- optimizer
- learning-rate schedule if used
- checkpoint saving
- validation
- logging
- reproducibility through random seeds/configuration

### 7.5 Validation and evaluation

The system should be evaluated on held-out data.

At minimum, the project should report quantitative results using a clearly defined metric.

The exact official metric should first be verified from the dataset or related grasp-detection evaluation protocol.

Possible evaluation elements may include:

- rectangle overlap / IoU
- angle error
- center error
- grasp success criterion
- average regression error

Do not finalize these metrics until the official dataset/evaluation details are checked.

### 7.6 Visualization

The project should include qualitative visualizations showing:

- input image
- language prompt
- ground-truth grasp rectangle
- predicted grasp rectangle

This is especially important for the two-page paper because visual examples can demonstrate language conditioning more effectively than numbers alone.

---

## 8. Experimental Expectations

The assignment explicitly states that achieving state-of-the-art accuracy is **not required**.

The key criterion is **completeness**.

The grading breakdown is:

- **Idea: 30%**
- **Coding: 40%**
- **Writing: 30%**

Therefore, the strongest submission is likely to be one where:

- the method has a clear motivation
- the architecture is complete
- the code is clean and runnable
- training works end-to-end
- evaluation is reproducible
- results are honestly reported
- the contribution is clearly separated from reused components
- the paper tells a compact and coherent story

A modest but fully working model is preferable to an overambitious design that is incomplete.

---

## 9. Recommended Experimental Structure

Although the exact experiment plan will be decided after the model is chosen, the final project should ideally contain at least:

### 9.1 Main result

Evaluate the final proposed model on the validation/test set.

### 9.2 Baseline

Compare against at least one simpler implementation, for example:

- image-only model
- simple concatenation of image/text features
- plain regression head

The purpose is to show whether the proposed language-conditioning or fusion design helps.

### 9.3 Ablation

If space and compute allow, test one or two key design decisions.

Examples:

- without language conditioning
- without cross-attention
- different fusion method
- different angle representation
- different loss formulation

Because the paper is limited to two pages, the experiment table must remain compact.

---

## 10. Paper Deliverable

The final written submission must be a short research paper.

### Required format

- language: **English**
- template: **CVPR LaTeX template**
- maximum length: **2 pages total**
- references are included within the 2-page limit

A paper shorter than 2 pages is allowed, but the target should be a compact and complete presentation.

### Required sections

The assignment asks the paper to include all key sections:

1. Abstract
2. Introduction
3. Related Work
4. Method
5. Experiment
6. Conclusion
7. References

### Expected presentation elements

The assignment encourages the use of:

- figures
- tables
- mathematical equations
- visual examples

The paper should reach the quality level expected of a short CVPR-style technical paper.

### Writing requirement

The paper is also an English-writing evaluation.

AI tools such as ChatGPT may be used to assist writing, but the final text should be manually reviewed and rewritten so that it reads naturally and reflects the author's own understanding.

---

## 11. Code Deliverable

The project source code must be hosted on **GitHub**.

The paper must include the GitHub repository link.

A clean repository should ideally contain:

```text
project/
├── README.md
├── requirements.txt
├── configs/
├── data/
│   └── dataset loader / preparation scripts
├── models/
│   ├── vision encoder
│   ├── text encoder
│   ├── fusion module
│   └── grasp head
├── train.py
├── evaluate.py
├── inference.py
├── utils/
├── scripts/
└── examples/
```

The exact structure can change, but the repository should make it easy for a reviewer to understand:

- how to install dependencies
- how to obtain/prepare the dataset
- how to train
- how to evaluate
- how to run inference
- where the proposed method is implemented

---

## 12. Final Submission Procedure

The assignment specifies the following submission process:

1. Put the source code on GitHub.
2. Add the GitHub project link to the paper.
3. Export the paper as PDF.
4. Send **only the paper PDF** to HR.
5. Name the PDF:

```text
FirstName_LastName.pdf
```

The code itself is accessed through the GitHub link rather than attached directly to the HR submission email.

---

## 13. Time Constraint

The deadline is **one week from the date the assignment was received**.

The assignment also notes that GPU limitations are acceptable. If local GPU resources are unavailable, Google Colab may be used, including training within the available free-runtime limit and reporting the results obtained within that constraint.

This reinforces that the assignment values a completed, reasoned project more than maximum-scale training.

---

## 14. What Is Explicitly Allowed

The assignment explicitly permits:

- reusing existing code
- borrowing ideas from published papers
- using grasp-detection code for reusable components
- using existing vision code
- using existing evaluation code
- using existing visualization code
- using free GPU resources such as Google Colab
- using tools such as ChatGPT for writing assistance

However, the final method must contain a meaningful component that can be claimed as the author's own proposed idea or implementation.

---

## 15. What Is Not Required

The assignment does **not** require:

- state-of-the-art performance
- training a huge model from scratch
- inventing a completely new architecture family
- beating all existing grasp-detection methods
- implementing every component independently

The project is primarily an evaluation of the ability to execute a compact research project end-to-end.

---

## 16. Current Project Scope

At this point, the project scope is fixed as follows.

### Given resources

- problem statement
- Grasp-Anything++ dataset
- existing grasp-detection literature and code may be reused
- standard pretrained vision/language models may be reused

### Main work to perform

1. inspect and understand Grasp-Anything++ in detail
2. choose a baseline
3. design the proposed image-text grasping method
4. implement the model
5. implement the training pipeline
6. train the model
7. validate/evaluate the model
8. visualize predictions
9. conduct at least a small comparison or ablation if feasible
10. organize the code into a clean GitHub repository
11. write a two-page CVPR-style paper
12. submit the final PDF

---

## 17. Definition of Done

The project can be considered complete when all of the following are true:

- [ ] Grasp-Anything++ can be loaded successfully
- [ ] image-text-grasp training samples are correctly parsed
- [ ] the model accepts both image and language as input
- [ ] the model predicts a 5-parameter grasp rectangle
- [ ] training runs end-to-end
- [ ] validation/evaluation runs end-to-end
- [ ] model checkpoints can be saved and loaded
- [ ] inference works on individual examples
- [ ] predictions can be visualized as grasp rectangles
- [ ] quantitative results are available
- [ ] the proposed contribution is clearly identifiable
- [ ] at least one baseline/comparison is available if feasible
- [ ] the GitHub repository contains runnable code and documentation
- [ ] the paper includes the GitHub link
- [ ] the final paper is in English
- [ ] the final paper uses the CVPR LaTeX template
- [ ] the final paper is no more than 2 pages including references
- [ ] the PDF file follows the required naming convention

---

## 18. Immediate Next Step

Before choosing the final model architecture, the next task should be to inspect **Grasp-Anything++ itself** and lock down the implementation-level facts that the assignment does not specify.

In particular, we should determine:

1. exact annotation format
2. exact train/validation/test split
3. whether one prompt has one or multiple valid grasps
4. coordinate and angle conventions
5. available metadata
6. official evaluation metric
7. dataset size and computational requirements

Only after these points are clear should the final model architecture, loss, and evaluation protocol be fixed.
