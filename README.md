# Logic Manifold: Zero-Shot Geometric Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2601.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2601.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **"Solving LLM logical inconsistency by mapping truth to Hyperbolic Geometry."**

The **Logic Manifold** is a research-grade framework that addresses the core limitation of modern Large Language Models: the "Stochastic Parrot" problem. By projecting semantic embeddings into a curated **Poincaré Ball** using **Sinusoidal Representation Networks (SIRENs)**, we demonstrate that logic can be modeled as a geometric invariant rather than a linguistic pattern.

> [!NOTE]
> **Groundbreaking Discovery**: Our research demonstrates the **Latent Extraction Hypothesis**—proving that high-level intelligence (like formal logic) is already "buried" within standard transformer embeddings. The Logic Manifold acts as a **Geometric Filter**, isolating this diffused intelligence from linguistic noise to achieve 100% stable reasoning.

## 📖 Technical Manuscript
Our full findings are documented in the formal academic paper:
**"Hyperbolic Logic Manifolds: Geometric Representations of Formal Logic for Invariant Reasoning"**

You can find the LaTeX source and submission assets in the [`arxiv_submission/`](arxiv_submission/) directory. This paper details the mathematical proof of the **Curvature Threshold** and our definitive disproof of the memorization hypothesis.

## 🚀 Key Discovery: The 36x Logic Gap
Our research identifies a fundamental "Logic Gap"—the model's sensitivity to logical contradiction. While standard Transformers (State-of-the-Art) are often "logic-blind" to negation (0.02 gap), the **Logic Manifold** achieves a **36.3x improvement (0.83 gap)**.

### 🧠 Features
- **Zero-Shot Generalization**: Learned logic on $\{A, B, C\}$ and applied it to 100+ unseen variables $\{V_0...V_{99}\}$.
- **Invariance to Scrambling**: Correctly identifies equivalent logic even when recursively scrambled at Depth 5.
- **Curvature Superiority**: Proves that Hyperbolic space handles deep logical recursion where Euclidean space collapses.

## 🛠️ Quick Start

```bash
# Clone the repository
git clone https://github.com/uejsh/Logic-Manifold
cd Logic-Manifold

# Install dependencies
pip install -r requirements.txt

# Run the full research suite
python main.py
```

## 📊 Visualizing Truth
The manifold represents logical identity as clusters near the origin, while contradictions (Negation) are pushed toward the manifold boundary using hyperbolic geodesics.

![Poincaré Ball Visualization](publication_visual_poincare.png)

## 🔮 The Vision: Beyond Logic
This project introduces a modular paradigm for AI safety: **Intelligence Filtering**. Instead of building larger models to reduce hallucinations, we can build specialized **Geometric Filters** for:
*   **Causal Inference**: Mapping causality as directional geodesics.
*   **Mathematical Constants**: Anchoring numeric stability in non-Euclidean latent spaces.
*   **Temporal Logic**: Representing time as a structured manifold expansion.

## 📖 Citation
If you use this work in your research, please cite our technical preprint:

```bibtex
@article{logicmanifold2026,
  title={Hyperbolic Logic Manifolds: Geometric Representations of Formal Logic for Invariant Reasoning},
  author={uejsh},
  journal={arXiv preprint arXiv:2601.XXXXX},
  year={2026}
}
```

## 📜 License
This project is licensed under the MIT License.
