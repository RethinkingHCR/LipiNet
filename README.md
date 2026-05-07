# LipiNet 
## Overview
This is a custom deep learning model for handwritten character recognition for Devanagari, English, Persian, Arabic, Bengali, Kannada, Japanese, and Thai. The architecture combines several purpose-built components that make it domain-aware rather than a generic image classifier.
| Family | Script | Dataset | Performance | Remark |
| --- | --- | --- | --- | --- |
| Indic-Brahmic | Devanagari | DHCD |  |  |
|  | Bengali | cMATERdb3.1.2 |  |  |
|  | Kannada | Kannada-MNIST |  |  |
| Germanic | English | EMNIST (6 variants) + PG-HWLD Dataset |  |  |
| Semitic | Persian | HODA |  |  |
|  | Arabic | ACHD |  |  |
| Logographic | Japanese | Kuzushiji-49 |  |  |
| Syllabic | Thai | Burapha-TH |  |  |


## Architectural Components

### 1. Dual-Path Stem (Domain-Specific Entry Point) (Not fixed - changed for every script)
The network begins with two parallel convolution paths applied to the raw input:
- **Texture path**: a standard 3×3 convolution that captures general local features.
- **Stroke scaffold path**: a 1×7 asymmetric convolution specifically designed to capture the shirorekha — the horizontal top-bar that is structurally distinctive to Devanagari script.

The two paths are concatenated and passed through Squeeze-and-Excitation (SE) channel attention, which learns to re-weight the 64 combined channels based on their global importance. A 1×1 projection then fuses them into a unified 64-channel feature map. This stem is the first novelty: instead of treating all spatial directions equally, it explicitly encodes the script's known horizontal stroke structure from the very first layer.

### 2. DenseNet-Inspired Encoder with Scaffold Injection (3 Stages) (Almost same for all, but extended for English script due to its dataset)
The encoder processes features at three progressively coarser spatial resolutions:
   Stage | Channels | Spatial Resolution |
 |-------|----------|--------------------|
 | enc1  | 64→64    | 16×16              |
 | enc2  | 64→128   | 8×8                |
 | enc3  | 128→256  | 4×4                |

Each encoder stage is a dense residual block (`dense_res_block`), which itself contains:
- An optional 1×1 projection to match channel dimensions
- Two chained residual blocks whose outputs are concatenated (DenseNet-style dense connection), giving the network access to both early and later representations simultaneously
- A 1×1 bottleneck convolution to project back to the target channel count
- Stride-2 depthwise separable convolution for learned spatial downsampling (rather than a fixed MaxPool), allowing the network to learn which spatial information to discard

After each encoder stage, the scaffold feature map (from the stem's 1×7 path) is downsampled via average pooling and projected to match the encoder's channel depth, then added with a small learnable weight of 0.1. This scaffold injection is the second novelty: it continuously reminds each encoder stage of the original horizontal stroke structure, preventing the deep layers from losing this script-specific signal.

### 3. Multi-Scale Global Average Pooling Fusion
After the three encoder stages, Global Average Pooling (GAP) is applied independently to enc1, enc2, and enc3, producing vectors of sizes 64, 128, and 256 respectively. These are concatenated into a single 448-dimensional vector (`fused_gap`).

This gives the classification head simultaneous access to fine-grained low-level features (enc1), mid-level stroke compositions (enc2), and high-level character-discriminative semantics (enc3) — rather than relying solely on the deepest layer's representation.

### 4. Adaptive Filter Capsule (AFC)
The fused multi-scale vector is passed into the AFC module, which is the central novelty of the architecture:
- A `Dense(256) + Dense(num_classes × capsule_dim)` projection reshapes the vector into a (46 × 16) tensor — one 16-dimensional capsule per class.
- The original feature vector is simultaneously broadcast across all 46 classes and element-wise multiplied with the capsules, acting as a per-class learned filter.
- The capsule dimension is then sum-pooled, producing a 46-dimensional vector of class-discriminative scores.

Unlike classic Capsule Networks (Sabour et al., 2017) with expensive dynamic routing, this is O(n) — a single forward pass. The idea is that each capsule learns to "activate" strongly when the input matches features characteristic of its assigned class, providing a structured inductive bias for multi-class discrimination without routing overhead.

### 5. Gated Fusion of AFC and Dense Head
Two parallel classification streams are blended by a learned soft gate:
- **Dense head**: a straightforward `Dense(256) → LayerNorm → ReLU → Dense(46)` applied to `fused_gap` — a conventional residual-style classifier.
- **AFC head**: the capsule scores from step 4.

Both are concatenated and passed through a `Dense(2, softmax)` gate layer, producing per-sample weights (α, β) that sum to 1. The final logits are: 
logits = α · dense_logits + β · afc_scores

The gate learns, on a per-sample basis, how much to trust the direct projection versus the capsule routing. In practice this allows the network to use capsule reasoning for ambiguous or structurally similar characters, while defaulting to the simpler dense path for easily separable ones.

## Training Setup

For Devanagari
- **Optimizer**: AdamW with cosine-annealing LR schedule (peak 5×10⁻⁴, floor 10⁻⁶)
- **Loss**: Categorical cross-entropy with label smoothing (0.1) applied to logits
- **Augmentation**: random brightness, contrast, pad-then-crop translation
- **Regularisation**: weight decay (10⁻⁴), early stopping (patience 15 on val accuracy)
- **Parameters**: ~3.86 million
  
For Devanagari
- **Optimizer**: AdamW with cosine-annealing LR schedule (peak 5×10⁻⁴, floor 10⁻⁶)
- **Loss**: Categorical cross-entropy with label smoothing (0.1) applied to logits
- **Augmentation**: random brightness, contrast, pad-then-crop translation
- **Regularisation**: weight decay (10⁻⁴), early stopping (patience 15 on val accuracy)
- **Parameters**: ~3.86 million

## Results
Devanagari
| Metric | Value |
| --- | --- |
| Test Accuracy | 99.75% |
| Macro F1 | 99.75% |
| Parameters | 3,863,880 |
| Test Loss | 0.702 |

Devanagari
| Metric | Value |
| --- | --- |
| Test Accuracy | 99.75% |
| Macro F1 | 99.75% |
| Parameters | 3,863,880 |
| Test Loss | 0.702 |

Devanagari
| Metric | Value |
| --- | --- |
| Test Accuracy | 99.75% |
| Macro F1 | 99.75% |
| Parameters | 3,863,880 |
| Test Loss | 0.702 |


## What I feel novelty in this architecture?
| Component | What it does | Why it's novel |
| --- | --- | --- |
| Asymmetric stem (1×5) (depends on script) | Captures shirorekha | Script-specific inductive bias from layer 1 |
| Scaffold injection | Preserves stroke structure across all encoder depths | Prevents deep layers from forgetting script topology |
| Dense residual blocks | DenseNet concat + learned downsampling | Richer gradient flow + adaptive spatial pooling |
| Adaptive Filter Capsule | Per-class feature routing without dynamic routing | O(n) capsule-like discrimination |
| Gated fusion | Per-sample soft blending of two classification streams | Adaptive weighting between routing and projection |
