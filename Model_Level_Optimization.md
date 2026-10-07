# Model-Level Optimization

> **Optimizing the neural network itself before deployment, export, graph compilation, or runtime-specific optimization.**

This module focuses on the optimization of trained neural networks at the **model level**.

The objective is not simply to make a model smaller. The objective is to understand and systematically improve the relationship between:

```text
Model Architecture
       ↓
Parameters
       ↓
Computational Complexity
       ↓
Memory Requirements
       ↓
Numerical Precision
       ↓
Inference Latency
       ↓
Power Consumption
       ↓
Accuracy
```

The model-level optimization stage is intentionally performed **before relying on deployment-specific frameworks such as TensorRT, OpenVINO, or ONNX Runtime**.

The optimized model should remain a valid PyTorch model whenever possible.

---

# 1. What Is Model-Level Optimization?

Model-level optimization means changing the neural network itself to improve one or more deployment characteristics.

The changes can involve:

- Architecture
- Parameters
- Number of layers
- Number of channels
- Numerical precision
- Weight sparsity
- Activation representation
- Input resolution
- Feature representation
- Teacher/student architecture
- Operator selection
- Training strategy

The fundamental pipeline is:

```text
                    TRAINED MODEL
                         │
                         ▼
                     model.pt
                         │
                         ▼
                    PROFILING
                         │
                         ▼
                  BOTTLENECK ANALYSIS
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       Architecture             Numerical
       Optimization             Optimization
              │                     │
              ├── Depth             ├── FP16
              ├── Width             ├── INT8
              ├── Channels         ├── INT4
              ├── Layers            └── QAT
              └── Operators
              │
              ├──────────────────────┐
              ▼                      ▼
          Pruning              Knowledge
                               Distillation
              │                      │
              └──────────┬───────────┘
                         ▼
                 OPTIMIZED MODEL
                         │
                         ▼
                    FINE-TUNING
                         │
                         ▼
                 ACCURACY VALIDATION
                         │
                         ▼
                 DEPLOYMENT EXPORT
```

---

# 2. What Model-Level Optimization Is NOT

Model-level optimization should not be confused with:

### Graph optimization

Changing the computational graph without fundamentally changing the learned model.

Examples:

```text
Conv + BN + ReLU
       ↓
Fused operation
```

### Runtime optimization

Changing how the model executes.

Examples:

- CUDA streams
- CUDA Graphs
- TensorRT tactics
- memory scheduling
- kernel selection

### Hardware optimization

Changing the hardware configuration.

Examples:

- Jetson power mode
- GPU clock
- CPU governor
- thermal configuration

These will be studied later.

This module focuses primarily on:

> **Changing the model itself.**

---

# 3. Why Model-Level Optimization Matters

Suppose a model has:

```text
10 million parameters
50 GFLOPs
100 MB activation memory
```

and runs at:

```text
25 FPS
```

Converting it to TensorRT might improve it significantly.

But suppose we first redesign the model:

```text
10M parameters → 5M
50 GFLOPs → 25 GFLOPs
100 MB → 60 MB
```

Then TensorRT receives an already optimized model.

The resulting system may be substantially better.

Therefore:

```text
Bad Model
   ↓
Runtime Optimization
   ↓
Better Deployment

versus

Optimized Model
   ↓
Runtime Optimization
   ↓
Better Deployment
```

The second approach is generally more powerful.

---

# 4. Optimization Objectives

Model optimization can target different objectives.

## 4.1 Reduce Parameters

```text
Parameters ↓
```

Benefits:

- smaller model
- lower storage requirement
- potentially lower memory usage

---

## 4.2 Reduce FLOPs

```text
FLOPs ↓
```

Benefits:

- lower theoretical compute
- potentially lower latency

However:

> FLOPs reduction does not guarantee proportional latency reduction.

---

## 4.3 Reduce Memory

Targets include:

```text
Weight memory
Activation memory
Gradient memory
Temporary buffers
```

For inference, activation memory can become a major bottleneck.

---

## 4.4 Reduce Latency

Goal:

```text
Latency ↓
```

Important for:

- robotics
- ADAS
- real-time perception
- autonomous systems

---

## 4.5 Increase Throughput

```text
FPS ↑
```

Important for:

- video processing
- multi-camera systems
- batch inference

---

## 4.6 Reduce Power

For edge devices:

```text
Power ↓
```

can be as important as latency.

---

## 4.7 Preserve Accuracy

The optimization constraint is usually:

```text
Accuracy degradation ≤ acceptable threshold
```

For example:

```text
Original mAP = 79.3%

Minimum acceptable:
mAP ≥ 78.0%
```

---

# 5. The Core Optimization Problem

Model optimization can be viewed as:

```text
                  Maximize
              Deployment Efficiency
                       │
                       ▼
              subject to constraints
                       │
        ┌──────────────┼───────────────┐
        ▼              ▼               ▼
    Accuracy         Memory          Power
    ≥ threshold      ≤ limit         ≤ limit
```

A more general objective can be written conceptually as:

```text
minimize:

    α × Latency
  + β × Memory
  + γ × Power
  + δ × Model Size

subject to:

    Accuracy ≥ required_accuracy
```

The weights `α`, `β`, `γ`, and `δ` depend on the application.

For ADAS:

```text
Latency + Accuracy
```

may dominate.

For a battery-powered robot:

```text
Power + Accuracy
```

may dominate.

---

# 6. First Rule: Profile Before Optimizing

Never optimize blindly.

The first question is:

> **Where is the model spending its computation?**

Before changing anything, collect:

```text
Parameter count
FLOPs
MACs
Layer count
Activation sizes
Weight sizes
Per-layer latency
Memory usage
Operator distribution
Input resolution
Output resolution
```

---

# 7. Model Profiling

A baseline report should look conceptually like:

```text
Model: YOLO-like detector

Input:
640 × 640 × 3

Parameters:
7.2 M

FLOPs:
18.4 G

Model Size:
28.8 MB

Largest Activation:
12.4 MB

Total Activation Memory:
86 MB

Inference:
21.4 ms

FPS:
46.7
```

Then investigate individual layers.

Example:

```text
Layer                     Latency
--------------------------------
Conv 1                    0.8 ms
Conv 2                    1.2 ms
Backbone Block 1          2.4 ms
Backbone Block 2          4.8 ms
Attention Block           3.7 ms
Detection Head            4.1 ms
Postprocess               2.3 ms
```

Now optimization can target actual bottlenecks.

---

# 8. Parameter Count

A parameter is a learned value.

For a convolution:

```text
Parameters =
Kernel Height × Kernel Width
× Input Channels
× Output Channels
```

plus bias if present.

Example:

```text
3 × 3 convolution
Input channels = 64
Output channels = 128
```

Parameters:

```text
3 × 3 × 64 × 128
= 73,728
```

If bias exists:

```text
73,728 + 128
= 73,856
```

Parameter reduction can be achieved through:

- fewer layers
- fewer channels
- smaller kernels
- factorization
- pruning
- smaller architecture

---

# 9. FLOPs and MACs

FLOPs describe floating-point operations.

MAC:

```text
Multiply-Accumulate
```

One MAC is often approximately counted as:

```text
1 multiplication + 1 addition
= 2 FLOPs
```

Different tools use different conventions, so benchmark reports must clearly state the convention.

---

# 10. FLOPs Are Not Latency

This is one of the most important lessons.

Consider:

```text
Model A
10 GFLOPs
```

and:

```text
Model B
15 GFLOPs
```

Model B may still be faster.

Why?

Because latency depends on:

- memory access
- operator implementation
- tensor layout
- parallelism
- kernel efficiency
- hardware support
- cache behavior
- synchronization
- framework overhead

Therefore:

```text
FLOPs ≠ Latency
```

FLOPs are a useful theoretical metric, not the final deployment metric.

---

# 11. Architecture Optimization

Architecture optimization changes the structure of the network.

Major dimensions include:

```text
Depth
Width
Channels
Kernel Size
Resolution
Block Type
Backbone
Neck
Head
Activation
Normalization
```

---

# 12. Depth Reduction

Depth means the number of sequential computational blocks/layers.

Example:

```text
Original

Block 1
Block 2
Block 3
Block 4
Block 5
Block 6
Block 7
Block 8
```

Possible optimized version:

```text
Block 1
Block 2
Block 3
Block 4
Block 5
Block 6
```

Benefits:

- fewer operations
- lower latency
- fewer parameters

Risk:

- reduced representational capacity
- reduced accuracy

Depth reduction should therefore be tested experimentally.

---

# 13. Width Reduction

Width usually refers to the number of channels/features.

Example:

```text
Original:
256 channels

Optimized:
192 channels
```

or:

```text
256 → 128
```

For a convolution:

```text
Cost ∝ Input Channels × Output Channels
```

Therefore width reduction can produce a significant compute reduction.

---

# 14. Channel Reduction

Channels can be reduced globally or selectively.

Example:

```text
Layer:

128 channels
     ↓
64 channels
```

This can reduce:

- parameters
- FLOPs
- activation memory
- memory bandwidth

But channel removal must be structurally consistent with neighboring layers.

---

# 15. Width Multipliers

Many architectures use a width multiplier.

Example:

```text
width = 1.0
```

could become:

```text
width = 0.75
```

or:

```text
width = 0.5
```

This produces a smaller network.

Experiments:

```text
1.00×
0.75×
0.50×
0.35×
```

Benchmark:

```text
Width
Parameters
FLOPs
Accuracy
Latency
FPS
```

---

# 16. Depth Multipliers

Similarly:

```text
depth = 1.0
```

could become:

```text
0.75
0.50
```

This reduces repeated blocks.

A width-depth sweep can become a useful experiment:

```text
              Width
          0.5   0.75   1.0
Depth
0.5       A      B      C

0.75      D      E      F

1.0       G      H      I
```

Then identify the best accuracy/performance configuration.

---

# 17. Input Resolution Optimization

Input resolution has enormous impact on computation.

Example:

```text
640 × 640
```

versus:

```text
512 × 512
```

Number of pixels:

```text
640² = 409,600

512² = 262,144
```

That's approximately:

```text
36% fewer pixels
```

This can produce significant compute savings.

But accuracy may decrease, especially for small objects.

Therefore:

```text
Resolution
     ↓
Compute
     ↓
Latency
     ↓
Accuracy
```

must be measured together.

---

# 18. Dynamic Resolution

Instead of using one fixed resolution, the system can potentially select resolution based on conditions.

Example:

```text
Far / simple scene
     ↓
Lower resolution

Complex scene
     ↓
Higher resolution
```

This becomes a form of adaptive inference.

---

# 19. Kernel Size Optimization

Compare:

```text
7 × 7
```

with:

```text
5 × 5
```

and:

```text
3 × 3
```

Smaller kernels generally reduce computation.

But receptive field must be considered.

---

# 20. Convolution Factorization

A large convolution can sometimes be replaced by smaller operations.

Example:

```text
7 × 7
```

can potentially be approximated using:

```text
3 × 3
+
3 × 3
```

or other factorized structures.

The goal is:

```text
Similar representation
+
Lower computational cost
```

---

# 21. Depthwise Separable Convolution

Standard convolution:

```text
Input
 ↓
Full Convolution
 ↓
Output
```

Depthwise separable convolution:

```text
Input
 ↓
Depthwise Convolution
 ↓
Pointwise 1×1 Convolution
 ↓
Output
```

This can dramatically reduce computation.

It is used extensively in efficient architectures.

---

# 22. Grouped Convolution

Instead of connecting every input channel to every output channel:

```text
Full convolution
```

channels can be divided into groups.

```text
Input
 ├── Group 1 → Output Group 1
 ├── Group 2 → Output Group 2
 ├── Group 3 → Output Group 3
 └── Group 4 → Output Group 4
```

This reduces computation and parameter count.

---

# 23. Bottleneck Architecture

A common optimization pattern is:

```text
Wide
 ↓
Compress
 ↓
Compute
 ↓
Expand
```

Example:

```text
256 channels
      ↓
64 channels
      ↓
64 channels
      ↓
256 channels
```

This can significantly reduce intermediate computation.

---

# 24. Operator Replacement

Some operations are computationally expensive.

Possible replacements:

```text
Standard Conv
      ↓
Depthwise + Pointwise Conv
```

or:

```text
Expensive activation
      ↓
Cheaper activation
```

or:

```text
Complex block
      ↓
Simplified block
```

Operator replacement must be evaluated for:

- accuracy
- latency
- hardware support

---

# 25. Activation Function Optimization

Possible activations include:

```text
ReLU
ReLU6
LeakyReLU
SiLU / Swish
GELU
Hard-Swish
```

Some activations may be more expensive than others.

But changing activations can alter model behavior significantly.

Therefore activation replacement requires retraining or fine-tuning.

---

# 26. Normalization Optimization

Common normalization methods:

```text
BatchNorm
LayerNorm
GroupNorm
InstanceNorm
```

Different architectures and hardware platforms behave differently.

For deployment, normalization layers may also be candidates for fusion or replacement.

---

# 27. Backbone Optimization

In vision models, the backbone often dominates computation.

Example:

```text
Input
 ↓
Backbone
 ↓
Feature Pyramid
 ↓
Head
```

Potential experiments:

```text
Large backbone
      ↓
Smaller backbone
```

Examples conceptually:

```text
ResNet-50
   ↓
ResNet-18

Large custom backbone
   ↓
Mobile-style backbone
```

The key experiment:

```text
Backbone
Accuracy
FLOPs
Latency
Memory
```

---

# 28. Neck Optimization

Detection and segmentation models may contain expensive feature aggregation.

Possible optimization:

- reduce feature scales
- reduce channels
- simplify feature fusion
- remove redundant blocks
- replace expensive feature aggregation

---

# 29. Head Optimization

Detection/segmentation heads can sometimes be simplified.

For example:

```text
Large Detection Head
       ↓
Reduced Detection Head
```

Possible changes:

- fewer channels
- fewer branches
- simplified prediction layers
- reduced feature scales

---

# 30. Early Exit Networks

Not every input requires the complete network.

Concept:

```text
Input
 ↓
Early Layers
 ↓
Confidence?
 ├── YES → Output
 └── NO
       ↓
   More Layers
```

Easy samples exit early.

Hard samples continue through the full network.

This can reduce average inference cost.

---

# 31. Dynamic Neural Networks

A model can dynamically select computation.

Examples:

```text
Input
 ↓
Scene complexity
 ↓
Choose computation path
```

Possible strategies:

- dynamic depth
- dynamic width
- conditional computation
- early exit
- sparse activation

These are advanced topics and can be explored later.

---

# 32. Pruning

Pruning removes unnecessary model components.

Three major categories:

```text
Unstructured
Structured
Semi-structured
```

---

# 33. Unstructured Pruning

Individual weights are removed.

Example:

```text
Original:

[0.2  0.8  0.1  0.7]
[0.4  0.3  0.9  0.2]

After pruning:

[0.2  0.8  0.0  0.7]
[0.4  0.0  0.9  0.2]
```

High sparsity can reduce storage.

However:

```text
Sparse model
≠
Automatically faster model
```

Hardware and runtime support are critical.

---

# 34. Structured Pruning

Remove complete structures.

Examples:

```text
Channels
Filters
Heads
Blocks
Layers
```

Example:

```text
128 channels
     ↓
96 channels
```

Structured pruning is generally more deployment-friendly.

---

# 35. Semi-Structured Pruning

A middle ground between structured and unstructured sparsity.

Example:

```text
2:4 sparsity
```

means that in groups of four weights, two are zero.

This can be useful on hardware that supports structured sparsity.

---

# 36. Importance Metrics for Pruning

How do we decide what to remove?

Possible criteria:

### Magnitude

```text
importance ∝ |weight|
```

Small weights are considered less important.

### Gradient

Measure sensitivity through gradients.

### Activation

Determine how strongly a channel is used.

### Taylor-based importance

Estimate the effect of removing a parameter/channel.

### Learned importance

Train an importance mechanism.

---

# 37. Iterative Pruning

Instead of removing 50% immediately:

```text
Original
 ↓
10% pruning
 ↓
Fine-tune
 ↓
20%
 ↓
Fine-tune
 ↓
30%
 ↓
Fine-tune
```

This often preserves accuracy better.

---

# 38. Pruning + Fine-Tuning

A typical workflow:

```text
Train
 ↓
Prune
 ↓
Fine-tune
 ↓
Evaluate
 ↓
Prune more
 ↓
Fine-tune
 ↓
Evaluate
```

This should become one of the core experiments in this project.

---

# 39. Quantization

Quantization reduces numerical precision.

Typical representations:

```text
FP32
FP16
BF16
INT8
INT4
```

---

# 40. FP32

Standard floating point.

Advantages:

- high numerical precision
- baseline reference

Disadvantages:

- larger memory
- more computation
- higher bandwidth

FP32 should generally be the baseline.

---

# 41. FP16

Half precision.

Advantages:

- lower memory
- higher throughput on supported GPUs
- lower memory bandwidth

Potential issue:

- numerical instability for some operations

Inference is generally safer than training.

---

# 42. BF16

Brain Floating Point 16.

Compared with FP16:

```text
FP16:
More mantissa
Smaller exponent range

BF16:
Larger exponent range
Less mantissa precision
```

BF16 can be attractive for training and some inference hardware.

---

# 43. INT8

8-bit integer representation.

Advantages:

```text
4× smaller than FP32
```

in raw weight storage.

Potential benefits:

- lower memory
- lower bandwidth
- higher inference throughput
- lower power

Potential drawback:

```text
Accuracy degradation
```

---

# 44. Quantization Types

## Dynamic Quantization

Values are quantized during execution.

Useful for certain model types and operations.

---

## Static Quantization

Activation ranges are determined ahead of time.

Requires calibration.

---

## Quantization-Aware Training

The model is trained while simulating quantization.

```text
FP32 Training
     ↓
Fake Quantization
     ↓
QAT
     ↓
INT8 Deployment
```

---

# 45. Per-Tensor Quantization

One scale is used for an entire tensor.

```text
Tensor
 ↓
Single Scale
```

Simple but potentially less accurate.

---

# 46. Per-Channel Quantization

Different channels use different scales.

```text
Channel 1 → Scale 1
Channel 2 → Scale 2
Channel 3 → Scale 3
```

Often useful for convolution weights.

---

# 47. Symmetric Quantization

Approximately:

```text
[-127, +127]
```

with zero centered around zero.

---

# 48. Asymmetric Quantization

Uses a zero-point:

```text
quantized = scale × value + zero_point
```

This can better represent distributions that are not centered around zero.

---

# 49. Calibration

INT8 calibration estimates numerical ranges.

Calibration data should be representative of real deployment data.

Bad calibration data can produce poor INT8 accuracy.

Therefore we must study:

```text
Calibration Dataset
        ↓
Activation Statistics
        ↓
Scale Selection
        ↓
Quantized Model
        ↓
Accuracy
```

---

# 50. Quantization Sensitivity Analysis

Not every layer responds equally to quantization.

Example:

```text
Layer 1     INT8  ✓
Layer 2     INT8  ✓
Layer 3     INT8  ✓
Layer 4     INT8  ✗
Layer 5     INT8  ✓
```

Layer 4 may need:

```text
FP16
```

instead.

This leads to mixed precision.

---

# 51. Mixed Precision

Different layers use different numerical formats.

Example:

```text
Backbone
   ↓
FP16

Sensitive layer
   ↓
FP32

Detection head
   ↓
INT8
```

The goal is:

```text
Maximum performance
+
Acceptable accuracy
```

---

# 52. Knowledge Distillation

Knowledge distillation transfers information from a large model to a smaller model.

```text
Teacher
Large / Accurate
       │
       │
       ▼
   Knowledge
       │
       ▼
Student
Small / Fast
```

---

# 53. Logit Distillation

Teacher and student predictions are compared.

```text
Teacher logits
      ↓
Soft targets

Student logits
      ↓
Match teacher
```

---

# 54. Feature Distillation

Intermediate feature maps are also matched.

```text
Teacher Feature
       │
       ▼
Student Feature
       │
       ▼
Feature Loss
```

This can help smaller models retain useful representations.

---

# 55. Distillation for Detection

Potential components:

```text
Classification
Objectness
Bounding Boxes
Feature Maps
```

Different losses can be combined.

---

# 56. Distillation for Segmentation

Possible distillation targets:

```text
Feature maps
Class probabilities
Pixel-level logits
Boundary information
```

This is particularly interesting for segmentation models.

---

# 57. Low-Rank Factorization

Large matrices can sometimes be approximated by lower-rank matrices.

Conceptually:

```text
Large Matrix W

       ↓

W ≈ A × B
```

where:

```text
A and B
```

are smaller matrices.

This can reduce:

- parameters
- computation

Potential applications:

- fully connected layers
- attention projections
- convolution factorization

---

# 58. Weight Sharing

Multiple parameters can share values.

Instead of:

```text
w1
w2
w3
w4
w5
```

use a smaller codebook:

```text
c1
c2
c3
```

and map weights to those values.

This can reduce storage.

---

# 59. Weight Clustering

Weights are grouped into clusters.

Example:

```text
Original:

0.91
0.89
0.87
0.21
0.19
0.23

Clustered:

0.89
0.89
0.89
0.21
0.21
0.21
```

This can reduce representation complexity.

---

# 60. Knowledge-Based Architecture Design

Instead of optimizing an existing architecture blindly, a deployment-aware architecture can be designed from the beginning.

Consider:

```text
Hardware
   ↓
Available operators
   ↓
Memory constraints
   ↓
Latency requirement
   ↓
Architecture
```

This is called hardware-aware model design.

---

# 61. Hardware-Aware Optimization

A model optimized for a desktop GPU may not be optimal for Jetson Orin.

Consider:

```text
Desktop GPU
     ≠
Embedded GPU
```

Optimization must consider:

- compute capability
- memory bandwidth
- cache
- supported precision
- supported kernels
- power budget

Therefore, the project will ultimately use Jetson Orin as the main optimization target.

---

# 62. Hardware-Aware Neural Architecture Search

Advanced topic:

```text
Search Space
    ↓
Candidate Models
    ↓
Train/Evaluate
    ↓
Hardware Benchmark
    ↓
Select Best Model
```

Instead of minimizing FLOPs, optimize actual hardware latency.

Conceptually:

```text
Best Architecture =
Minimum latency
subject to
Accuracy constraint
```

This can become an advanced extension.

---

# 63. Activation Optimization

Weights are not the only source of memory.

Activations can be large.

Example:

```text
Input
 ↓
Feature map
256 × 160 × 160
```

could require substantial memory.

Potential optimization:

- reduce channels
- reduce resolution
- checkpointing during training
- operator redesign
- memory reuse
- activation quantization

---

# 64. Activation Sparsity

Some activations contain many zeros.

Example:

```text
[0, 0, 0, 0.7, 0, 0.4, 0, 0]
```

Sparse activation patterns may potentially be exploited.

This is an advanced topic.

---

# 65. Input Optimization

The model's input can also be optimized.

Examples:

```text
Resolution
Color channels
Image format
Frame skipping
Temporal sampling
Region of interest
```

For example:

```text
RGB
 ↓
Selected ROI
 ↓
Lower resolution
 ↓
Model
```

can dramatically reduce computation.

---

# 66. Temporal Optimization

For video applications, not every frame necessarily requires full inference.

Possible strategy:

```text
Frame 1 → Full inference
Frame 2 → Tracking
Frame 3 → Tracking
Frame 4 → Full inference
```

This can reduce average model execution.

However, temporal optimization belongs partly to the system/runtime level and should be treated as an advanced extension.

---

# 67. Multi-Task Model Optimization

Since this project includes custom, OD, SS, and potentially multi-task models, optimization should consider shared computation.

Example:

```text
                Backbone
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
       Detection Segmentation Pose
```

Instead of:

```text
Three independent models
```

a shared backbone may reduce total computation.

But optimization becomes more complex because tasks compete for representational capacity.

---

# 68. Multi-Task Optimization Trade-Off

Measure:

```text
Detection accuracy
Segmentation accuracy
Pose accuracy
Total latency
Memory
```

rather than optimizing one task independently.

---

# 69. Model Surgery

Model surgery means programmatically modifying an existing trained model.

Examples:

```text
Remove layer
Replace layer
Replace activation
Change channels
Replace normalization
Modify head
Insert quantization modules
```

This is especially important for this project because it teaches how PyTorch models are actually constructed.

---

# 70. State Dictionary Manipulation

A trained model can be represented by:

```python
state_dict
```

which contains learned tensors.

Example:

```python
for name, parameter in model.named_parameters():
    print(name, parameter.shape)
```

This allows analysis of:

- layer sizes
- parameter distributions
- weight magnitudes
- sparsity

---

# 71. Weight Distribution Analysis

Before pruning or quantization, inspect distributions.

Example:

```text
Weight magnitude

0.0 ┤████████████████
0.1 ┤██████████████
0.2 ┤████████
0.3 ┤████
0.4 ┤██
```

This can reveal:

- near-zero weights
- outliers
- heavy-tailed distributions
- quantization sensitivity

---

# 72. Sensitivity Analysis

Instead of applying one optimization level everywhere:

```text
30% pruning everywhere
```

measure layer sensitivity.

Example:

```text
Layer       10%   20%   30%
--------------------------------
Layer 1      ✓     ✓     ✓
Layer 2      ✓     ✓     ✗
Layer 3      ✓     ✗     ✗
Layer 4      ✓     ✓     ✓
```

Then use different pruning ratios.

This is much more intelligent than uniform optimization.

---

# 73. Layer-Wise Optimization

Every layer can have a different optimization strategy.

Example:

```text
Backbone:
Structured pruning

Middle layers:
FP16

Sensitive head:
FP32

Other layers:
INT8
```

This produces a heterogeneous optimization strategy.

---

# 74. Accuracy Recovery

Optimization often causes accuracy loss.

Recovery techniques include:

```text
Fine-tuning
Knowledge distillation
Learning-rate adjustment
Data augmentation
Longer training
QAT
Layer freezing
Selective retraining
```

Typical workflow:

```text
Baseline
   ↓
Optimization
   ↓
Accuracy loss
   ↓
Fine-tuning
   ↓
Accuracy recovery
```

---

# 75. Fine-Tuning After Pruning

A common workflow:

```text
Pretrained Model
      ↓
Pruning
      ↓
Masked/Reduced Model
      ↓
Fine-tune
      ↓
Optimized Model
```

The fine-tuned model must be evaluated on the original validation set.

---

# 76. Optimization Experiments

Each optimization technique should be tested independently first.

Example experiment sequence:

```text
EXP-001
Baseline

EXP-002
Input resolution

EXP-003
Width reduction

EXP-004
Depth reduction

EXP-005
Structured pruning

EXP-006
Unstructured pruning

EXP-007
FP16

EXP-008
PTQ INT8

EXP-009
QAT INT8

EXP-010
Knowledge distillation

EXP-011
Low-rank factorization

EXP-012
Combined optimization
```

Only after understanding individual techniques should they be combined.

---

# 77. Combined Optimization

Eventually:

```text
Architecture reduction
        +
Structured pruning
        +
Quantization
        +
Knowledge distillation
        +
Fine-tuning
```

could produce a substantially smaller model.

Example:

```text
Original
100%

       ↓

Architecture optimization
75%

       ↓

Pruning
55%

       ↓

INT8
~25% weight storage

       ↓

Distillation
Accuracy recovery
```

These percentages are illustrative; actual results must be measured.

---

# 78. Pareto Optimization

There is rarely one universally best model.

You may obtain:

```text
Model A
90 FPS
75% mAP

Model B
70 FPS
79% mAP

Model C
50 FPS
81% mAP
```

These form a performance/accuracy trade-off.

The goal is to identify the **Pareto frontier**.

```text
Accuracy ↑

81% |        C
80% |
79% |              B
78% |
77% |
76% |                       A
    +----------------------------→ FPS
```

Models dominated by another configuration can be discarded.

---

# 79. What Should Be Measured?

Every experiment should record:

## Model Metrics

```text
Parameters
Trainable parameters
Non-zero parameters
Model size
FLOPs
MACs
Number of layers
```

## Memory Metrics

```text
Weight memory
Activation memory
Peak memory
GPU memory
CPU memory
```

## Performance Metrics

```text
Preprocessing latency
Inference latency
Postprocessing latency
End-to-end latency
FPS
Throughput
```

## Accuracy Metrics

Task dependent:

```text
mAP
IoU
mIoU
Dice
Precision
Recall
F1
NME
PCK
```

## Hardware Metrics

```text
GPU utilization
CPU utilization
Power
Temperature
Clock frequency
```

---

# 80. Standard Optimization Report

Every experiment should produce a report such as:

```text
====================================================
MODEL OPTIMIZATION REPORT
====================================================

Model:
YOLO Detection

Experiment:
Structured Pruning 30%

Hardware:
Jetson Orin

----------------------------------------------------
MODEL
----------------------------------------------------

Parameters:
6.2M → 4.1M

FLOPs:
12.4G → 8.1G

Model Size:
25.0 MB → 16.5 MB

----------------------------------------------------
ACCURACY
----------------------------------------------------

mAP@50:
79.3 → 78.8

mAP@50:95:
56.5 → 55.9

----------------------------------------------------
PERFORMANCE
----------------------------------------------------

Latency:
18.2 ms → 13.6 ms

FPS:
55 → 73

----------------------------------------------------
MEMORY
----------------------------------------------------

GPU Memory:
XXX → XXX MB

----------------------------------------------------
CONCLUSION
----------------------------------------------------

30% structured pruning provides:

+32.7% FPS
-25.3% latency
-34% parameters

with:

-0.5 mAP@50 degradation
====================================================
```

---

# 81. Recommended Tooling

The exact toolset can evolve, but the learning stack should include:

## PyTorch

Core model manipulation and training.

## Torch Profiler

Model-level execution profiling.

## `torchinfo`

Model architecture inspection.

## `fvcore`

FLOPs and operator analysis.

## `thop`

MAC/FLOPs estimation.

## PyTorch Quantization APIs

Quantization experiments.

## PyTorch Pruning APIs

Initial pruning experiments.

## Custom PyTorch code

Important for understanding what the optimization libraries actually do.

---

# 82. Important Principle: Implement Some Techniques Yourself

Libraries should not hide the concepts.

For example, don't only do:

```python
prune.l1_unstructured(...)
```

Understand what is happening mathematically.

Similarly, don't only call a quantization API.

Understand:

```text
scale
zero point
clipping
rounding
dequantization
```

At least some simplified versions should be implemented manually for learning.

---

# 83. Mathematical Foundation

The project should gradually cover:

### Convolution complexity

```text
FLOPs ≈
2 × H × W × Cin × Cout × K²
```

for a standard 2D convolution.

### Parameter count

```text
Parameters =
Cin × Cout × K²
```

### Quantization

Conceptually:

```text
q = round(x / scale) + zero_point
```

### Dequantization

```text
x ≈ scale × (q - zero_point)
```

### Sparsity

```text
Sparsity =
Zero Parameters / Total Parameters
```

### Compression ratio

```text
Compression Ratio =
Original Size / Optimized Size
```

These equations should be connected to actual experiments.

---

# 84. Project Learning Levels

## Level 1 — Model Understanding

Learn:

- model architecture
- tensors
- layers
- parameters
- activations
- FLOPs
- MACs
- receptive field

---

## Level 2 — Profiling

Learn:

- PyTorch Profiler
- per-layer latency
- memory
- bottleneck identification

---

## Level 3 — Architecture Optimization

Learn:

- depth
- width
- channels
- kernels
- blocks
- backbones
- heads

---

## Level 4 — Pruning

Learn:

- unstructured pruning
- structured pruning
- semi-structured pruning
- sensitivity analysis
- iterative pruning

---

## Level 5 — Quantization

Learn:

- FP16
- BF16
- INT8
- dynamic quantization
- static quantization
- QAT
- calibration
- mixed precision

---

## Level 6 — Distillation

Learn:

- teacher/student
- logits
- features
- task-specific distillation

---

## Level 7 — Advanced Compression

Learn:

- low-rank factorization
- weight sharing
- clustering
- sparsity
- activation compression

---

## Level 8 — Hardware-Aware Optimization

Learn:

- actual hardware latency
- memory bandwidth
- operator support
- hardware-aware architecture design
- hardware-aware NAS

---

## Level 9 — Combined Optimization

Combine:

```text
Architecture
+
Pruning
+
Quantization
+
Distillation
+
Fine-tuning
```

---

# 85. Recommended Experiment Progression

The initial practical roadmap should be:

```text
                         BASELINE
                            │
                            ▼
                    Profile .pt model
                            │
                            ▼
                 Analyze architecture
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
        Width sweep    Depth sweep   Resolution sweep
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                  Best architecture
                            │
                            ▼
                  Structured pruning
                            │
                            ▼
                       Fine-tune
                            │
                            ▼
                       Quantization
                            │
                            ▼
                    Quantization-aware
                       fine-tuning
                            │
                            ▼
                     Distillation
                            │
                            ▼
                  Final optimized .pt
```

Then this model moves to:

```text
ONNX
 ↓
ONNX Runtime
 ↓
OpenVINO
 ↓
TensorRT
 ↓
Jetson Orin
```

---

# 86. Definition of a Successful Optimization

An optimization is successful only if it satisfies the project's requirements.

For example:

```text
Accuracy:
mAP ≥ 78%

Latency:
≤ 15 ms

Memory:
≤ 2 GB

Power:
≤ 20 W
```

A model that achieves:

```text
5 ms latency
```

but:

```text
50% accuracy
```

is not a successful optimization.

Likewise:

```text
79% mAP
```

with:

```text
200 ms latency
```

may not satisfy a real-time application.

---

# 87. Model-Level Optimization Checklist

Before declaring a model optimized:

```text
[ ] Baseline model measured
[ ] Parameter count calculated
[ ] FLOPs/MACs calculated
[ ] Layer-wise profiling performed
[ ] Memory usage measured
[ ] Bottlenecks identified

[ ] Architecture alternatives tested
[ ] Width optimization tested
[ ] Depth optimization tested
[ ] Resolution optimization tested

[ ] Structured pruning tested
[ ] Unstructured pruning tested
[ ] Sensitivity analysis performed

[ ] FP16 tested
[ ] INT8 tested
[ ] Calibration evaluated
[ ] QAT evaluated
[ ] Mixed precision evaluated

[ ] Knowledge distillation tested
[ ] Fine-tuning performed
[ ] Accuracy validated

[ ] Optimized .pt saved
[ ] Model reproducibility verified
[ ] Benchmark report generated
```

---

# 88. Final Model-Level Pipeline

The complete model-level optimization workflow is:

```text
                    TRAINED .PT
                        │
                        ▼
                   VALIDATION
                        │
                        ▼
                    PROFILING
                        │
                        ▼
              BOTTLENECK ANALYSIS
                        │
                        ▼
              ARCHITECTURE SEARCH
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
         Width        Depth      Resolution
            │           │           │
            └───────────┼───────────┘
                        ▼
                  MODEL SURGERY
                        │
                        ▼
                    PRUNING
                        │
                        ▼
                   FINE-TUNING
                        │
                        ▼
                  QUANTIZATION
                        │
                        ▼
             QUANTIZATION AWARE
                    TRAINING
                        │
                        ▼
                  DISTILLATION
                        │
                        ▼
                 FINAL .PT MODEL
                        │
                        ▼
               ACCURACY VALIDATION
                        │
                        ▼
              DEPLOYMENT BENCHMARK
                        │
                        ▼
                OPTIMIZATION REPORT
```

---

# 89. End Goal

The ultimate goal of this module is to move from:

```text
"I have a trained model."
```

to:

```text
"I understand exactly where my model spends its
compute and memory, what can be removed or
compressed, how each optimization affects accuracy,
and which optimized architecture is appropriate for
my deployment constraints."
```

The final output is not simply:

```text
model.pt
```

It is:

```text
Optimized Model
+
Optimization Configuration
+
Accuracy Results
+
Performance Results
+
Memory Results
+
Power Results
+
Benchmark Results
+
Reproducible Experiment
```

---

# 90. Project Principle

> **Never optimize a model because an optimization technique exists. Optimize because profiling shows a bottleneck and the resulting change can be measured.**

The core engineering loop is:

```text
             MEASURE
                ↓
             ANALYZE
                ↓
             MODIFY
                ↓
             TRAIN
                ↓
            VALIDATE
                ↓
           BENCHMARK
                ↓
            COMPARE
                ↓
             REPEAT
```

This loop is the foundation of the entire AI Model Optimization project.

---

# 91. Future Extension

Once model-level optimization is understood, this optimized `.pt` model becomes the input to the next stage:

```text
MODEL-LEVEL OPTIMIZATION
            │
            ▼
       Optimized .pt
            │
            ▼
GRAPH-LEVEL OPTIMIZATION
            │
            ▼
     ONNX / IR / Graph
            │
            ▼
RUNTIME-SPECIFIC OPTIMIZATION
            │
      ┌─────┼────────┐
      ▼     ▼        ▼
     ORT OpenVINO TensorRT
      │     │        │
      └─────┼────────┘
            ▼
      HARDWARE LEVEL
            │
            ▼
       Jetson Orin
            │
            ▼
      REAL-TIME SYSTEM
```

The model-level stage therefore forms the **foundation of the entire project**.

If this stage is understood deeply, the later ONNX, OpenVINO, TensorRT, CUDA, and Jetson optimization stages become much easier to reason about because you will understand **what is being optimized and why**, rather than simply using conversion tools.
