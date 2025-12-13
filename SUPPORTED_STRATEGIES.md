# VBench: Supported Evaluation Workflow Strategies

This document identifies which strategies from the **Unified Evaluation Workflow** are natively supported by VBench in its full installation. A strategy is considered "supported" only if VBench provides it natively—meaning that once the harness is fully installed, the strategy can be executed directly without implementing custom modules or integrating external libraries such as monitoring tools or insight-generation components.

## Analysis Methodology

This analysis is based on:
- VBench repository documentation (README.md, setup.py, requirements.txt)
- VBench, VBench++, and VBench-2.0 source code examination
- Installation procedures and dependencies
- Native evaluation capabilities provided out-of-the-box

---

## Phase 0: Provisioning (The Runtime)

### Step A: Harness Installation

#### ✅ Strategy 1: Git Clone
**Status:** **SUPPORTED**

**Evidence:**
- README.md provides git clone instructions:
  ```bash
  git clone https://github.com/Vchitect/VBench.git
  ```
- Source installation is a primary installation method documented in the repository

#### ✅ Strategy 2: PyPI Packages  
**Status:** **SUPPORTED**

**Evidence:**
- VBench is available on PyPI: `pip install vbench`
- setup.py (v0.1.5) provides PyPI package configuration
- README-pypi.md documents PyPI installation
- requirements.txt specifies Python package dependencies (numpy, torch, transformers, etc.)
- Git-based dependencies are supported: `detectron2@git+https://github.com/facebookresearch/detectron2.git`

#### ❌ Strategy 3: Node Package
**Status:** **NOT SUPPORTED**

**Evidence:**
- No package.json, npm configuration, or Node.js dependencies found
- VBench is Python-only

#### ❌ Strategy 4: Binary Packages
**Status:** **NOT SUPPORTED**

**Evidence:**
- No standalone executable binaries provided
- Installation requires Python environment and pip

#### ❌ Strategy 5: Container Images
**Status:** **NOT SUPPORTED**

**Evidence:**
- No Dockerfile or docker-compose.yml for VBench itself
- Dockerfiles found are only in deeply nested third_party subdirectories (VBench-2.0/vbench2/third_party/YOLO-World) for external dependencies, not for VBench
- No container images mentioned in VBench installation documentation
- No Docker-based installation workflow documented

### Step B: Service Authentication

#### ❌ Strategy 1: API Provider Authentication
**Status:** **NOT SUPPORTED**

**Evidence:**
- VBench evaluates pre-generated videos, not live model APIs
- No API key configuration for commercial model providers (OpenAI, Anthropic, etc.)
- No environment variable handling for model API authentication in core codebase
- Note: VBench-2.0 uses local LLM judges (Qwen) but doesn't call external APIs

#### ✅ Strategy 2: Repository Authentication
**Status:** **SUPPORTED**

**Evidence:**
- VBench downloads models from HuggingFace Hub:
  ```python
  wget_command = ['wget', 'https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/...', ...]
  ```
- Uses `torch.hub.load()` for model loading which can use HuggingFace authentication
- Users need HuggingFace authentication to access gated models
- vbench/utils.py downloads weights from huggingface.co

#### ❌ Strategy 3: Evaluation Platform Authentication
**Status:** **NOT SUPPORTED**

**Evidence:**
- No CLI authentication flows for evaluation platforms
- Leaderboard submission is manual via web form, not programmatic API
- README states: "Submit your eval_results.zip files to the VBench Leaderboard's [T2V]Submit here! form"

---

## Phase I: Specification (The Contract)

### Step A: SUT Preparation

#### ❌ Strategy 1: Model-as-a-Service (Remote Inference)
**Status:** **NOT SUPPORTED**

**Evidence:**
- VBench evaluates pre-generated videos, not models directly
- No HTTP endpoint configuration or API client code for remote inference
- Users must generate videos externally before evaluation
- README: "VBench evaluates videos, and video generative models" (videos, not live models)

#### ✅ Strategy 2: Model-in-Process (Local Inference)
**Status:** **SUPPORTED**

**Evidence:**
- VBench loads evaluation models locally for scoring:
  - DINO model: `torch.hub.load()` (subject_consistency.py)
  - CLIP model: `clip.load()` (aesthetic_quality.py)
  - UMT model: PyTorch checkpoint loading (human_action.py)
  - VBench-2.0 loads LLaVA-Video and Qwen models locally (complex_plot.py)
- Models are loaded into memory with full access to features and logits
- Traditional ML models used for evaluation (CNNs, transformers)

#### ❌ Strategy 3: Algorithm Implementation (In-Memory Structures)
**Status:** **NOT SUPPORTED**

**Evidence:**
- VBench does not instantiate ANN algorithms, BM25 indexes, or vector search structures
- Evaluation metrics use neural models, not algorithmic data structures

#### ❌ Strategy 4: Policy/Agent Instantiation (Stateful Controllers)
**Status:** **NOT SUPPORTED**

**Evidence:**
- No reinforcement learning policy or agent instantiation code
- VBench evaluates video outputs, not interactive agents or robot controllers

### Step B: Benchmark Preparation (Inputs)

#### ✅ Strategy 1: Benchmark Dataset Preparation (Offline)
**Status:** **SUPPORTED**

**Evidence:**
- VBench loads pre-existing benchmark datasets:
  - VBench_full_info.json contains prompt suite and metadata
  - `load_dimension_info()` function loads benchmark prompts
  - Videos are loaded from disk: `load_video(video_path)`
  - Data formatting: video frames are preprocessed with transforms (dino_transform, clip_transform)
- Dataset splits managed via dimension-specific prompt files
- README: "Prompt Suite" and "Sampled Videos" sections document benchmark data

#### ❌ Strategy 2: Synthetic Data Generation (Generative)
**Status:** **NOT SUPPORTED**

**Evidence:**
- VBench does not generate test videos
- No input perturbation, test augmentation, or trajectory generation code
- static_filter.py filters static videos but doesn't generate new ones

#### ❌ Strategy 3: Simulation Environment Setup (Simulated)
**Status:** **NOT SUPPORTED**

**Evidence:**
- No 3D environment initialization, scene construction, or physics simulation
- VBench evaluates pre-recorded videos, not interactive simulations

#### ❌ Strategy 4: Production Traffic Sampling (Online)
**Status:** **NOT SUPPORTED**

**Evidence:**
- No real-time traffic streaming or production inference monitoring
- VBench is an offline benchmark evaluation tool

### Step C: Benchmark Preparation (References)

#### ✅ Strategy 1: Ground Truth Preparation
**Status:** **SUPPORTED**

**Evidence:**
- Ground truth annotations in VBench_full_info.json and VBench2_full_info.json
- Human action labels: kinetics_400_categories.txt
- Prompt-video mappings for reference comparison
- Auxiliary information in prompts: composition.py uses `prompt_dict['auxiliary_info']['judge']`
- Pre-computed reference features used in scoring

#### ✅ Strategy 2: Judge Preparation
**Status:** **SUPPORTED** (VBench-2.0 only)

**Evidence:**
- VBench-2.0 loads pre-trained judge models:
  - Qwen LLM judge: complex_plot.py, motion_order_understanding.py
  - LLaVA-Video multimodal judge: complex_plot.py
  ```python
  # Actual code from complex_plot.py:
  qwen_model = AutoModelForCausalLM.from_pretrained(
      qwen_model_name,
      torch_dtype="auto",
      device_map="auto",
      cache_dir=submodules_dict['qwen']
  )
  qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name, cache_dir=submodules_dict['qwen'])
  llava_model, llava_tokenizer, image_processor, max_length = load_pretrained_model(...)
  ```
- System prompts define judge behavior: "You are a brilliant plot consistency judger"
- README: "we highly recommend users to download the model before evaluation"

**Note:** Base VBench (v1.0) does not use LLM judges; only VBench-2.0 does

---

## Phase II: Execution (The Run)

### Step A: SUT Invocation

#### ✅ Strategy 1: Batch Inference
**Status:** **SUPPORTED**

**Evidence:**
- VBench processes multiple video samples in batch:
  ```python
  for video_path in tqdm(video_list):
      images = load_video(video_path)
      # evaluate each video
  ```
- evaluate.py runs evaluation across all videos in a directory
- Batch processing of frames: `for i in range(0, len(images), batch_size)`
- Evaluation runs on pre-generated video batches

#### ❌ Strategy 2: Arena Battle
**Status:** **NOT SUPPORTED**

**Evidence:**
- VBench evaluates each model's videos independently
- No concurrent multi-model comparison during evaluation
- Comparisons happen post-hoc via leaderboard, not during execution
- Note: "VBench Arena" is a web interface for human voting, not a programmatic arena battle

#### ❌ Strategy 3: Interactive Loop
**Status:** **NOT SUPPORTED**

**Evidence:**
- No stateful environment stepping or action-observation loops
- Videos are pre-recorded, not generated via interactive simulation
- No tool-based reasoning or multi-agent coordination during evaluation

#### ❌ Strategy 4: Production Streaming
**Status:** **NOT SUPPORTED**

**Evidence:**
- No real-time production traffic processing
- No drift monitoring or live metric collection
- Evaluation is offline batch processing

---

## Phase III: Assessment (The Score)

### Step A: Individual Scoring

#### ✅ Strategy 1: Deterministic Measurement
**Status:** **SUPPORTED**

**Evidence:**
- VBench implements deterministic metrics:
  - Edit distance: temporal_flickering.py
  - Token-based metrics: BLEU, ROUGE (via pycocoevalcap dependency)
  - Geometric calculations: cosine similarity in subject_consistency.py
  - Pass/fail checks: human_action.py checks if predicted action matches ground truth
  ```python
  sim_fir = F.cosine_similarity(first_image_features, image_features).item()
  ```

#### ✅ Strategy 2: Embedding Measurement
**Status:** **SUPPORTED**

**Evidence:**
- Semantic similarity via embeddings:
  - CLIP embeddings: aesthetic_quality.py uses `clip_model.encode_image()`
  - DINO embeddings: subject_consistency.py uses DINO features
  - ViCLIP embeddings: overall_consistency.py (based on directory structure)
  ```python
  image_features = model(image)
  image_features = F.normalize(image_features, dim=-1, p=2)
  ```
- Embedding-based comparisons for visual similarity

#### ✅ Strategy 3: Subjective Measurement
**Status:** **SUPPORTED** (VBench-2.0 only)

**Evidence:**
- VBench-2.0 uses LLM judges for subjective evaluation:
  - complex_plot.py: Qwen judges plot consistency
  - motion_order_understanding.py: Qwen judges action order
  ```python
  sys_prompt = "You are a helpful assistant and a brilliant plot consistency judger."
  response = judge(prompt, sys_prompt, qwen_tokenizer, qwen_model)
  ```
- LLMs assess subjective attributes that require human-like judgment

**Note:** Base VBench does not have subjective measurement; only VBench-2.0 does

#### ❌ Strategy 4: Performance Measurement
**Status:** **NOT SUPPORTED**

**Evidence:**
- No latency, throughput, or memory measurement code
- No FLOPs counting or energy consumption tracking
- VBench evaluates quality, not performance/efficiency
- No timing or resource profiling in evaluation scripts

### Step B: Aggregate Scoring

#### ✅ Strategy 1: Distributional Statistics
**Status:** **SUPPORTED**

**Evidence:**
- VBench aggregates per-instance scores:
  ```python
  aesthetic_avg /= num  # averaging
  sim_per_frame = sim / cnt  # mean calculation
  acc = cor_num / cnt  # accuracy aggregation
  ```
- scripts/cal_final_score.py calculates Total Score, Quality Score, Semantic Score
- Weighted aggregation documented in README:
  ```
  Total Score = w1 * Quality Score + w2 * Semantic Score
  ```
- Rank aggregation: results organized by model performance

#### ❌ Strategy 2: Uncertainty Quantification
**Status:** **NOT SUPPORTED**

**Evidence:**
- No bootstrap resampling or confidence interval calculation
- No Prediction-Powered Inference (PPI) implementation
- Scores are point estimates without uncertainty bounds

---

## Phase IV: Reporting (The Output)

### Step A: Insight Presentation

#### ❌ Strategy 1: Execution Tracing
**Status:** **NOT SUPPORTED**

**Evidence:**
- No detailed step-by-step execution logs showing intermediate states
- Logging is minimal (basic info level)
- No function call tracing or data transformation visualization

#### ❌ Strategy 2: Subgroup Analysis
**Status:** **NOT SUPPORTED**

**Evidence:**
- No automatic stratification by demographics, domains, or task categories
- VBench has dimension-level results but not automatic subgroup breakdowns
- Users must manually organize results by category

#### ❌ Strategy 3: Regression Alerting
**Status:** **NOT SUPPORTED**

**Evidence:**
- No automatic comparison against historical baselines
- No performance degradation detection or alerting
- No threshold-based regression monitoring

#### ❌ Strategy 4: Chart Generation
**Status:** **NOT SUPPORTED**

**Evidence:**
- No native chart generation code in the repository
- Radar charts shown in README are externally generated
- No matplotlib/plotly chart creation in evaluation code

#### ❌ Strategy 5: Dashboard Creation
**Status:** **NOT SUPPORTED**

**Evidence:**
- No interactive web interface code in the repository
- VBench Arena and leaderboards are hosted externally on HuggingFace Spaces
- Evaluation code outputs JSON files, not dashboards:
  ```python
  save_json(results_dict, output_name)
  ```

#### ✅ Strategy 6: Leaderboard Publication
**Status:** **SUPPORTED** (Manual submission)

**Evidence:**
- VBench supports leaderboard submission workflow:
  - scripts/cal_final_score.py prepares submission files
  - README documents submission: "Submit the json file to HuggingFace"
  - Leaderboard at https://huggingface.co/spaces/Vchitect/VBench_Leaderboard
  - Structured output format for leaderboard integration
  
**Note:** Submission is manual via web form, not programmatic API

---

## Summary Table

| Phase | Step | Strategy | Supported | Notes |
|-------|------|----------|-----------|-------|
| **Phase 0** | **A: Harness Installation** | | | |
| | | 1. Git Clone | ✅ | Primary installation method |
| | | 2. PyPI Packages | ✅ | `pip install vbench` |
| | | 3. Node Package | ❌ | Python-only |
| | | 4. Binary Packages | ❌ | No standalone binaries |
| | | 5. Container Images | ❌ | No Docker images |
| | **B: Service Authentication** | | | |
| | | 1. API Provider Authentication | ❌ | Evaluates videos, not live APIs |
| | | 2. Repository Authentication | ✅ | HuggingFace Hub for models |
| | | 3. Evaluation Platform Authentication | ❌ | Manual web form submission |
| **Phase I** | **A: SUT Preparation** | | | |
| | | 1. Model-as-a-Service | ❌ | No remote inference |
| | | 2. Model-in-Process | ✅ | Local evaluation models |
| | | 3. Algorithm Implementation | ❌ | No ANN/BM25 algorithms |
| | | 4. Policy/Agent Instantiation | ❌ | No RL agents |
| | **B: Benchmark Preparation (Inputs)** | | | |
| | | 1. Benchmark Dataset Preparation | ✅ | VBench_full_info.json |
| | | 2. Synthetic Data Generation | ❌ | No video generation |
| | | 3. Simulation Environment Setup | ❌ | No 3D simulation |
| | | 4. Production Traffic Sampling | ❌ | Offline only |
| | **C: Benchmark Preparation (References)** | | | |
| | | 1. Ground Truth Preparation | ✅ | Annotations in JSON |
| | | 2. Judge Preparation | ✅ | VBench-2.0 only (Qwen, LLaVA) |
| **Phase II** | **A: SUT Invocation** | | | |
| | | 1. Batch Inference | ✅ | Processes video batches |
| | | 2. Arena Battle | ❌ | Independent evaluation |
| | | 3. Interactive Loop | ❌ | Pre-recorded videos |
| | | 4. Production Streaming | ❌ | Offline only |
| **Phase III** | **A: Individual Scoring** | | | |
| | | 1. Deterministic Measurement | ✅ | Cosine sim, accuracy, etc. |
| | | 2. Embedding Measurement | ✅ | CLIP, DINO, ViCLIP |
| | | 3. Subjective Measurement | ✅ | VBench-2.0 only (LLM judges) |
| | | 4. Performance Measurement | ❌ | No latency/throughput |
| | **B: Aggregate Scoring** | | | |
| | | 1. Distributional Statistics | ✅ | Averaging, weighted scoring |
| | | 2. Uncertainty Quantification | ❌ | No confidence intervals |
| **Phase IV** | **A: Insight Presentation** | | | |
| | | 1. Execution Tracing | ❌ | Minimal logging |
| | | 2. Subgroup Analysis | ❌ | No automatic stratification |
| | | 3. Regression Alerting | ❌ | No baseline comparison |
| | | 4. Chart Generation | ❌ | No native charts |
| | | 5. Dashboard Creation | ❌ | External HuggingFace Spaces |
| | | 6. Leaderboard Publication | ✅ | Manual submission workflow |

---

## Supported Strategy Count: 13 out of 42

**Supported Strategies (13):**
1. Phase 0-A-1: Git Clone
2. Phase 0-A-2: PyPI Packages
3. Phase 0-B-2: Repository Authentication
4. Phase I-A-2: Model-in-Process (Local Inference)
5. Phase I-B-1: Benchmark Dataset Preparation (Offline)
6. Phase I-C-1: Ground Truth Preparation
7. Phase I-C-2: Judge Preparation (VBench-2.0 only)
8. Phase II-A-1: Batch Inference
9. Phase III-A-1: Deterministic Measurement
10. Phase III-A-2: Embedding Measurement
11. Phase III-A-3: Subjective Measurement (VBench-2.0 only)
12. Phase III-B-1: Distributional Statistics
13. Phase IV-A-6: Leaderboard Publication

**Not Supported (29):** All other strategies

---

## Key Characteristics of VBench

**VBench is:**
- A **video quality evaluation benchmark** for pre-generated videos
- An **offline batch evaluation system**
- A **Python package** installable via pip/git
- Focused on **quality assessment**, not performance/efficiency measurement
- Using **local model inference** for evaluation metrics
- Supporting **deterministic and embedding-based metrics** (all versions)
- Supporting **LLM-based subjective evaluation** (VBench-2.0 only)

**VBench is NOT:**
- A model inference/serving platform
- A real-time monitoring system
- A video generation framework
- An interactive simulation environment
- A dashboard/visualization tool (uses external HuggingFace Spaces)
- A performance benchmarking tool (no latency/throughput metrics)

---

## Version Differences

### VBench (v1.0)
- 16 evaluation dimensions (quality and semantic)
- Deterministic and embedding-based metrics only
- No LLM judges

### VBench++ (TPAMI 2025)
- Adds VBench-I2V (image-to-video)
- Adds VBench-Long (long videos ≥5s)
- Adds VBench-Trustworthiness (fairness, bias, safety)

### VBench-2.0
- 18 evaluation dimensions (intrinsic faithfulness)
- **Adds LLM judge models** (Qwen, LLaVA-Video)
- Focuses on commonsense reasoning, physics, human motion, composition

---

*This analysis is based on the VBench repository as of the current codebase state.*
