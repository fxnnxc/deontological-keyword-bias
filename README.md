# 🧠 Deontological Keyword Bias

<div align="center">

<img src="https://img.shields.io/badge/ACL-2025-blue?style=for-the-badge&logo=academia" alt="ACL 2025" style="pointer-events: none;" />
<img src="https://img.shields.io/badge/Paper-ArXiv-orange?style=for-the-badge&logo=arxiv" alt="Paper" style="pointer-events: none;" />
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" style="pointer-events: none;" />


**Deontological Keyword Bias: The Impact of Modal Expressions on Normative Judgments of Language Models**

[**🌐 Project Page**](https://bumjini.github.io/articles/acl2025_deontology/) | [**📄 Paper**](https://arxiv.org/abs/2506.11068) |

</div>

---

## 🎯 Abstract

This paper investigates a systematic bias in large language models (LLMs) where they interpret sentences containing modal expressions (e.g., *must*, *should*, *ought to*) as **moral obligations**, even when human annotators do not perceive them as such. We term this phenomenon **Deontological Keyword Bias (DKB)** and demonstrate its prevalence across multiple state-of-the-art language models.

### 🔑 Key Contributions
- **Discovery**: First systematic identification of DKB in LLMs
- **Analysis**: Comprehensive evaluation across 6+ models and 2,000+ sentences
- **Solution**: Novel reasoning-based few-shot method to reduce bias
- **Impact**: Improved alignment between model and human normative judgments


## 🔍 TL;DR  
Large language models (LLMs) systematically interpret sentences with modal expressions (like *must*, *should*) as **moral obligations**, even when humans do not.  
We call this **Deontological Keyword Bias (DKB)** and propose a **reasoning-based few-shot method** to reduce it.

<img src="https://d2acbkrrljl37x.cloudfront.net/research/publication/acl2025_exp2.jpeg" width="90%" height="auto" class="styled-image"/>

## 📁 Repository Structure
```
deontological-keyword-bias/
├── dataset/                    # Datasets for experiments
│   ├── deontology.csv         # Deontic dataset (+modal, neutral)
│   ├── deontology_negation.csv # Deontic dataset with negation
│   ├── commonsense.csv        # Non-deontic dataset
│   ├── commonsense_negation.csv # Non-deontic dataset with negation
│   ├── morality_high.csv      # High moral ambiguity cases
│   └── morality_low.csv       # Low moral ambiguity cases
├── assets/                     # Project assets
│   ├── alignment.png          # Alignment visualization
│   └── main1.png             # Main figure
├── models.py                  # Model loading utilities
├── run_deontology.py         # Baseline deontology experiments
├── run_commonsense.py        # Baseline commonsense experiments
├── run_deontology_negation.py # Deontology negation experiments
├── run_commonsense_negation.py # Commonsense negation experiments
├── run_reduce_effects_deontology.py # Debiasing experiments (deontology)
├── run_reduce_effects_commonsense.py # Debiasing experiments (commonsense)
├── run_baseline.sh           # Baseline experiment runner
├── run_debias.sh            # Debiasing experiment runner
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### Supported Models
The following models are supported in `models.py`:
- **Llama 3.1**: `llama3_1--8b`, `llama3_1--70b`
- **Gemma 2**: `gemma2--9b`
- **Qwen 2**: `qwen2--7b`
- **EXAONE**: `exaone--8b`
- **OpenAI**: `openai--gpt-4o-mini`

### Running Experiments

#### 1. Baseline Experiments
```bash
# Run baseline experiments on all configured models
bash run_baseline.sh

# Or run individual experiments
python run_deontology.py --model llama3_1--8b
python run_commonsense.py --model llama3_1--8b
```

#### 2. Debiasing Experiments
```bash
# Run debiasing experiments with few-shot learning
bash run_debias.sh

# Or run individual debiasing experiments
python run_reduce_effects_deontology.py --model llama3_1--8b --prompt-version "fewshot_2_2"
python run_reduce_effects_commonsense.py --model llama3_1--8b --prompt-version "logicshot_2_2"
```

#### 3. Negation Experiments
```bash
# Run experiments with negation
python run_deontology_negation.py --model llama3_1--8b
python run_commonsense_negation.py --model llama3_1--8b
```

### Experiment Configuration
Edit the model list in `run_baseline.sh` or `run_debias.sh` to select which models to evaluate:

```bash
models=(
    "llama3_1--8b"
    "gemma2--9b"
    "qwen2--7b"
    # Add or comment out models as needed
)
```

### Prompt Versions for Debiasing
- `fewshot_X_Y`: Few-shot learning with X positive and Y negative examples
- `logicshot_X_Y`: In-context reasoning with X positive and Y negative examples

---

## 📊 Datasets

| Dataset | Description | Size |
|---------|-------------|------|
| `deontology.csv` | Deontic sentences with/without modal expressions | 446 samples |
| `commonsense.csv` | Non-deontic sentences for control experiments | 450 samples |
| `morality_high.csv` | High moral ambiguity cases | 451 samples |
| `morality_low.csv` | Low moral ambiguity cases | 451 samples |
| `*_negation.csv` | Negation variants of main datasets | ~450 samples each |



## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{park2025dkb,
  title={Deontological Keyword Bias: The Impact of Modal Expressions on Normative Judgments of Language Models},
  author={Park, Bumjin and Lee, Jinsil and Choi, Jaesik},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2025}
}
```
