# Phage Lifecycle Prediction — Hybrid Deep Learning Approach

🔬 **Live demo:** [https://huggingface.co/spaces/prav104/content](https://huggingface.co/spaces/prav104/content)

A deep learning web app that predicts whether a bacteriophage follows a **lytic (virulent)** or **lysogenic (temperate)** lifecycle, given its NCBI accession ID. Built with a hybrid CNN + Bidirectional LSTM model and deployed via Gradio on Hugging Face Spaces.

---

## Overview

Bacteriophages (phages) are viruses that infect bacteria. Their lifecycle strategy — lytic vs. lysogenic — is critical for applications like phage therapy. This tool automates lifecycle classification directly from a phage genome sequence fetched from NCBI, using a trained hybrid deep learning model.

- **Lytic/Virulent:** The phage rapidly replicates and lyses the host cell. Preferred for phage therapy.
- **Lysogenic/Temperate:** The phage integrates into host DNA and enters a dormant state. Less suitable for therapy.

---

## Demo

Enter a phage NCBI accession ID (e.g., `NC_000866.4`) to get:

- Predicted lifecycle (Virulent / Temperate)
- Confidence score
- Genome length
- First 200 bp DNA preview
- Phage name
- Biological explanation
- Lifecycle diagram

---

## How It Works

1. **Sequence Fetching** — The app queries NCBI's Entrez API using the provided accession ID and retrieves the FASTA sequence.
2. **Dual Encoding** — The sequence is encoded two ways in parallel:
   - **CNN input:** 3-mer k-mer index encoding across the full sequence window. Every consecutive triplet of nucleotides is mapped to one of 64 possible integer indices, capturing local sequence composition.
   - **Bi-LSTM input:** Sliding-window GC content and GC skew signals (500 bp windows, 250 bp step), producing a two-dimensional time-series that reflects how the genome's compositional character changes across its length.
3. **Sliding Window Prediction** — For long genomes, the model runs over 50,000 bp windows with 25,000 bp overlap and averages the probabilities across all windows to produce a final prediction.
4. **Classification** — A threshold of 0.5 on the averaged probability determines the lifecycle label. Confidence is power-scaled (exponent 0.6) and capped at 95% to reflect the model's known uncertainty.

---

## Model Architecture

The model is a **hybrid CNN with Attention + Bidirectional LSTM** architecture, trained on phage genome data with focal loss to handle class imbalance.

### Architecture Overview

| Branch | Components | Purpose |
|--------|-----------|---------|
| CNN branch | `Conv1D` → `Attention` → `GlobalMaxPooling1D` + `GlobalAveragePooling1D` | Extracts local k-mer sequence motifs; Attention focuses the model on biologically relevant positions; dual pooling captures both peak signals and overall composition |
| Bi-LSTM branch | `Bidirectional(LSTM)` | Processes the GC content/skew time-series in both forward and backward directions, capturing genomic structural patterns regardless of reading direction |
| Fusion | `Concatenate` → `Dense` → `Dense` (sigmoid) | Merges both branch representations for final classification |

The **Bidirectional LSTM** processes the GC signal in both directions simultaneously, which is important because genomic signals like replication strand bias don't have a single meaningful reading direction. The **Attention layer** on the CNN branch allows the model to weight positions in the sequence differently, rather than treating all k-mers uniformly.

Focal loss (γ=2, α=0.5) was used during training to down-weight the abundant easy-to-classify examples (temperate phages) and force the model to focus on the harder, underrepresented virulent class.

### Model Files

| File | Description |
|------|-------------|
| `phage_lifecycle_model.keras` | Final deployed model (best performer) |
| `phage_lifecycle_model_Adam_B16_A32.keras` | Variant: Adam optimizer, batch size 16, 32 Bi-LSTM units |
| `phage_lifecycle_model_Adam_B16_A64.keras` | Variant: Adam optimizer, batch size 16, 64 Bi-LSTM units |
| `phage_lifecycle_model_Adam_B32_A32.keras` | Variant: Adam optimizer, batch size 32, 32 Bi-LSTM units |
| `phage_lifecycle_model_SGD_B16_A32.keras` | Variant: SGD optimizer, batch size 16, 32 Bi-LSTM units |
| `label_encoder.pkl` | Fitted sklearn LabelEncoder — maps integer predictions back to "Virulent" / "Temperate" strings |
| `all_histories.pkl` | Per-epoch training and validation loss/accuracy histories for all four experiments, used for learning curve analysis |
| `final_metrics.pkl` | Final test-set evaluation metrics (accuracy, precision, recall, F1, confusion matrix) for all four model variants |

The `Adam_B16_A64` variant achieved the best test accuracy (61.5%) and highest virulent precision (0.857) and was selected as the deployed model.

---

## Project Structure

```
├── app.py                          # Gradio app (main entry point)
├── requirements.txt                # Python dependencies
├── phage_lifecycle_model.keras     # Deployed model
├── label_encoder.pkl               # Label encoder
├── lifecycle.png                   # Lifecycle diagram shown in UI
├── lifestyle_annotation.xlsx       # Raw dataset: accession IDs + lifecycle labels
├── purified_dataset.pkl            # Cleaned and deduplicated dataset
├── processed_data.pkl              # Fully encoded CNN + Bi-LSTM training data (~200 MB)
├── smote_processed_data.pkl        # SMOTE-augmented training data for class balancing (~184 MB)
├── all_histories.pkl               # Training histories across all four experiments
├── final_metrics.pkl               # Test-set evaluation metrics for all variants
└── .github/workflows/
    └── update_space.yml            # CI/CD: auto-deploy to Hugging Face Spaces on push
```

---

## Installation & Local Run

**Prerequisites:** Python 3.9+

```bash
# Clone the repo
git clone https://huggingface.co/spaces/prav104/content
cd content

# (Optional) create and activate a virtual environment
python -m venv env
source env/bin/activate        # Linux/macOS
env\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

The Gradio interface will launch at `http://localhost:7860`.

---

## Requirements

```
tensorflow-cpu
biopython
gradio
numpy
scikit-learn==1.6.1
```

---

## Deployment

This project is configured to auto-deploy to **Hugging Face Spaces** via GitHub Actions whenever changes are pushed to `main`.

To set this up yourself:
1. Create a Hugging Face Space with the Gradio SDK.
2. Add your Hugging Face token as a GitHub secret named `hf_token`.
3. Push to `main` — the workflow in `.github/workflows/update_space.yml` handles the rest.

---

## Notes

- NCBI queries use Entrez with a registered email. To use your own, update the `Entrez.email` field in `app.py`.
- Sequences shorter than 50,000 bp are padded with `N` bases and may yield less reliable predictions.
- scikit-learn is pinned to version 1.6.1 because the `label_encoder.pkl` was serialized with that version — loading it under a different version may cause errors.
