# GRIP-DFFI

**Graph-Reinforced Intelligent Personalization with Diffusion-Enhanced Federated Feature Intelligence**

GRIP-DFFI is a heterogeneous multi-corpus federated learning framework for binary Network Intrusion Detection Systems (NIDS). The framework addresses challenges arising from dataset heterogeneity, feature incompatibility, label inconsistency, client non-IID distributions, and cross-corpus generalization.

The proposed system integrates leakage-aware semantic governance, graph-reinforced feature intelligence, federated relevance aggregation, universality-guided shared/private routing, personalized federated learning, and a diffusion-enhanced transformer-inspired shared backbone to improve both predictive performance and reliability.

## Repository Contents

This repository contains:

* Complete implementation of the GRIP-DFFI framework
* Validation and reproducibility notebooks
* Figure and table generation scripts
* Robustness and ablation analyses
* Supplementary Document S1
* Experimental outputs and generated analysis materials

### Experiment Structure

Each numbered folder corresponds to a major validation study, including:

* Federated learning baselines
* Personalized federated learning validation
* Feature-selection and feature-ranking analyses
* Client heterogeneity studies
* Backbone and architecture validation
* Diffusion and routing validation
* Classical machine learning comparisons
* Cross-corpus generalization
* Split-ratio robustness
* Multi-seed statistical validation
* Missing-data and corruption robustness
* Label-noise robustness
* Hash-bucket sensitivity analysis
* Cross-feature interaction studies

## Viewing Notebook Outputs

GitHub may not render large `.ipynb` files correctly.

For each experiment folder, open the file ending with:

`_OUTPUT_VIEW.md`

These files contain the notebook code, outputs, figures, and generated results in a GitHub-friendly format.

The `.ipynb` notebooks are retained as the original source files.

## Supplementary Material

**Supplementary Document S1** contains the complete validation tables, robustness analyses, and supporting experimental results referenced in the paper.

## Data Availability

The datasets used in this study are publicly available from their respective official sources and are cited in the manuscript.

This repository does **not** redistribute raw datasets. Instead, it provides:

* Official dataset references and links
* Download guidance
* Preprocessing configurations
* Validation scripts
* Reproducibility materials

allowing the complete experimental workflow to be reproduced from public sources.

## Citation

If you use GRIP-DFFI in your research, please cite the associated publication.
