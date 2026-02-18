# GCF Network Inference & Analysis

This repository contains the computational framework and analysis pipeline developed for my thesis. The project performs an end-to-end analysis of Gene Cluster Family (GCF) co-occurrence and co-abundance data using three distinct network inference methods: Jaccard Index, Graphical LASSO (GLASSO), and Spearman correlation.

## Repository Structure

To ensure clarity and reproducibility, the project is organized as follows:

- **/code**: 
  - `master.py`: The primary execution script that automates data processing, network inference, and visualization.
  - `01_...` to `06_...` (Jupyter Notebooks): A chronological record of notebooks used during the research and exploratory phase.
- **/data**: Targeted directory for curated metagenomics datasets.
- **/results_thesis**: The central directory for all final outputs:
  - **Visualizations**: High-resolution heatmaps and community composition bar charts.
  - **Network Files**: Exported `.graphml` files for use in software like Cytoscape.
  - `execution_summary.txt`: A technical log providing a transparent record of the final run.
- **/exported-comms-representatives-corr**: antismash results and clinker files for the selected case studies

---
## Data Availability & Manual Setup

Due to GitHub's file size limitations, the raw datasets are **not included** in this repository. To execute the pipeline, you must manually place the following files into the `/data` directory:

* `metalog_bgcs_with_gcf_and_tax.tsv`
* `mgnify_bgcs_with_gcf_and_tax.tsv`
* `metalog_samples.tsv`
* `mgnify_samples.tsv`

### Requirements
Ensure you have conda installed. Create and activate the environment using the provided file:
```bash
conda env create -f environment.yml
conda activate gcf_network_analysis
```
### Running the Pipeline
To reproduce the results presented in the thesis, navigate to the `/code` directory and execute the master file:

```bash
python master.py
```
