# GCF Network Inference & Analysis

This repository contains the computational framework and analysis pipeline developed for my thesis. The project performs an end-to-end analysis of Gene Cluster Family (GCF) co-occurrence and co-abundance data using three distinct network inference methods: Jaccard Index, Graphical LASSO (GLASSO), and Spearman correlation.

## Repository Structure

To ensure clarity and reproducibility, the project is organized as follows:

- **/code**: 
  - `master.py`: The primary execution script that automates data processing, network inference, and visualization.
  - `01_...`, `02_...` (Jupyter Notebooks): A chronological record of Jupyter Notebooks used during the initial research and exploratory phase.
- **/data**: Contains the curated and harmonized metagenomics datasets (GCF tables and metadata).
- **/results_thesis**: The central directory for all final outputs:
  - **Visualizations**: High-resolution heatmaps and community composition bar charts.
  - **Network Files**: Exported `.graphml` files for use in software like Cytoscape.
  - `execution_summary.txt`: A technical log providing a transparent record of the final run, including modularity scores and matrix dimensions.

### Running the Pipeline
To reproduce the results presented in the thesis, navigate to the `/code` directory and execute the master file:

```bash
python master.py
