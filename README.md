# scBridge
[![DOI](https://zenodo.org/badge/642258769.svg)](https://zenodo.org/badge/latestdoi/642258769)
This is the official implementation of *scBridge embraces cell heterogeneity in single-cell RNA-seq and ATAC-seq data integration*.

## Installation
You may create an anaconda environment for scBridge with the following commands:
```bash
git clone https://github.com/XLearning-SCU/scBridge
cd scBridge
conda env create -f environment.yml
conda activate scBridge
```
Note: scBridge runs on a single GPU.

## Quick Start
### Data Preparation
scBridge accepts AnnData (https://anndata.readthedocs.io/en/latest/index.html) as inputs, including the source (scRNA-seq) and target (scATAC-seq) h5ad files. The source data should include the cell type annotation in `obs.["CellType"]`. The integration process does not require and would not utilize the annotation of the target data. If the target data annotation is provided, the model would evaluate the integration performance by computing metrics such as label transfer accuracy and silhouette score. Put the two h5ad files under the same folder (see the PBMC folder for example).

Note: Please unzip the example data with the `gunzip` command.

### Data Integration
To perform data integration, simply run the following command
```bash
python main.py --data_path="PBMC/" --source_data="CITE-seq.h5ad" --target_data="ASAP-seq.h5ad" --umap_plot
```
Here `--data_path` corresponds to the data folder, `--source_data` and `--target_data` correspond to the source and target h5ad data file names.

After training, the integrated results would be saved as two AnnData files under the same data folder as the inputs.

The integrated source data additionally includes
- `.obsm["Embedding"]` (integrated cell embedding)
- `.obsm["X_umap"]` (umap embedding).

The integrated target data additionally includes
- `.obsm["Embedding"]` (integrated cell embedding)
- `.obsm["X_umap"]` (umap embedding)
- `.obs["Prediction"]` (transferred label)
- `.obs["Reliability"]` (cell reliability).

The umap plot colored by cell type, domain, and reliability would be saved to the `figures` folder.

### Configs
In addition to `--data_path`, `--source_data`, and `--target_data` which must be provided to perform integration, there are some optional configs, including:
- `--umap_plot` enables umap visualization
- `--novel_type` enables novel type discovery with the structure loss, where most unreliable cells would be predicted as novel type

The following configs influence the integration performance, which we recommend leave them as the default:
- `--source_preprocess=["Standard", "TFIDF"]` (Default="Standard") the preprocessing strategy for source data
- `--target_preprocess=["Standard", "TFIDF"]` (Default="TFIDF") the preprocessing strategy for target data
- `--reliability_threshold=[0.0-1.0]` (Default=0.95) threshold for selecting reliable cells
- `--align_loss_epoch=[int]` (Default=1) when to enable alignment loss
- `--prototype_momentum=[0.0-1.0]` (Default=0.9) momentum for prototype update
- `--early_stop_acc=[0.0-1.0]` (Default=0.99) early stop training if source data classification accuracy meets the requirement
- `--max_iteration=[int]` (Default=20) maximum number of integration iterations
- `--batch_size=[int]` (Default=512) size of mini-batch
- `--train_epoch=[int]` (Default=20) training epochs
- `--learning_rate=[float]` (Default=5e-4) learning rate
- `--random_seed=[int]` (Default=2023) random seed
