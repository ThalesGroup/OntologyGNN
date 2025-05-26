# OntologyGNN

Interpretable GNN model for ontology based data.

## 📊 Model Architecture

![GNN Architecture](img/GNNmodel.png)

---

## Description

The model is based on Graph Attention Model to get output predictions for an Ontology based data task (classification/regression). It is also designed to detect important sub-graphs/communities within the ontology graph which are important for the model's predictions.

## Dataset

The code works with two data sources - the Titanic data for suvival prediction, and the TCGA gene expression data for cancer classification

TCGA dataset can be downloaded from [GDC portal](https://portal.gdc.cancer.gov/). 

## Usage

To work with the above two datasets, the user needs to structure the data files in the data directory as:

--data
    --titanic
        --ontology_file (ttl, rdf, owl)
        --feature-to-class_map file (json)
    --tcga
        --matrix_connection_truncated.csv
        --tcga.npz
        --tcga_graph.pickle

### Example on Titanic dataset:

#### Train

```bash
python3 main.py --dataset data/titanic --n_communities 3 --epochs 100
```

Optional - specify ontology or feature-to-class mappings files separately. If not provided, model will use default files based in config.yaml

```bash
python3 main.py --dataset data/titanic --n_communities 3 --epochs 100 --ontology_file ontology.ttl
```

### Example on TCGA dataset:


#### Train

<!-- On the microarray dataset:
```bash
python3 scripts/GraphGONet.py --save --n_inputs=36834 --n_nodes=10663 --n_nodes_annotated=8249 --n_classes=1 --selection_op="top" --selection_ratio=0.001 --n_epochs=50 --es --patience=5 --class_weight 
```
-->

```bash
python3 main.py --dataset data/TCGA --n_communities 3 --epochs 100
```


#### Train the model with a small number of training samples

```bash
python3 main.py --dataset data/TCGA --n_communities 3 --epochs 100 --n_samples 1000
```

#### To run on cluster

```bash
sbatch train_job.slurm
```

###  notebooks

Please see the notebook entitled *ontologyGNN.ipynb* (located in the notebooks directory) to run the model in a jupyter notebook. 