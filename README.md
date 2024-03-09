# local_curvature_profile

## Requirements
To configure and activate the conda environment for this repository, run
```
conda env create -f environment.yml
conda activate borf 
pip install -r requirements.txt
```

## Experiments
### 1. For graph classification
To run experiments for the TUDataset benchmark, run the file ```run_graph_classification.py```. The following command will run the benchmark with the LCP based on the ORC:
```bash
python run_graph_classification.py --encoding LCP
```

To use a different model or add more layers, add the --layer_type and --num_layers options
```bash
python run_graph_classification.py --encoding LCP --layer_type GIN \
	--num_layers 8
```

### 2. For node classification
To run node classification, simply change the script name to `run_node_classification.py`. For example:
```bash
python run_graph_classification.py --encoding LCP
```

## Other encoding methods
To compare the LCP against other encoding methods, simply run
```bash
# runs graph classification with Laplacian Eigenvector Positional Encodings
python run_graph_classification.py --encoding LAPE
```

## Citation and reference
For technical details and full experiment results, please check [our paper](https://arxiv.org/abs/2311.14864).
```
@inproceedings{fesser2023effective,
  title={Effective Structural Encodings via Local Curvature Profiles},
  author={Fesser, Lukas and Weber, Melanie},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```
