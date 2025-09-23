# IBM Transaction Dataset + Modularity-Aware GAE Integration

This repository provides a complete pipeline for integrating the IBM Transactions for Anti-Money Laundering (AML) dataset with the Modularity-Aware Graph Autoencoder framework.

## Overview

The IBM AML dataset contains financial transaction records that need to be converted into graph format for use with Graph Neural Networks. This integration enables:

- **Anti-Money Laundering Detection**: Identify suspicious transaction patterns
- **Community Detection**: Find clusters of related accounts
- **Link Prediction**: Predict future transaction relationships
- **Anomaly Detection**: Detect unusual transaction behaviors

## Key Features

- **Automatic Graph Construction**: Convert tabular transaction data to graph format
- **Feature Engineering**: Extract meaningful account-level features
- **GAE Compatibility**: Format data for direct use with GAE models
- **AML-Specific Adaptations**: Handle financial domain requirements

## Files Generated

1. `ibm_transaction_gae_integration.ipynb` - Main processing notebook
2. `run_gae_with_ibm_data.py` - Script to run GAE with processed data
3. `gae_modifications.txt` - Required modifications for GAE code
4. `gae_data/` - Processed data in GAE-compatible format

## Complete Integration Workflow

### 1. Download the IBM AML Dataset
- Go to: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
- Download `HI-Small_Trans.csv` or similar files
- Place the CSV file in the same directory as the notebook

### 2. Run the Integration Notebook
- Open `ibm_transaction_gae_integration.ipynb`
- Execute all cells to process the transaction data
- This creates the `gae_data/` folder with processed files

### 3. Set up the GAE Repository
```bash
git clone https://github.com/GuillaumeSalhaGalvan/modularity_aware_gae
cd modularity_aware_gae
python setup.py install
```

### 4. Copy Processed Data
```bash
cp -r gae_data/ modularity_aware_gae/
```

### 5. Modify GAE Code
- Open `modularity_aware_gae/input_data.py`
- Add the IBM data loading cases using the code in `gae_modifications.txt`
- This adds support for the 'ibm_transactions' dataset

### 6. Run GAE Training
```bash
cd modularity_aware_gae
python train.py --dataset=ibm_transactions --features=True --task=task_2 --model=linear_vae --iterations=300 --learning_rate=0.01 --hidden=32 --dimension=16 --beta=0.5 --lamb=0.75 --gamma=0.5 --s_reg=2 --fastgae=False --nb_run=1
```

## Graph Construction Strategy

The integration converts transaction data into an **Account-to-Account Graph**:

- **Nodes**: Bank accounts (identified by bank + account number)
- **Edges**: Transaction relationships with weights based on transaction amounts
- **Features**: Account-level features including:
  - Transaction volume and frequency
  - Network centrality measures (degree, betweenness, PageRank)
  - Risk indicators (laundering involvement ratio)
  - Behavioral patterns (currency diversity, payment formats)

## Feature Engineering

Each account (node) is characterized by 15 features:
1. Log-transformed total transaction amount
2. Log-transformed average transaction amount
3. Transaction count
4. Laundering involvement ratio
5. Currency diversity
6. Payment format diversity
7. Degree centrality
8. In-degree centrality
9. Out-degree centrality
10. Betweenness centrality
11. PageRank score
12. Degree ratio (out/in)
13. Weight balance (outgoing - incoming)
14. In-degree
15. Out-degree

## Model Parameters for AML Detection

Recommended GAE parameters for transaction data:
- **Model**: `linear_vae` (good starting point)
- **Features**: `True` (use extracted features)
- **Task**: `task_2` (joint community detection and link prediction)
- **Beta**: `0.5` (modularity regularization)
- **Lambda**: `0.75` (community loss weight)
- **Gamma**: `0.5` (additional regularization)

## Expected Results

The GAE model will provide:
1. **Embeddings**: Low-dimensional representations of accounts
2. **Community Detection**: Groups of related accounts
3. **Link Prediction**: Probability of future transactions
4. **Anomaly Scores**: Accounts with unusual patterns

## Troubleshooting

### Common Issues:
1. **File not found**: Ensure the IBM dataset CSV is in the correct location
2. **Memory issues**: Reduce the dataset size or use `fastgae=True` for large graphs
3. **TensorFlow version**: GAE requires TensorFlow 1.x (install with `pip install tensorflow==1.15`)

### Performance Tips:
- Start with smaller subsets of the data
- Adjust the `min_transactions` parameter to filter low-activity accounts
- Use GPU acceleration for faster training
- Experiment with different GAE model types (linear_ae, gcn_vae, etc.)

## Next Steps

After successful integration:
1. **Hyperparameter Tuning**: Optimize GAE parameters for your specific dataset
2. **Evaluation**: Use the generated embeddings for downstream AML tasks
3. **Visualization**: Plot the graph structure and learned embeddings
4. **Production**: Scale to full dataset and deploy for real-time detection

## Citation

If you use this integration in your research, please cite:

- The original GAE paper: Salha-Galvan et al. "Modularity-Aware Graph Autoencoders for Joint Community Detection and Link Prediction" (2022)
- The IBM AML dataset: IBM "Transactions for Anti Money Laundering (AML)" on Kaggle

## Support

For questions or issues with this integration, please refer to:
- The main GAE repository: https://github.com/GuillaumeSalhaGalvan/modularity_aware_gae
- The IBM AML dataset documentation on Kaggle
- The integration notebook for detailed implementation