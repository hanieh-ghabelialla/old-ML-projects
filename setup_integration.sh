#!/bin/bash
"""
Complete Integration Example: IBM Transactions + Modularity-Aware GAE

This script demonstrates the complete workflow for integrating 
IBM transaction data with the Modularity-Aware GAE framework.

Prerequisites:
1. Download IBM AML dataset from Kaggle
2. Clone the GAE repository
3. Install dependencies
"""

# Step 1: Convert IBM transaction data to GAE format
echo "Converting IBM transaction data to GAE format..."
python ibm_to_gae_converter.py --input HI-Small_Trans.csv --output gae_data

# Step 2: Clone and setup GAE repository (if not already done)
if [ ! -d "modularity_aware_gae" ]; then
    echo "Cloning GAE repository..."
    git clone https://github.com/GuillaumeSalhaGalvan/modularity_aware_gae
    cd modularity_aware_gae
    pip install -e .
    cd ..
fi

# Step 3: Copy processed data to GAE directory
echo "Copying processed data to GAE repository..."
cp -r gae_data/ modularity_aware_gae/

# Step 4: Patch GAE input_data.py (manual step required)
echo "Manual step required: Add IBM dataset support to input_data.py"
echo "Use the modifications in gae_modifications.txt"

# Step 5: Run GAE training with IBM data
echo "To run GAE training with IBM data, execute:"
echo "cd modularity_aware_gae"
echo "python train.py --dataset=ibm_transactions --features=True --task=task_2 --model=linear_vae --iterations=300 --learning_rate=0.01 --hidden=32 --dimension=16 --beta=0.5 --lamb=0.75 --gamma=0.5 --s_reg=2 --fastgae=False --nb_run=1"

echo "Integration setup complete!"