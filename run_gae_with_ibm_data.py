#!/usr/bin/env python3
"""
Example script to run Modularity-Aware GAE with IBM Transaction data.

Usage:
    1. Process IBM transaction data using the notebook
    2. Copy the generated gae_data/ folder to the GAE repository
    3. Add the load_ibm_transaction_data function to input_data.py
    4. Run this script
"""

import sys
import os

# Add the path to modularity_aware_gae
sys.path.append('path/to/modularity_aware_gae')

from modularity_aware_gae.train import main as train_gae
import tensorflow as tf

# Set up command line arguments for IBM transaction data
def run_gae_with_ibm_data():
    """
    Run GAE with IBM transaction data.
    """
    
    # Configure GAE parameters for transaction data
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    
    # Override default parameters
    FLAGS.dataset = 'ibm_transactions'  # You'll need to add this case to input_data.py
    FLAGS.features = True  # Use the extracted features
    FLAGS.task = 'task_2'  # Joint community detection and link prediction
    FLAGS.model = 'linear_vae'  # Start with linear VAE
    FLAGS.iterations = 300
    FLAGS.learning_rate = 0.01
    FLAGS.hidden = 32
    FLAGS.dimension = 16
    FLAGS.beta = 0.5  # Modularity regularization
    FLAGS.lamb = 0.75  # Community loss weight
    FLAGS.gamma = 0.5  # Additional regularization
    FLAGS.s_reg = 2  # Sparsity regularization
    FLAGS.fastgae = False
    FLAGS.nb_run = 1
    
    print("Running Modularity-Aware GAE with IBM Transaction Data...")
    print(f"Dataset: {FLAGS.dataset}")
    print(f"Model: {FLAGS.model}")
    print(f"Features: {FLAGS.features}")
    
    # Run the training
    train_gae()

if __name__ == "__main__":
    run_gae_with_ibm_data()