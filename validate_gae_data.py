"""
Validation script for IBM Transaction + GAE Integration

This script validates that the generated data is in the correct format
for use with the Modularity-Aware GAE framework.
"""

import numpy as np
import scipy.sparse as sp
import pickle
import os


def validate_gae_data(data_dir="gae_data"):
    """Validate that GAE data is correctly formatted."""
    
    print(f"Validating GAE data in {data_dir}/...")
    
    # Check if all required files exist
    required_files = [
        'adjacency_matrix.npz',
        'feature_matrix.npz', 
        'node_labels.npy',
        'node_mapping.pkl',
        'edgelist.txt'
    ]
    
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f"❌ Missing file: {file}")
            return False
        else:
            print(f"✅ Found: {file}")
    
    try:
        # Load and validate adjacency matrix
        adj = sp.load_npz(os.path.join(data_dir, 'adjacency_matrix.npz'))
        print(f"✅ Adjacency matrix: {adj.shape} (sparsity: {1-adj.nnz/(adj.shape[0]*adj.shape[1]):.3f})")
        
        # Load and validate features
        features = sp.load_npz(os.path.join(data_dir, 'feature_matrix.npz'))
        print(f"✅ Feature matrix: {features.shape}")
        
        # Load and validate labels
        labels = np.load(os.path.join(data_dir, 'node_labels.npy'))
        print(f"✅ Labels: {len(labels)} nodes ({labels.sum()} suspicious)")
        
        # Load and validate node mapping
        with open(os.path.join(data_dir, 'node_mapping.pkl'), 'rb') as f:
            node_mapping = pickle.load(f)
        print(f"✅ Node mapping: {len(node_mapping)} nodes")
        
        # Validate dimensions match
        if adj.shape[0] == features.shape[0] == len(labels) == len(node_mapping):
            print("✅ All dimensions match correctly")
        else:
            print("❌ Dimension mismatch detected")
            return False
        
        # Check if adjacency matrix is symmetric (for undirected graph)
        if not np.allclose(adj.data, adj.T.data):
            print("⚠️  Warning: Adjacency matrix is not symmetric")
        else:
            print("✅ Adjacency matrix is properly symmetric")
            
        print(f"\n🎉 Data validation successful!")
        print(f"Ready for GAE training with {adj.shape[0]} nodes and {features.shape[1]} features")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate GAE data format')
    parser.add_argument('--data_dir', default='gae_data', help='Data directory to validate')
    
    args = parser.parse_args()
    
    if validate_gae_data(args.data_dir):
        print("\n✅ Integration ready! You can now:")
        print("1. Copy this data to your GAE repository")
        print("2. Add IBM dataset support to input_data.py")
        print("3. Run GAE training")
    else:
        print("\n❌ Please fix the issues above before proceeding")