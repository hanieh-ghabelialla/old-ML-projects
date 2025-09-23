"""
IBM Transaction Data to GAE Format Converter

This utility script provides functions to convert IBM transaction data 
to Modularity-Aware GAE compatible format.

Usage:
    python ibm_to_gae_converter.py --input HI-Small_Trans.csv --output gae_data/
"""

import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
import argparse
import os
import pickle


def create_sample_transaction_data():
    """Create a sample transaction dataset for demonstration purposes."""
    np.random.seed(42)
    n_transactions = 1000
    n_accounts = 200
    
    # Generate sample data
    data = {
        'Timestamp': pd.date_range('2020-01-01', periods=n_transactions, freq='H'),
        'From Bank': np.random.choice(['Bank_A', 'Bank_B', 'Bank_C'], n_transactions),
        'Account': np.random.choice(range(1, n_accounts+1), n_transactions),
        'To Bank': np.random.choice(['Bank_A', 'Bank_B', 'Bank_C'], n_transactions),
        'Account.1': np.random.choice(range(1, n_accounts+1), n_transactions),
        'Amount Received': np.random.lognormal(8, 2, n_transactions),
        'Receiving Currency': np.random.choice(['USD', 'EUR', 'GBP'], n_transactions),
        'Amount Paid': np.random.lognormal(8, 2, n_transactions),
        'Payment Currency': np.random.choice(['USD', 'EUR', 'GBP'], n_transactions),
        'Payment Format': np.random.choice(['Cash', 'Transfer', 'Check'], n_transactions),
        'Is Laundering': np.random.choice([0, 1], n_transactions, p=[0.95, 0.05])
    }
    
    df = pd.DataFrame(data)
    print("Created sample transaction dataset for demonstration.")
    return df


def load_transaction_data(file_path):
    """Load IBM transaction dataset."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} transactions from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        print("Using sample data for demonstration.")
        return create_sample_transaction_data()


def create_transaction_graph(df, min_transactions=2):
    """Convert transaction dataframe to NetworkX graph."""
    G = nx.DiGraph()
    account_stats = {}
    
    print(f"Processing {len(df)} transactions...")
    
    for _, row in df.iterrows():
        # Create unique account identifiers
        from_account = f"Bank_{row['From Bank']}_Acc_{row['Account']}"
        to_account = f"Bank_{row['To Bank']}_Acc_{row['Account.1']}"
        
        amount = row['Amount Received']
        is_laundering = row.get('Is Laundering', 0)
        
        # Add/update edge
        if G.has_edge(from_account, to_account):
            G[from_account][to_account]['weight'] += amount
            G[from_account][to_account]['count'] += 1
            G[from_account][to_account]['laundering_count'] += is_laundering
        else:
            G.add_edge(from_account, to_account, 
                      weight=amount, count=1, laundering_count=is_laundering)
        
        # Update account statistics
        for account in [from_account, to_account]:
            if account not in account_stats:
                account_stats[account] = {
                    'total_amount': 0, 'transaction_count': 0,
                    'laundering_involvement': 0, 'currencies': set(),
                    'payment_formats': set()
                }
            
            stats = account_stats[account]
            stats['total_amount'] += amount
            stats['transaction_count'] += 1
            stats['laundering_involvement'] += is_laundering
            stats['currencies'].add(row['Receiving Currency'])
            stats['payment_formats'].add(row['Payment Format'])
    
    # Filter accounts by minimum transaction threshold
    accounts_to_keep = [acc for acc, stats in account_stats.items() 
                       if stats['transaction_count'] >= min_transactions]
    
    G_filtered = G.subgraph(accounts_to_keep).copy()
    print(f"Graph created: {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")
    
    return G_filtered, account_stats


def extract_node_features(G, account_stats):
    """Extract feature vectors for each node."""
    nodes = list(G.nodes())
    features = []
    labels = []
    
    # Calculate centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G)
    
    print(f"Extracting features for {len(nodes)} nodes...")
    
    for node in nodes:
        stats = account_stats.get(node, {})
        
        # Basic features
        total_amount = stats.get('total_amount', 0)
        transaction_count = stats.get('transaction_count', 0)
        avg_amount = total_amount / max(transaction_count, 1)
        laundering_involvement = stats.get('laundering_involvement', 0)
        laundering_ratio = laundering_involvement / max(transaction_count, 1)
        
        # Diversity features
        currency_diversity = len(stats.get('currencies', set()))
        payment_diversity = len(stats.get('payment_formats', set()))
        
        # Network features
        degree_cent = degree_centrality.get(node, 0)
        betweenness_cent = betweenness_centrality.get(node, 0)
        pagerank_score = pagerank.get(node, 0)
        
        # Degree features
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        
        # Weight features
        in_weight = sum([G[pred][node]['weight'] for pred in G.predecessors(node)])
        out_weight = sum([G[node][succ]['weight'] for succ in G.successors(node)])
        
        feature_vector = [
            np.log1p(total_amount),
            np.log1p(avg_amount),
            transaction_count,
            laundering_ratio,
            currency_diversity,
            payment_diversity,
            degree_cent,
            betweenness_cent,
            pagerank_score,
            in_degree,
            out_degree,
            np.log1p(in_weight),
            np.log1p(out_weight),
            out_degree / max(in_degree, 1),
            np.log1p(abs(out_weight - in_weight))
        ]
        
        features.append(feature_vector)
        labels.append(1 if laundering_involvement > 0 else 0)
    
    return np.array(features), np.array(labels), nodes


def convert_to_gae_format(G, features, labels, nodes):
    """Convert to GAE-compatible sparse matrices."""
    n_nodes = len(nodes)
    node_mapping = {node: i for i, node in enumerate(nodes)}
    
    # Create adjacency matrix
    adj_matrix = np.zeros((n_nodes, n_nodes))
    for edge in G.edges(data=True):
        from_idx = node_mapping[edge[0]]
        to_idx = node_mapping[edge[1]]
        weight = edge[2]['weight']
        
        # Make symmetric for undirected version
        adj_matrix[from_idx, to_idx] = weight
        adj_matrix[to_idx, from_idx] = weight
    
    adj_sparse = sp.csr_matrix(adj_matrix)
    features_sparse = sp.csr_matrix(features)
    
    return adj_sparse, features_sparse, labels, node_mapping


def save_gae_data(adj_matrix, feature_matrix, labels, node_mapping, output_dir):
    """Save data in GAE-compatible format."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save matrices
    sp.save_npz(f'{output_dir}/adjacency_matrix.npz', adj_matrix)
    sp.save_npz(f'{output_dir}/feature_matrix.npz', feature_matrix)
    np.save(f'{output_dir}/node_labels.npy', labels)
    
    with open(f'{output_dir}/node_mapping.pkl', 'wb') as f:
        pickle.dump(node_mapping, f)
    
    # Create edge list
    rows, cols = adj_matrix.nonzero()
    with open(f'{output_dir}/edgelist.txt', 'w') as f:
        for i, j in zip(rows, cols):
            if i < j:  # Avoid duplicates
                f.write(f"{i} {j}\n")
    
    print(f"Data saved to {output_dir}/")
    print(f"- Adjacency matrix: {adj_matrix.shape}")
    print(f"- Feature matrix: {feature_matrix.shape}")
    print(f"- Labels: {len(labels)} ({labels.sum()} suspicious accounts)")


def main():
    parser = argparse.ArgumentParser(description='Convert IBM transaction data to GAE format')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', default='gae_data', help='Output directory')
    parser.add_argument('--min_transactions', type=int, default=2, 
                       help='Minimum transactions per account')
    
    args = parser.parse_args()
    
    # Load and process data
    df = load_transaction_data(args.input)
    if df is None:
        return
    
    # Create graph
    G, account_stats = create_transaction_graph(df, args.min_transactions)
    
    # Extract features
    features, labels, nodes = extract_node_features(G, account_stats)
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    # Convert to GAE format
    adj_matrix, feature_matrix, labels, node_mapping = convert_to_gae_format(
        G, features_normalized, labels, nodes
    )
    
    # Save data
    save_gae_data(adj_matrix, feature_matrix, labels, node_mapping, args.output)
    
    print("\nConversion complete! You can now use this data with GAE.")
    print("Next steps:")
    print("1. Copy the output directory to your GAE repository")
    print("2. Add the IBM dataset case to input_data.py")
    print("3. Run GAE training with --dataset=ibm_transactions")


if __name__ == "__main__":
    main()