import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import re
import sys
import graphviz

# Load your TSV file
if len(sys.argv) < 2:
    print("Usage: python CS3_viz.py <input_file>")
    sys.exit(1)

input_file = sys.argv[1]
df = pd.read_csv(input_file, sep='\t')

def clean_label(label):
    # remove ^^ and content between < and > from labels
    
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return label

    label = str(label)
    
    # Remove content between < and > (including the brackets)
    label = re.sub(r'<[^>]*>', '', label)
    
    # Remove ^^
    label = label.replace('^^', '')
    
    # Remove quotes
    label = label.replace('"', '')
    label = label.replace("'", '')
    
    # Remove extra whitespace
    label = ' '.join(label.split())    
    
    return label.strip()


# Apply cleaning to species names
df['sourceName'] = df['sourceName'].apply(clean_label)
df['targetName'] = df['targetName'].apply(clean_label)
df['intxnName'] = df['intxnName'].apply(clean_label)

print("=" * 80)
print("INTERACTION NETWORK SUMMARY")
print("=" * 80)
print(f"Total interactions: {len(df):,}")
print(f"Unique source species: {df['sourceName'].nunique():,}")
print(f"Unique target species: {df['targetName'].nunique():,}")
print(f"Unique interaction types: {df['intxnName'].nunique():,}")

# ============================================================================
# CREATE NETWORK GRAPH
# ============================================================================
print("\n" + "=" * 80)
print("BUILDING INTERACTION NETWORK...")
print("=" * 80)

# Create directed graph
G = nx.DiGraph()

# Add edges with attributes
for _, row in df.iterrows():
    G.add_edge(row['sourceName'], row['targetName'], 
               interaction=row['intxnName'],
               location=row['loc'])

print(f"Network nodes (species): {G.number_of_nodes()}")
print(f"Network edges (interactions): {G.number_of_edges()}")

# Calculate network metrics
degree_dict = dict(G.degree())
top_connected = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:20]

print("\nTop 20 Most Connected Species:")
for i, (species, degree) in enumerate(top_connected, 1):
    print(f"{i:2d}. {species:60s} {degree:3d} connections")

# ============================================================================
# VISUALIZATION: FULL NETWORK WITH IMPROVED AESTHETICS
# ============================================================================
print("\nGenerating network visualization...")

plt.subplots(layout="constrained")
plt.figure(figsize=(28, 24), facecolor='white')

# Use spring layout with optimized parameters for better spacing
#pos = nx.spring_layout(G, k=3, iterations=100, seed=42)

pos = nx.nx_agraph.graphviz_layout(G, prog="sfdp", args="-Goverlap=prism -Gsplines=true")

# Calculate node sizes based on degree (much larger scaling)
node_sizes = [500 + degree_dict[node] * 200 for node in G.nodes()]

# Create color gradient based on degree
node_degrees = [degree_dict[node] for node in G.nodes()]
max_degree = max(node_degrees)
node_colors = [degree / max_degree for degree in node_degrees]

# Draw nodes with gradient coloring
nodes = nx.draw_networkx_nodes(G, pos, 
                               node_size=node_sizes,
                               node_color=node_colors,
                               cmap='YlOrRd',
                               alpha=0.85,
                               edgecolors='darkblue',
                               linewidths=2.5,
                               vmin=0,
                               vmax=1)

# Count edges between nodes to scale edge width
edge_weights = {}
for edge in G.edges():
    edge_weights[edge] = edge_weights.get(edge, 0) + 1

# Draw edges with increased width
edge_widths = [2.5 + edge_weights.get(edge, 1) * 0.5 for edge in G.edges()]

nx.draw_networkx_edges(G, pos, 
                       edge_color='#2c3e50',
                       alpha=0.4,
                       arrows=True,
                       arrowsize=20,
                       arrowstyle='->',
                       connectionstyle='arc3,rad=0.1',
                       width=edge_widths,
                       node_size=node_sizes)

# Label only top 20 most connected nodes with larger, clearer labels
top_nodes = [node for node, degree in top_connected]
labels = {node: node for node in top_nodes}

# Draw labels with white background box for better readability
nx.draw_networkx_labels(G, pos, labels, 
                       font_size=11, 
                       font_weight='bold',
                       font_color='black',
                       bbox=dict(boxstyle='round,pad=0.4', 
                                facecolor='white', 
                                edgecolor='darkblue',
                                alpha=0.9,
                                linewidth=1.5))

# Add colorbar
cbar = plt.colorbar(nodes, label='Relative Connectivity', shrink=0.8, pad=0.02)
cbar.ax.tick_params(labelsize=11)
cbar.set_label('Relative Connectivity', fontsize=13, fontweight='bold')

# Title and formatting
plt.title('Plant Interaction Network for Diterpenoid-Producing Species\n' + 
          f'(Network: {G.number_of_nodes()} species, {G.number_of_edges()} interactions | ' +
          'Node size & color = connection count | Only top 20 species labeled)', 
          fontsize=18, fontweight='bold', pad=30)

plt.axis('off')
#plt.tight_layout()

plt.savefig('interaction_network.png', dpi=300, bbox_inches='tight', facecolor='white')
print("\n✓ Saved: interaction_network.png")

# ============================================================================
# EXPORT NETWORK STATISTICS
# ============================================================================
print("\nExporting network statistics...")

# Node statistics with cleaned names
node_stats = pd.DataFrame({
    'species': list(G.nodes()),
    'total_connections': [degree_dict[node] for node in G.nodes()],
    'out_degree': [G.out_degree(node) for node in G.nodes()],
    'in_degree': [G.in_degree(node) for node in G.nodes()]
}).sort_values('total_connections', ascending=False)
node_stats.to_csv('network_statistics.csv', index=False)
print("✓ Saved: network_statistics.csv")

print("\n" + "=" * 80)
print("NETWORK ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  1. interaction_network.png - Full network visualization")
print("  2. network_statistics.csv - Node connection statistics")
