import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import re
import sys
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python CS3_viz.py <input_file>")
    sys.exit(1)

input_file = sys.argv[1]
df = pd.read_csv(input_file, sep='\t')

def clean_label(label):
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return label
    label = str(label)
    label = re.sub(r'<[^>]*>', '', label)
    label = label.replace('^^', '').replace('"', '').replace("'", '')
    label = ' '.join(label.split())
    return label.strip()

df['sourceName'] = df['sourceName'].apply(clean_label)
df['targetName'] = df['targetName'].apply(clean_label)
df['intxnName'] = df['intxnName'].apply(clean_label)

G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_edge(row['sourceName'], row['targetName'], 
               interaction=row['intxnName'], location=row['loc'])

degree_dict = dict(G.degree())
top_connected = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:20]

plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['font.weight'] = 'bold'

plt.figure(figsize=(32, 28), facecolor='white')

try:
    pos = nx.nx_agraph.graphviz_layout(G, prog="sfdp", args="-Goverlap=prism -Gsplines=true")
except:
    pos = nx.spring_layout(G, k=3, iterations=100, seed=42)

node_sizes = [600 + degree_dict[node] * 250 for node in G.nodes()]
node_degrees = [degree_dict[node] for node in G.nodes()]
max_degree = max(node_degrees)
node_colors = [degree / max_degree for degree in node_degrees]

nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                               cmap='YlOrRd', alpha=0.9, edgecolors='#1a252f',
                               linewidths=3, vmin=0, vmax=1)

edge_weights = {}
for edge in G.edges():
    edge_weights[edge] = edge_weights.get(edge, 0) + 1

edge_widths = [3 + edge_weights.get(edge, 1) * 0.7 for edge in G.edges()]

nx.draw_networkx_edges(G, pos, edge_color='#2c3e50', alpha=0.45, arrows=True,
                       arrowsize=25, arrowstyle='->', connectionstyle='arc3,rad=0.1',
                       width=edge_widths, node_size=node_sizes)

top_nodes = [node for node, degree in top_connected]
labels = {node: node for node in top_nodes}

nx.draw_networkx_labels(G, pos, labels, font_size=14, font_weight='bold',
                       font_color='black',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                edgecolor='#1a252f', alpha=0.95, linewidth=2))

cbar = plt.colorbar(nodes, label='Relative Connectivity', shrink=0.7, pad=0.02, aspect=30)
cbar.ax.tick_params(labelsize=14, width=2, length=6)
cbar.set_label('Relative Connectivity', fontsize=18, fontweight='bold')
cbar.outline.set_linewidth(2)

plt.axis('off')
plt.tight_layout()

plt.savefig('interaction_network.pdf', dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig('interaction_network.png', dpi=600, bbox_inches='tight', facecolor='white')
print("Saved: interaction_network.pdf/.png")

node_stats = pd.DataFrame({
    'species': list(G.nodes()),
    'total_connections': [degree_dict[node] for node in G.nodes()],
    'out_degree': [G.out_degree(node) for node in G.nodes()],
    'in_degree': [G.in_degree(node) for node in G.nodes()]
}).sort_values('total_connections', ascending=False)
node_stats.to_csv('network_statistics.csv', index=False)
print("Saved: network_statistics.csv")
