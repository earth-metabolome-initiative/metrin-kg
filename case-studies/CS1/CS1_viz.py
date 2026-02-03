import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch

SEP = "\t"

if len(sys.argv) < 4:
    print("Usage: python CS1_viz.py <interactions_file> <traits_file> <metabolites_file>")
    sys.exit(1)

INTERACTIONS_FILE = sys.argv[1]
TRAITS_FILE = sys.argv[2]
METABOLITES_FILE = sys.argv[3]


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

def load_tsv(path):
    # load and clean TSV file
    try:
        df = pd.read_csv(path, sep=SEP)
    except FileNotFoundError:
        sys.exit(f"❌ Missing file: {path}")
    
    for c in df.columns:
        df[c] = df[c].apply(clean_label)
    df.columns = df.columns.str.strip().str.replace('?', '', regex=False)
    return df


def build_network(interactions, traits, metabolites, common_only=False):
    """Build network graph with species as central nodes
    
    Args:
        common_only: If True, only include species present in all three datasets
    """
    G = nx.Graph()
    
    # Collect species from each dataset
    species_traits = set(traits["sourceName"].dropna())
    species_metabolites = set(metabolites["sourceName"].dropna())
    species_interactions = set(interactions["sourceName"].dropna()) | set(interactions["targetName"].dropna())
    
    if common_only:
        # Only keep species present in ALL three datasets
        all_species = species_traits & species_metabolites & species_interactions
        print(f"Found {len(all_species)} species common to all three datasets")
    else:
        # Use all species from any dataset
        all_species = species_traits | species_metabolites | species_interactions
        print(f"Found {len(all_species)} total species across all datasets")
    
    # Add species nodes
    for species in all_species:
        G.add_node(species, node_type='species')
    
    # Add trait edges (trait -> species)
    for _, row in traits.iterrows():
        species = row["sourceName"]
        trait = row.get("tryDataLab", None)
        if pd.notna(species) and pd.notna(trait) and species in all_species:
            G.add_edge(species, trait, edge_type='trait')
    
    # Add metabolite edges (species -> metabolite)
    for _, row in metabolites.iterrows():
        species = row["sourceName"]
        metabolite = row.get("wd_chem", None)
        if pd.notna(species) and pd.notna(metabolite) and species in all_species:
            G.add_edge(species, metabolite, edge_type='metabolite')
    
    # Add interaction edges (species -> species)
    for _, row in interactions.iterrows():
        source = row["sourceName"]
        target = row["targetName"]
        if pd.notna(source) and pd.notna(target) and source in all_species and target in all_species:
            G.add_edge(source, target, edge_type='interaction')
    
    return G, all_species


def get_species_data_completeness(species, traits_df, metabolites_df, interactions_df):
    """Calculate data completeness score for a species"""
    score = 0
    if species in traits_df["sourceName"].values:
        score += 1
    if species in metabolites_df["sourceName"].values:
        score += 1
    if species in interactions_df["sourceName"].values or species in interactions_df["targetName"].values:
        score += 1
    return score


def plot_clean_network(G, all_species, traits_df, metabolites_df, interactions_df, filter_top=False, top_n=20):
    """Create a tripartite columnar layout visualization
    
    Args:
        filter_top: If True, filter to top N species by data completeness
        top_n: Number of top species to show (only used if filter_top=True)
    """
    
    # Separate edges by type
    trait_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'trait']
    metabolite_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'metabolite']
    interaction_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'interaction']
    
    # Separate nodes by type
    species_nodes = list(all_species)
    trait_nodes = [n for n in G.nodes() if n not in all_species and 
                   any(G.get_edge_data(s, n, {}).get('edge_type') == 'trait' or 
                       G.get_edge_data(n, s, {}).get('edge_type') == 'trait' for s in all_species)]
    metabolite_nodes = [n for n in G.nodes() if n not in all_species and n not in trait_nodes]
    
    # Optionally filter to top N species
    if filter_top:
        species_scores = {s: get_species_data_completeness(s, traits_df, metabolites_df, interactions_df) 
                         for s in species_nodes}
        top_species = sorted(species_scores.items(), key=lambda x: (-x[1], x[0]))[:top_n]
        top_species_names = [s[0] for s in top_species]
        title_suffix = f"Top {top_n} Most Data-Rich Species"
        filename = f"tripartite_network_top{top_n}.png"
    else:
        top_species_names = species_nodes
        species_scores = {s: 3 for s in species_nodes}  # All have complete data
        title_suffix = "Common Species Across All Datasets"
        filename = "tripartite_network_common_species.png"
    
    # Filter traits and metabolites connected to selected species
    filtered_traits = [t for t in trait_nodes if any(G.has_edge(s, t) or G.has_edge(t, s) for s in top_species_names)]
    filtered_metabolites = [m for m in metabolite_nodes if any(G.has_edge(s, m) or G.has_edge(m, s) for s in top_species_names)]
    
    # Create position dictionary for tripartite layout
    pos = {}
    
    # Left column: Traits
    trait_spacing = 1.0
    for i, trait in enumerate(sorted(filtered_traits)):
        pos[trait] = (0, i * trait_spacing)
    
    # Middle column: Species (sized by data completeness)
    species_spacing = (len(filtered_traits) * trait_spacing) / (len(top_species_names) + 1) if len(top_species_names) > 0 else 1.0
    for i, species in enumerate(sorted(top_species_names)):
        pos[species] = (2, i * species_spacing)
    
    # Right column: Metabolites
    metabolite_spacing = (len(filtered_traits) * trait_spacing) / (len(filtered_metabolites) + 1) if len(filtered_metabolites) > 0 else 1.0
    for i, metabolite in enumerate(sorted(filtered_metabolites)):
        pos[metabolite] = (4, i * metabolite_spacing)
    
    # Filter edges to only those involving selected species
    filtered_trait_edges = [(u, v) for u, v in trait_edges if u in top_species_names or v in top_species_names]
    filtered_metabolite_edges = [(u, v) for u, v in metabolite_edges if u in top_species_names or v in top_species_names]
    filtered_interaction_edges = [(u, v) for u, v in interaction_edges if u in top_species_names and v in top_species_names]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 12))
    
    # Draw edges first (so they appear behind nodes)
    nx.draw_networkx_edges(G, pos, edgelist=filtered_trait_edges, 
                          edge_color='#3498DB', width=1.5, alpha=0.4)
    
    nx.draw_networkx_edges(G, pos, edgelist=filtered_metabolite_edges, 
                          edge_color='#2ECC71', width=1.5, alpha=0.4)
    
    nx.draw_networkx_edges(G, pos, edgelist=filtered_interaction_edges, 
                          edge_color='#E74C3C', width=2, alpha=0.5)
    
    # Draw nodes with size based on data completeness for species
    species_sizes = [species_scores.get(s, 1) * 800 for s in sorted(top_species_names)]
    
    nx.draw_networkx_nodes(G, pos, nodelist=sorted(top_species_names), 
                          node_color='#E74C3C', node_size=species_sizes, 
                          alpha=0.7, edgecolors='black', linewidths=1.5)
    
    # Count how many species connect to each trait
    trait_counts = {t: sum(1 for s in top_species_names if G.has_edge(s, t) or G.has_edge(t, s)) 
                   for t in filtered_traits}
    trait_sizes = [trait_counts.get(t, 1) * 300 for t in sorted(filtered_traits)]
    
    nx.draw_networkx_nodes(G, pos, nodelist=sorted(filtered_traits), 
                          node_color='#3498DB', node_size=trait_sizes,
                          alpha=0.7, node_shape='s', edgecolors='black', linewidths=1.5)
    
    # Count how many species connect to each metabolite
    metabolite_counts = {m: sum(1 for s in top_species_names if G.has_edge(s, m) or G.has_edge(m, s)) 
                        for m in filtered_metabolites}
    metabolite_sizes = [metabolite_counts.get(m, 1) * 300 for m in sorted(filtered_metabolites)]
    
    nx.draw_networkx_nodes(G, pos, nodelist=sorted(filtered_metabolites), 
                          node_color='#2ECC71', node_size=metabolite_sizes,
                          alpha=0.7, node_shape='^', edgecolors='black', linewidths=1.5)
    
    # Draw labels
    # Clean up species names
    species_labels = {s: s.split('^^')[0].replace('<http://www.w3.org/2001/XMLSchema#string>', '').strip('"<>')
                     for s in sorted(top_species_names)}
    nx.draw_networkx_labels(G, pos, species_labels, font_size=8, font_weight='bold')
    
    # Clean up trait names
    trait_labels = {t: t.split('/')[-1].replace('>', '').replace('_', ' ')[:30] 
                   for t in sorted(filtered_traits)}
    trait_pos = {k: (v[0] - 0.3, v[1]) for k, v in pos.items() if k in filtered_traits}
    nx.draw_networkx_labels(G, trait_pos, trait_labels, font_size=7, 
                           horizontalalignment='right')
    
    # Clean up metabolite names
    metabolite_labels = {m: m.split('/')[-1].replace('>', '')[:20] 
                        for m in sorted(filtered_metabolites)}
    metabolite_pos = {k: (v[0] + 0.3, v[1]) for k, v in pos.items() if k in filtered_metabolites}
    nx.draw_networkx_labels(G, metabolite_pos, metabolite_labels, font_size=7,
                           horizontalalignment='left')
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C', 
               markersize=12, label=f'Species (size = data completeness)', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#3498DB', 
               markersize=10, label=f'Traits (size = # species)', markeredgecolor='black'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#2ECC71', 
               markersize=10, label=f'Metabolites (size = # species)', markeredgecolor='black'),
        Line2D([0], [0], color='#3498DB', linewidth=2, label='Has trait'),
        Line2D([0], [0], color='#2ECC71', linewidth=2, label='Produces metabolite'),
        Line2D([0], [0], color='#E74C3C', linewidth=2, label='Biotic interaction'),
    ]
    
    plt.legend(handles=legend_elements, loc='upper left', framealpha=0.95, fontsize=11)
    plt.title(f"Integrated Network: {title_suffix}\n(Traits, Metabolites, and Biotic Interactions)", 
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"NETWORK SUMMARY ({title_suffix})")
    print(f"{'='*60}")
    print(f"Nodes:")
    print(f"  - Species: {len(top_species_names)}")
    print(f"  - Traits: {len(filtered_traits)}")
    print(f"  - Metabolites: {len(filtered_metabolites)}")
    print(f"\nEdges:")
    print(f"  - Trait edges: {len(filtered_trait_edges)}")
    print(f"  - Metabolite edges: {len(filtered_metabolite_edges)}")
    print(f"  - Interaction edges: {len(filtered_interaction_edges)}")
    
    if filter_top:
        print(f"\nTop species by data completeness:")
        for species, score in top_species[:10]:
            clean_name = species.split('^^')[0].strip('"<>')
            print(f"  {clean_name}: {score}/3 datasets")
    
    print(f"{'='*60}\n")


def main():
    # Load data
    print("Loading data files...")
    interactions = load_tsv(INTERACTIONS_FILE)
    traits = load_tsv(TRAITS_FILE)
    metabolites = load_tsv(METABOLITES_FILE)
    print("✓ Data loaded successfully\n")
    
    # Build network using only common species
    print("Building network (common species only)...")
    G, all_species = build_network(interactions, traits, metabolites, common_only=True)
    print("✓ Network built successfully\n")
    
    # Plot network with all common species (no filtering)
    print("Creating visualization...")
    plot_clean_network(G, all_species, traits, metabolites, interactions, filter_top=False)


if __name__ == "__main__":
    main()
