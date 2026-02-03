import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from collections import Counter
import sys
import re
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "png"

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

# Comprehensive list of fungal genera and species that should NOT be allelopaths
# These are fungi/oomycetes that are parasites themselves, not allelopathic plants
FUNGAL_TAXA = [
    # Major fungal genera commonly seen in the data
    'Puccinia', 'Fusarium', 'Claviceps', 'Alternaria', 'Bipolaris', 'Cochliobolus',
    'Cladosporium', 'Eudarluca', 'Sclerotinia', 'Trichothecium', 'Rhizoctonia',
    'Ustilaginoidea', 'Colletotrichum', 'Thanatephorus', 'Leptosphaeria', 
    'Peronospora', 'Periconia', 'Tuberculina', 'Corticium', 'Nectria', 'Phoma',
    'Trichoderma', 'Pithomyces', 'Nigrospora', 'Anthracocystis', 'Ustilago',
    'Tranzschelia', 'Dibotryon', 'Armillaria', 'Cryptococcus', 'Eutypella',
    'Erysiphe', 'Albugo', 'Podosphaera', 'Cicinobolus', 'Sphaerellopsis',
    'Exserohilum', 'Leptosphaerulina', 'Myrothecium', 'Neonectria',
    
    # Specific species that are clearly fungi
    'Fusarium graminearum', 'Fusarium heterosporum', 'Fusarium oxysporum',
    'Fusarium acuminatum', 'Fusarium avenaceum', 'Fusarium proliferatum',
    'Claviceps purpurea', 'Puccinia coronata', 'Puccinia graminis',
    'Puccinia pruni-spinosae', 'Puccinia asparagi', 'Puccinia magnusiana',
    'Puccinia purpurea', 'Puccinia poae-sudeticae', 'Puccinia malvacearum',
    'Sclerotinia sclerotiorum', 'Rhizoctonia solani', 'Thanatephorus cucumeris',
    'Colletotrichum gloeosporioides', 'Colletotrichum graminicola',
    'Alternaria alternata', 'Bipolaris cynodontis', 'Bipolaris zeae',
    'Bipolaris bicolor', 'Cochliobolus bicolor', 'Cochliobolus eragrostidis',
    'Cladosporium herbarum', 'Cladosporium oxysporum', 'Cladosporium uredinicola',
    'Cladosporium aecidiicola', 'Trichothecium roseum', 'Pithomyces chartarum',
    'Ustilaginoidea virens', 'Ustilago avenae', 'Armillaria mellea',
    'Dibotryon morbosum', 'Tranzschelia pruni-spinosae', 'Tuberculina persicina',
    'Nectria episphaeria', 'Nectria coccinea', 'Erysiphe polygoni',
    'Peronospora parasitica', 'Albugo candida', 'Anthracocystis flocculosa',
    'Eudarluca australis', 'Eudarluca caricis', 'Periconia byssoides',
    'Periconia echinochloae', 'Leptosphaeria michotii', 'Corticium solani',
    'Cryptococcus fagi', 'Eutypella', 'Podosphaera leucotricha',
    'Cicinobolus cesatii', 'Trichoderma viride', 'Nigrospora sphaerica',
    'Sphaerellopsis filum', 'Exserohilum rostratum', 'Phoma sorghina',
    'Leptosphaerulina trifolii', 'Myrothecium roridum', 'Neonectria faginata'
]

def is_fungus(allelopath_name):
    """
    Check if the allelopath name is a fungus based on taxonomy
    Returns True if it's a fungus (should be removed)
    """
    if pd.isna(allelopath_name):
        return False
    
    allelopath_name = str(allelopath_name).strip()
    
    # Check if any fungal taxon appears in the name
    for fungal_taxon in FUNGAL_TAXA:
        if fungal_taxon.lower() in allelopath_name.lower():
            return True
    
    return False

# Load your TSV file
if len(sys.argv) < 2:
    print("Usage: python CS4_viz.py <input_file>")
    sys.exit(1)

input_file = sys.argv[1]
df = pd.read_csv(input_file, sep='\t', skipinitialspace=True)

# Read the data from file


# Clean up column names - remove leading/trailing spaces and question marks
df.columns = df.columns.str.strip().str.replace('?', '', regex=False)

print(f"Columns found: {len(df.columns)}")
print(df.columns.tolist())

# Clean up data - strip whitespace and clean labels
print("\nCleaning labels (removing ^^, <...> content, and quotes)...")
for col in df.columns:
    #if df[col].dtype == 'object':
        # Show before cleaning (first non-null row)
    if col in ['parasiteName', 'allelopathName', 'agriCropName']:
        try:
            first_valid = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            if first_valid:
                print(f"  Before cleaning {col}: {first_valid}")
        except:
            pass
        
    # Apply cleaning to all values (clean_label handles NaN internally)
    df[col] = df[col].apply(clean_label)
        
    # Show after cleaning (first non-null row)
    if col in ['parasiteName', 'allelopathName', 'agriCropName']:
        try:
            first_valid = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            if first_valid:
                print(f"  After cleaning {col}:  {first_valid}")
        except:
            pass

print("✓ Label cleaning complete!")

# Show initial statistics
print("\n" + "=" * 80)
print("INITIAL DATASET (before filtering fungi)")
print("=" * 80)
print(f"Total interactions: {len(df)}")
print(f"Unique allelopathic plants (including fungi): {df['allelopathName'].nunique()}")

# Identify fungal rows
df['is_fungal_allelopath'] = df['allelopathName'].apply(is_fungus)
fungal_rows = df[df['is_fungal_allelopath'] == True]

print(f"\nFungal organisms incorrectly labeled as allelopaths: {fungal_rows['allelopathName'].nunique()}")
print(f"Interactions with fungal 'allelopaths' (to be removed): {len(fungal_rows)}")

print("\nFungal organisms identified (will be removed):")
for i, fungus in enumerate(sorted(fungal_rows['allelopathName'].unique()), 1):
    count = len(fungal_rows[fungal_rows['allelopathName'] == fungus])
    print(f"  {i:2d}. {fungus:50s} ({count:3d} interactions)")

# Filter out fungal allelopaths
df_filtered = df[df['is_fungal_allelopath'] == False].copy()
df_filtered = df_filtered.drop('is_fungal_allelopath', axis=1)

# Basic Statistics after filtering
print("\n" + "=" * 80)
print("FILTERED DATASET (after removing fungal allelopaths)")
print("=" * 80)
print(f"Total interactions: {len(df_filtered)}")
print(f"Unique parasites: {df_filtered['parasiteName'].nunique()}")
print(f"Unique allelopathic plants (true plants only): {df_filtered['allelopathName'].nunique()}")
print(f"Unique agricultural crops: {df_filtered['agriCropName'].nunique()}")

# DOI analysis
study2_valid = df_filtered[df_filtered['study2_DOI'] != '-']['study2_DOI']
unique_dois = study2_valid.nunique()
print(f"Documented studies (DOIs): {unique_dois}")

# Check if we have data after filtering
if len(df_filtered) == 0:
    print("\nWARNING: No data remaining after filtering! All allelopaths were identified as fungi.")
    print("Please check the fungal taxa list or your data.")
    sys.exit(1)

print("\n" + "=" * 80)
print("Creating comprehensive visualizations and network analysis...")
print("=" * 80)

# Set style
sns.set_style("whitegrid")

print("Note: All labels have been cleaned (removed ^^, <...>, and quotes)")

# Create aggregated edge data for Sankey - labels are already cleaned in df_filtered
parasite_plant_flow = df_filtered.groupby(['parasiteName', 'allelopathName']).size().reset_index(name='value')
plant_crop_flow = df_filtered.groupby(['allelopathName', 'agriCropName']).size().reset_index(name='value')

# Create unique node lists - these are already cleaned
all_parasites = list(df_filtered['parasiteName'].unique())
all_plants = list(df_filtered['allelopathName'].unique())
all_crops = list(df_filtered['agriCropName'].unique())

# Verify labels are clean (just for debugging)
print(f"Sample parasite name: {all_parasites[0] if all_parasites else 'None'}")
print(f"Sample plant name: {all_plants[0] if all_plants else 'None'}")
print(f"Sample crop name: {all_crops[0] if all_crops else 'None'}")

# Create node list with indices
all_nodes = all_parasites + all_plants + all_crops
node_dict = {node: idx for idx, node in enumerate(all_nodes)}

# Create color mapping
parasite_color = 'rgba(214, 39, 40, 0.6)'  # Red
plant_color = 'rgba(44, 160, 44, 0.6)'      # Green
crop_color = 'rgba(255, 127, 14, 0.6)'      # Orange

node_colors = [parasite_color] * len(all_parasites) + [plant_color] * len(all_plants) + [crop_color] * len(all_crops)

# Create links
sources = []
targets = []
values = []
link_colors = []

# Parasite -> Plant links
for _, row in parasite_plant_flow.iterrows():
    sources.append(node_dict[row['parasiteName']])
    targets.append(node_dict[row['allelopathName']])
    values.append(row['value'])
    link_colors.append('rgba(214, 39, 40, 0.2)')

# Plant -> Crop links
for _, row in plant_crop_flow.iterrows():
    sources.append(node_dict[row['allelopathName']])
    targets.append(node_dict[row['agriCropName']])
    values.append(row['value'])
    link_colors.append('rgba(44, 160, 44, 0.2)')

# Create Sankey diagram
fig_sankey = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=all_nodes,
        color=node_colors
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color=link_colors
    )
)])

fig_sankey.update_layout(
    title_text="Allelopathy Tripartite Network: Parasite → True Allelopathic Plant → Protected Crop<br>" +
               "<sub>Red: Parasites | Green: Allelopathic Plants (FUNGI REMOVED) | Orange: Protected Crops</sub>",
    font_size=10,
    height=1000,
    width=1800
)

# Save as static image
try:
    fig_sankey.write_image("sankey_tripartite_full.svg", width=1800, height=1000)
    print("✓ Full tripartite Sankey diagram saved: sankey_tripartite_full.svg")
except Exception as e:
    print(f"⚠ Could not save PNG (kaleido may not be installed): {e}")
    print("  Saving as HTML instead...")
    fig_sankey.write_html('sankey_tripartite_full.html')
    print("✓ Full tripartite Sankey diagram saved: sankey_tripartite_full.html")


parasites = df_filtered['parasiteName'].unique()
plants = df_filtered['allelopathName'].unique()
crops = df_filtered['agriCropName'].unique()
# Save summary
print("\nSaving summary statistics...")
with open('analysis_summary.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("ALLELOPATHY NETWORK ANALYSIS - SUMMARY (FUNGI REMOVED)\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Original interactions: {len(df)}\n")
    f.write(f"Fungal 'allelopaths' removed: {len(fungal_rows)}\n")
    f.write(f"Final interactions: {len(df_filtered)}\n\n")
    f.write(f"Unique parasites: {len(parasites)}\n")
    f.write(f"Unique allelopathic plants (true plants): {len(plants)}\n")
    f.write(f"Unique crops: {len(crops)}\n")

# Save filtered dataset
df_filtered.to_csv('filtered_data_no_fungi.tsv', sep='\t', index=False)
print("✓ Filtered dataset saved: filtered_data_no_fungi.tsv")

print(f"\nRemoved {len(fungal_rows)} interactions with fungal 'allelopaths'")
print(f"Retained {len(df_filtered)} interactions with true plant allelopaths")
print("\n  Note: To save Sankey diagrams as PNG, install: pip install kaleido")
