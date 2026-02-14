import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import re
import plotly.graph_objects as go
import plotly.io as pio

# Try to import kaleido for PDF export
try:
    import kaleido
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False
    print("Warning: kaleido not installed. Install with: pip install kaleido")
    print("PDF/PNG export will be skipped. Only HTML will be generated.")

def clean_label(label):
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return label
    label = str(label)
    label = re.sub(r'<[^>]*>', '', label)
    label = label.replace('^^', '').replace('"', '').replace("'", '')
    label = ' '.join(label.split())
    return label.strip()

FUNGAL_TAXA = [
    'Puccinia', 'Fusarium', 'Claviceps', 'Alternaria', 'Bipolaris', 'Cochliobolus',
    'Cladosporium', 'Eudarluca', 'Sclerotinia', 'Trichothecium', 'Rhizoctonia',
    'Ustilaginoidea', 'Colletotrichum', 'Thanatephorus', 'Leptosphaeria', 
    'Peronospora', 'Periconia', 'Tuberculina', 'Corticium', 'Nectria', 'Phoma',
    'Trichoderma', 'Pithomyces', 'Nigrospora', 'Anthracocystis', 'Ustilago',
    'Tranzschelia', 'Dibotryon', 'Armillaria', 'Cryptococcus', 'Eutypella',
    'Erysiphe', 'Albugo', 'Podosphaera', 'Cicinobolus', 'Sphaerellopsis',
    'Exserohilum', 'Leptosphaerulina', 'Myrothecium', 'Neonectria',
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
    if pd.isna(allelopath_name):
        return False
    allelopath_name = str(allelopath_name).strip()
    for fungal_taxon in FUNGAL_TAXA:
        if fungal_taxon.lower() in allelopath_name.lower():
            return True
    return False

if len(sys.argv) < 2:
    print("Usage: python CS4_viz.py <input_file>")
    sys.exit(1)

input_file = sys.argv[1]
df = pd.read_csv(input_file, sep='\t', skipinitialspace=True)
df.columns = df.columns.str.strip().str.replace('?', '', regex=False)

for col in df.columns:
    df[col] = df[col].apply(clean_label)

df['is_fungal_allelopath'] = df['allelopathName'].apply(is_fungus)
fungal_rows = df[df['is_fungal_allelopath'] == True]
df_filtered = df[df['is_fungal_allelopath'] == False].copy()
df_filtered = df_filtered.drop('is_fungal_allelopath', axis=1)

if len(df_filtered) == 0:
    print("No data remaining after filtering fungi.")
    sys.exit(1)

plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['font.weight'] = 'bold'

parasite_plant_flow = df_filtered.groupby(['parasiteName', 'allelopathName']).size().reset_index(name='value')
plant_crop_flow = df_filtered.groupby(['allelopathName', 'agriCropName']).size().reset_index(name='value')

all_parasites = list(df_filtered['parasiteName'].unique())
all_plants = list(df_filtered['allelopathName'].unique())
all_crops = list(df_filtered['agriCropName'].unique())
all_nodes = all_parasites + all_plants + all_crops
node_dict = {node: idx for idx, node in enumerate(all_nodes)}

parasite_color = 'rgba(214, 39, 40, 0.7)'
plant_color = 'rgba(44, 160, 44, 0.7)'
crop_color = 'rgba(255, 127, 14, 0.7)'
node_colors = [parasite_color] * len(all_parasites) + [plant_color] * len(all_plants) + [crop_color] * len(all_crops)

sources = []
targets = []
values = []
link_colors = []

for _, row in parasite_plant_flow.iterrows():
    sources.append(node_dict[row['parasiteName']])
    targets.append(node_dict[row['allelopathName']])
    values.append(row['value'])
    link_colors.append('rgba(214, 39, 40, 0.25)')

for _, row in plant_crop_flow.iterrows():
    sources.append(node_dict[row['allelopathName']])
    targets.append(node_dict[row['agriCropName']])
    values.append(row['value'])
    link_colors.append('rgba(44, 160, 44, 0.25)')

fig_sankey = go.Figure(data=[go.Sankey(
    node=dict(pad=20, thickness=25, line=dict(color="black", width=1),
              label=all_nodes, color=node_colors,
              hovertemplate='%{label}<br>Count: %{value}<extra></extra>'),
    link=dict(source=sources, target=targets, value=values, color=link_colors,
              hovertemplate='%{source.label} → %{target.label}<br>Count: %{value}<extra></extra>')
)])

fig_sankey.update_layout(
    font=dict(size=22, family="Arial, sans-serif", color="black"),
    height=1200, 
    width=2000, 
    margin=dict(l=20, r=20, t=40, b=20),
    annotations=[
        dict(
            text='<b><span style="color:#d62728">■ Parasites</span></b>',
            xref="paper", yref="paper",
            x=0.75, y=0.95,
            xanchor='left', yanchor='top',
            showarrow=False,
            font=dict(size=22, family="Arial, sans-serif")
        ),
        dict(
            text='<b><span style="color:#2ca02c">■ Allelopathic Plants</span></b>',
            xref="paper", yref="paper",
            x=0.75, y=0.90,
            xanchor='left', yanchor='top',
            showarrow=False,
            font=dict(size=22, family="Arial, sans-serif")
        ),
        dict(
            text='<b><span style="color:#ff7f0e">■ Protected Crops</span></b>',
            xref="paper", yref="paper",
            x=0.75, y=0.85,
            xanchor='left', yanchor='top',
            showarrow=False,
            font=dict(size=22, family="Arial, sans-serif")
        )
    ]
)

if KALEIDO_AVAILABLE:
    try:
        fig_sankey.write_image("sankey_tripartite_full.svg", width=2000, height=1200)
        print("Saved: sankey_tripartite_full.svg")
    except Exception as e:
        print(f"SVG save failed: {e}")
    
    try:
        fig_sankey.write_image("sankey_tripartite_full.pdf", width=2000, height=1200, scale=2)
        print("Saved: sankey_tripartite_full.pdf")
    except Exception as e:
        print(f"PDF save failed: {e}")

    try:
        fig_sankey.write_image("sankey_tripartite_full.png", width=2000, height=1200, scale=3)
        print("Saved: sankey_tripartite_full.png")
    except Exception as e:
        print(f"PNG save failed: {e}")
else:
    print("Skipping SVG/PDF/PNG export (kaleido not available)")

fig_sankey.write_html('sankey_tripartite_full.html')
print("Saved: sankey_tripartite_full.html")

df_filtered.to_csv('filtered_data_no_fungi.tsv', sep='\t', index=False)
print("Saved: filtered_data_no_fungi.tsv")
