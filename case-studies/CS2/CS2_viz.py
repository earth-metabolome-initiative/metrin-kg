import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import sys
from matplotlib.patches import Patch

SEP = "\t"

if len(sys.argv) < 2:
    print("Usage: python CS2_viz.py <input_file>")
    sys.exit(1)

input_file = sys.argv[1]

def clean_label(label):
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return label
    label = str(label)
    label = re.sub(r'<[^>]*>', '', label)
    label = label.replace('^^', '').replace('"', '').replace("'", '')
    label = ' '.join(label.split())
    return label.strip()

def load_tsv(path):
    try:
        df = pd.read_csv(path, sep=SEP)
    except FileNotFoundError:
        sys.exit(f"Missing file: {path}")
    for c in df.columns:
        df[c] = df[c].apply(clean_label)
    df.columns = df.columns.str.strip().str.replace('?', '', regex=False)
    return df

df = load_tsv(input_file)

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

fig = plt.figure(figsize=(24, 18))

ax1 = plt.subplot(3, 2, 1)
top_species = df['sourceName'].value_counts().head(20)
ax1.barh(range(len(top_species)), top_species.values, color='#2ecc71', edgecolor='black', linewidth=0.5)
ax1.set_yticks(range(len(top_species)))
ax1.set_yticklabels(top_species.index, fontsize=12, weight='normal')
ax1.set_xlabel('Number of Records', fontsize=16, fontweight='bold')
ax1.invert_yaxis()
for i, v in enumerate(top_species.values):
    ax1.text(v + max(top_species.values)*0.01, i, f' {v:,}', va='center', fontsize=11, weight='normal')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2 = plt.subplot(3, 2, 2)
top_traits = df['tryDataLab'].value_counts().head(20)
ax2.barh(range(len(top_traits)), top_traits.values, color='#3498db', edgecolor='black', linewidth=0.5)
ax2.set_yticks(range(len(top_traits)))
ax2.set_yticklabels(top_traits.index, fontsize=12, weight='normal')
ax2.set_xlabel('Number of Records', fontsize=16, fontweight='bold')
ax2.invert_yaxis()
for i, v in enumerate(top_traits.values):
    ax2.text(v + max(top_traits.values)*0.01, i, f' {v:,}', va='center', fontsize=11, weight='normal')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax3 = plt.subplot(3, 2, 3)
species_measurement_counts = df['sourceName'].value_counts()
ax3.hist(species_measurement_counts, bins=50, color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=0.8)
ax3.set_xlabel('Number of Measurements', fontsize=16, fontweight='bold')
ax3.set_ylabel('Number of Species', fontsize=16, fontweight='bold')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3, linewidth=0.5)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

ax4 = plt.subplot(3, 2, 4)
trait_measurement_counts = df['tryDataLab'].value_counts()
ax4.hist(trait_measurement_counts, bins=50, color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.8)
ax4.set_xlabel('Number of Measurements', fontsize=16, fontweight='bold')
ax4.set_ylabel('Number of Traits', fontsize=16, fontweight='bold')
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3, linewidth=0.5)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

ax5 = plt.subplot(3, 2, 5)
trait_diversity = df.groupby('sourceName')['tryDataLab'].nunique().sort_values(ascending=False).head(15)
ax5.barh(range(len(trait_diversity)), trait_diversity.values, color='#f39c12', edgecolor='black', linewidth=0.5)
ax5.set_yticks(range(len(trait_diversity)))
ax5.set_yticklabels(trait_diversity.index, fontsize=12, weight='normal')
ax5.set_xlabel('Number of Unique Traits', fontsize=16, fontweight='bold')
ax5.invert_yaxis()
for i, v in enumerate(trait_diversity.values):
    ax5.text(v + max(trait_diversity.values)*0.01, i, f' {v}', va='center', fontsize=11, weight='normal')
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

ax6 = plt.subplot(3, 2, 6)
top_units = df['unit'].value_counts().head(15)
ax6.bar(range(len(top_units)), top_units.values, color='#1abc9c', edgecolor='black', linewidth=0.8)
ax6.set_xticks(range(len(top_units)))
ax6.set_xticklabels(top_units.index, rotation=45, ha='right', fontsize=12, weight='normal')
ax6.set_ylabel('Frequency', fontsize=16, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y', linewidth=0.5)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('trait_distribution_overview.pdf', dpi=600, bbox_inches='tight')
plt.savefig('trait_distribution_overview.png', dpi=600, bbox_inches='tight')
print("Saved: trait_distribution_overview.pdf/.png")

top_n_species = 30
top_n_traits = 30
top_species_list = df['sourceName'].value_counts().head(top_n_species).index
top_traits_list = df['tryDataLab'].value_counts().head(top_n_traits).index
df_filtered = df[df['sourceName'].isin(top_species_list) & df['tryDataLab'].isin(top_traits_list)]
heatmap_data = pd.crosstab(df_filtered['sourceName'], df_filtered['tryDataLab'])
heatmap_data = heatmap_data.reindex(index=top_species_list, columns=top_traits_list, fill_value=0)

plt.figure(figsize=(24, 18))
sns.heatmap(heatmap_data, cmap='YlOrRd', cbar_kws={'label': 'Measurement Count', 'shrink': 0.8},
            linewidths=0.5, linecolor='lightgray', square=False)
plt.xlabel('Traits', fontsize=18, fontweight='bold')
plt.ylabel('Species', fontsize=18, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=13, weight='normal')
plt.yticks(rotation=0, fontsize=13, weight='normal')
plt.tight_layout()
plt.savefig('species_trait_heatmap.pdf', dpi=600, bbox_inches='tight')
plt.savefig('species_trait_heatmap.png', dpi=600, bbox_inches='tight')
print("Saved: species_trait_heatmap.pdf/.png")

heatmap_binary = (heatmap_data > 0).astype(int)
fig, ax = plt.subplots(figsize=(24, 18))
colors = ['white', '#2E86AB']
cmap = sns.color_palette(colors, 2)
sns.heatmap(heatmap_binary, cmap=cmap, cbar=False, linewidths=0.5, linecolor='lightgray', square=False, ax=ax)
legend_elements = [Patch(facecolor='white', edgecolor='black', label='Trait Absent'),
                   Patch(facecolor='#2E86AB', edgecolor='black', label='Trait Present')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=16, frameon=True, 
          fancybox=True, shadow=True, bbox_to_anchor=(1.15, 1))
ax.set_xlabel('Traits', fontsize=18, fontweight='bold')
ax.set_ylabel('Species', fontsize=18, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=13, weight='normal')
plt.yticks(rotation=0, fontsize=13, weight='normal')
plt.tight_layout()
plt.savefig('species_trait_binary_heatmap.pdf', dpi=600, bbox_inches='tight')
plt.savefig('species_trait_binary_heatmap.png', dpi=600, bbox_inches='tight')
print("Saved: species_trait_binary_heatmap.pdf/.png")

species_summary = pd.DataFrame({
    'species': df['sourceName'].value_counts().index,
    'total_measurements': df['sourceName'].value_counts().values,
    'unique_traits': df.groupby('sourceName')['tryDataLab'].nunique().values
})
species_summary.to_csv('species_summary.csv', index=False)

trait_summary = pd.DataFrame({
    'trait': df['tryDataLab'].value_counts().index,
    'total_measurements': df['tryDataLab'].value_counts().values,
    'unique_species': df.groupby('tryDataLab')['sourceName'].nunique().values
})
trait_summary.to_csv('trait_summary.csv', index=False)
print("Saved: species_summary.csv, trait_summary.csv")
