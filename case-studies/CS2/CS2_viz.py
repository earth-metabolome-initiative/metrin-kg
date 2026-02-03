import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from collections import Counter

SEP = "\t"

if len(sys.argv) < 2:
    print("Usage: python CS2_viz.py <input_file>")
    sys.exit(1)

input_file = sys.argv[1]

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

df = load_tsv(input_file)
#df = pd.read_csv('metrin-kg_ZVBghY.tsv', sep='\t', 
#                 names=['source_wdx', 'sourceName', 'tryDataLab', 'tryDataVal', 'unit', 'unitComment'])

print("=" * 80)
print("DATASET SUMMARY STATISTICS")
print("=" * 80)
print(f"Total records: {len(df):,}")
print(f"Unique species: {df['sourceName'].nunique():,}")
print(f"Unique traits: {df['tryDataLab'].nunique():,}")
print(f"Unique sources: {df['source_wdx'].nunique():,}")
print(f"\nAverage measurements per species: {len(df) / df['sourceName'].nunique():.1f}")
print(f"Average measurements per trait: {len(df) / df['tryDataLab'].nunique():.1f}")

# ============================================================================
# 1. MOST STUDIED PLANTS
# ============================================================================
print("\n" + "=" * 80)
print("TOP 20 MOST STUDIED SPECIES")
print("=" * 80)
species_counts = df['sourceName'].value_counts().head(20)
print(species_counts.to_string())

# ============================================================================
# 2. TRAIT DISTRIBUTION
# ============================================================================
print("\n" + "=" * 80)
print("TOP 20 MOST COMMON TRAITS")
print("=" * 80)
trait_counts = df['tryDataLab'].value_counts().head(20)
print(trait_counts.to_string())

# ============================================================================
# 3. VISUALIZATIONS
# ============================================================================

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Create figure with subplots
fig = plt.figure(figsize=(20, 16))

# Plot 1: Top 20 Most Studied Species
ax1 = plt.subplot(3, 2, 1)
top_species = df['sourceName'].value_counts().head(20)
ax1.barh(range(len(top_species)), top_species.values, color='#2ecc71')
ax1.set_yticks(range(len(top_species)))
ax1.set_yticklabels(top_species.index, fontsize=9)
ax1.set_xlabel('Number of Records', fontsize=11, fontweight='bold')
ax1.set_title('Top 20 Most Studied Species', fontsize=13, fontweight='bold', pad=15)
ax1.invert_yaxis()
for i, v in enumerate(top_species.values):
    ax1.text(v + max(top_species.values)*0.01, i, f' {v:,}', va='center', fontsize=8)

# Plot 2: Top 20 Most Common Traits
ax2 = plt.subplot(3, 2, 2)
top_traits = df['tryDataLab'].value_counts().head(20)
ax2.barh(range(len(top_traits)), top_traits.values, color='#3498db')
ax2.set_yticks(range(len(top_traits)))
ax2.set_yticklabels(top_traits.index, fontsize=9)
ax2.set_xlabel('Number of Records', fontsize=11, fontweight='bold')
ax2.set_title('Top 20 Most Common Traits', fontsize=13, fontweight='bold', pad=15)
ax2.invert_yaxis()
for i, v in enumerate(top_traits.values):
    ax2.text(v + max(top_traits.values)*0.01, i, f' {v:,}', va='center', fontsize=8)

# Plot 3: Distribution of measurements per species
ax3 = plt.subplot(3, 2, 3)
species_measurement_counts = df['sourceName'].value_counts()
ax3.hist(species_measurement_counts, bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
ax3.set_xlabel('Number of Measurements', fontsize=11, fontweight='bold')
ax3.set_ylabel('Number of Species', fontsize=11, fontweight='bold')
ax3.set_title('Distribution of Measurements per Species', fontsize=13, fontweight='bold', pad=15)
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)

# Plot 4: Distribution of measurements per trait
ax4 = plt.subplot(3, 2, 4)
trait_measurement_counts = df['tryDataLab'].value_counts()
ax4.hist(trait_measurement_counts, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
ax4.set_xlabel('Number of Measurements', fontsize=11, fontweight='bold')
ax4.set_ylabel('Number of Traits', fontsize=11, fontweight='bold')
ax4.set_title('Distribution of Measurements per Trait', fontsize=13, fontweight='bold', pad=15)
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3)

# Plot 5: Species with most trait diversity
ax5 = plt.subplot(3, 2, 5)
trait_diversity = df.groupby('sourceName')['tryDataLab'].nunique().sort_values(ascending=False).head(15)
ax5.barh(range(len(trait_diversity)), trait_diversity.values, color='#f39c12')
ax5.set_yticks(range(len(trait_diversity)))
ax5.set_yticklabels(trait_diversity.index, fontsize=9)
ax5.set_xlabel('Number of Unique Traits', fontsize=11, fontweight='bold')
ax5.set_title('Species with Most Trait Diversity', fontsize=13, fontweight='bold', pad=15)
ax5.invert_yaxis()
for i, v in enumerate(trait_diversity.values):
    ax5.text(v + max(trait_diversity.values)*0.01, i, f' {v}', va='center', fontsize=8)

# Plot 6: Unit distribution
ax6 = plt.subplot(3, 2, 6)
top_units = df['unit'].value_counts().head(15)
ax6.bar(range(len(top_units)), top_units.values, color='#1abc9c')
ax6.set_xticks(range(len(top_units)))
ax6.set_xticklabels(top_units.index, rotation=45, ha='right', fontsize=9)
ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax6.set_title('Top 15 Most Common Units', fontsize=13, fontweight='bold', pad=15)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('trait_distribution_overview.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: trait_distribution_overview.png")

# ============================================================================
# 4. HEATMAP: SPECIES vs TRAITS
# ============================================================================
print("\nGenerating species-trait heatmap...")

# Create binary matrix: 1 if species has trait measured, 0 otherwise
# Use top 30 species and top 30 traits for visibility
top_n_species = 30
top_n_traits = 30

top_species_list = df['sourceName'].value_counts().head(top_n_species).index
top_traits_list = df['tryDataLab'].value_counts().head(top_n_traits).index

# Filter dataframe
df_filtered = df[df['sourceName'].isin(top_species_list) & df['tryDataLab'].isin(top_traits_list)]

# Create pivot table with counts
heatmap_data = pd.crosstab(df_filtered['sourceName'], df_filtered['tryDataLab'])

# Reorder to match frequency
heatmap_data = heatmap_data.reindex(index=top_species_list, columns=top_traits_list, fill_value=0)

# Create heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(heatmap_data, 
            cmap='YlOrRd', 
            cbar_kws={'label': 'Number of Measurements'},
            linewidths=0.5,
            linecolor='lightgray',
            square=False)
plt.xlabel('Traits', fontsize=13, fontweight='bold')
plt.ylabel('Species', fontsize=13, fontweight='bold')
plt.title(f'Heatmap: Top {top_n_species} Species vs Top {top_n_traits} Traits\n(Color intensity = measurement count)', 
          fontsize=15, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('species_trait_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: species_trait_heatmap.png")

# ============================================================================
# 5. ADDITIONAL HEATMAP: Binary presence/absence
# ============================================================================
print("Generating binary presence/absence heatmap...")

# Create binary version (presence = 1, absence = 0)
heatmap_binary = (heatmap_data > 0).astype(int)

plt.figure(figsize=(20, 16))
sns.heatmap(heatmap_binary, 
            cmap='Blues', 
            cbar_kws={'label': 'Trait Measured (1=Yes, 0=No)'},
            linewidths=0.5,
            linecolor='lightgray',
            square=False)
plt.xlabel('Traits', fontsize=13, fontweight='bold')
plt.ylabel('Species', fontsize=13, fontweight='bold')
plt.title(f'Binary Heatmap: Trait Coverage for Top {top_n_species} Species\n(Blue = trait measured for this species)', 
          fontsize=15, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('species_trait_binary_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: species_trait_binary_heatmap.png")

# ============================================================================
# 6. EXPORT SUMMARY STATISTICS TO CSV
# ============================================================================
print("\nExporting summary statistics...")

# Species summary
species_summary = pd.DataFrame({
    'species': df['sourceName'].value_counts().index,
    'total_measurements': df['sourceName'].value_counts().values,
    'unique_traits': df.groupby('sourceName')['tryDataLab'].nunique().values
})
species_summary.to_csv('species_summary.csv', index=False)
print("✓ Saved: species_summary.csv")

# Trait summary
trait_summary = pd.DataFrame({
    'trait': df['tryDataLab'].value_counts().index,
    'total_measurements': df['tryDataLab'].value_counts().values,
    'unique_species': df.groupby('tryDataLab')['sourceName'].nunique().values
})
trait_summary.to_csv('trait_summary.csv', index=False)
print("✓ Saved: trait_summary.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  1. trait_distribution_overview.png - Six-panel overview")
print("  2. species_trait_heatmap.png - Measurement counts heatmap")
print("  3. species_trait_binary_heatmap.png - Binary presence/absence")
print("  4. species_summary.csv - Species statistics")
print("  5. trait_summary.csv - Trait statistics")
