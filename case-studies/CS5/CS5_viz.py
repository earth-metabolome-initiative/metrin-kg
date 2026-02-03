#!/usr/bin/env python3
"""
Leaf Economic Spectrum (LES) Analysis Tool
Analyzes trait data and metabolite patterns across plant species
"""

import sys

# Check for required packages with helpful error messages
required_packages = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'scipy': 'scipy',
    'sklearn': 'scikit-learn'
}

missing_packages = []
for module_name, pip_name in required_packages.items():
    try:
        __import__(module_name)
    except ImportError:
        missing_packages.append(pip_name)

if missing_packages:
    print("ERROR: Missing required packages!")
    print(f"Please install: {', '.join(missing_packages)}")
    print("\nInstallation command:")
    print(f"  pip install {' '.join(missing_packages)}")
    print("\nOr install all at once:")
    print("  pip install pandas numpy matplotlib seaborn scipy scikit-learn")
    print("\nIMPORTANT: Use 'scikit-learn' not 'sklearn'!")
    sys.exit(1)

# Now import everything
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_trait_data(filepath):
    """Load and parse LES trait data"""
    try:
        # Try different separators and handle BOM
        df = None
        for sep in ['\t', ',', ';', '|']:
            try:
                df = pd.read_csv(filepath, sep=sep, encoding='utf-8-sig')  # Handle BOM
                if df.shape[1] > 1:  # Successfully parsed multiple columns
                    break
            except:
                try:
                    df = pd.read_csv(filepath, sep=sep, encoding='utf-8')
                    if df.shape[1] > 1:
                        break
                except:
                    continue
        
        if df is None or df.shape[1] == 1:
            print(f"ERROR: Could not parse file with standard separators")
            print(f"Tried: tab, comma, semicolon, pipe")
            print(f"Please ensure your file is properly formatted")
            sys.exit(1)
        
        # Clean column names - remove BOM, leading/trailing whitespace, and special characters
        df.columns = df.columns.str.replace('^\ufeff', '', regex=True)  # Remove BOM
        df.columns = df.columns.str.replace('^[?]+', '', regex=True)  # Remove leading ?
        df.columns = df.columns.str.strip()  # Remove whitespace
        
        print(f"Loaded trait data: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        
        # Show sample data
        #print("\nFirst few rows:")
        #print(df.head(3).to_string())
        
        return df
    except Exception as e:
        print(f"Error loading trait data: {e}")
        print(f"\nExpected format (TSV or CSV):")
        print("Required columns (can have different names):")
        print("  - Species ID column (e.g., 'source_wdx', 'species_id')")
        print("  - Species name column (e.g., 'sourceName', 'species')")
        print("  - Trait name column (e.g., 'traitLabel', 'traitCategory', 'trait')")
        print("  - Trait value column (e.g., 'tryDataVal', 'value', 'trait_value')")
        print("\nExample format:")
        print("source_wdx\tsourceName\ttraitLabel\ttryDataVal")
        print("Q1000821\tPterygota alata\tLeaf nitrogen content per dry mass (Nmass)\t2.28")
        sys.exit(1)


def load_metabolite_data(filepath):
    """Load and parse metabolite data"""
    try:
        # Try different separators and handle BOM
        df = None
        for sep in ['\t', ',', ';', '|']:
            try:
                df = pd.read_csv(filepath, sep=sep, encoding='utf-8-sig')  # Handle BOM
                if df.shape[1] > 1:
                    break
            except:
                try:
                    df = pd.read_csv(filepath, sep=sep, encoding='utf-8')
                    if df.shape[1] > 1:
                        break
                except:
                    continue
        
        if df is None or df.shape[1] == 1:
            print(f"ERROR: Could not parse metabolite file")
            sys.exit(1)
        
        # Clean column names - remove BOM, SPARQL ?, and whitespace
        df.columns = df.columns.str.replace('^\ufeff', '', regex=True)  # Remove BOM
        df.columns = df.columns.str.replace('^[?]+', '', regex=True)  # Remove SPARQL ?
        df.columns = df.columns.str.strip()  # Remove whitespace
        
        print(f"Loaded metabolite data: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        # Show sample data
        #print("\nFirst few rows:")
        #print(df.head(3).to_string())
        return df
    except Exception as e:
        print(f"Error loading metabolite data: {e}")
        sys.exit(1)


def clean_trait_data(df):
    """Clean and prepare trait data for analysis"""
    print(f"Input columns: {list(df.columns)}")
    
    # FIRST: Clean ALL string columns - remove XML Schema datatype annotations AND quotes
    # Use simple string replacement which is more reliable than regex on DataFrames
    for col in df.columns:
            # Clean each value individually to ensure it works
            # Remove XML Schema annotations
            df[col] = df[col].apply(lambda x: str(x).split('^^<http://')[0] if pd.notna(x) and '^^<http://' in str(x) else str(x))
            # Remove surrounding quotes (both single and double)
            df[col] = df[col].str.strip().str.strip('"').str.strip("'").str.strip()
    
    print("Cleaned XML Schema annotations and quotes from all string columns")
    
    # Find the trait value column (try multiple possible names)
    value_col = None
    possible_value_cols = ['tryDataVal', 'trait_value', 'value', 'traitValue', 'DataVal']
    for col in possible_value_cols:
        if col in df.columns:
            value_col = col
            break
    
    if value_col is None:
        print(f"ERROR: Could not find trait value column!")
        print(f"Looking for one of: {possible_value_cols}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Convert trait values to numeric
    df['trait_value'] = pd.to_numeric(df[value_col], errors='coerce')
    print(f"Using '{value_col}' as trait value column")
    
    # Find trait name column
    trait_col = None
    possible_trait_cols = ['traitLabel', 'traitCategory', 'trait', 'traitName']
    for col in possible_trait_cols:
        if col in df.columns:
            trait_col = col
            break
    
    if trait_col is None:
        print(f"ERROR: Could not find trait name column!")
        print(f"Looking for one of: {possible_trait_cols}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    print(f"Using '{trait_col}' as trait name column")
    
    # Show cleaned trait names
    unique_traits = df[trait_col].unique()
    print(f"Unique trait names found (after cleaning): {unique_traits[:5]}")
    if len(unique_traits) > 5:
        print(f"  ... and {len(unique_traits) - 5} more")
    
    # Create standardized trait categories - try multiple mapping strategies
    trait_mapping = {
        'Specific leaf area': 'SLA',
        'Leaf dry matter content': 'LDMC',
        'Leaf nitrogen content per dry mass (Nmass)': 'Leaf_N',
        'Leaf phosphorus content per dry mass (Pmass)': 'Leaf_P',
        'Leaf nitrogen content': 'Leaf_N',
        'Leaf phosphorus content': 'Leaf_P',
        # SLA variations
        'SLA: undefined if petiole in- or excluded': 'SLA',
        'SLA: petiole excluded': 'SLA',
        'SLA: petiole included': 'SLA',
        'SLA: undefined if petiole in- or excluded (1)': 'SLA',
        'SLA: petiole  excluded': 'SLA',  # Note: double space
        'SLA: petiole  included': 'SLA',  # Note: double space
        'SLA': 'SLA',
        'LDMC': 'LDMC',
        'Leaf N': 'Leaf_N',
        'Leaf P': 'Leaf_P',
        'Nmass': 'Leaf_N',
        'Pmass': 'Leaf_P'
    }
    
    # Map traits to standardized names
    df['trait_std'] = df[trait_col].map(trait_mapping)
    
    # Show what was mapped
    mapped = df[df['trait_std'].notna()][trait_col].unique()
    unmapped = df[df['trait_std'].isna()][trait_col].unique()
    
    if len(mapped) > 0:
        print(f"✓ Successfully mapped {len(mapped)} trait types")
        for trait in mapped[:10]:  # Show first 10
            print(f"  - '{trait}' → {df[df[trait_col]==trait]['trait_std'].iloc[0]}")
    
    if len(unmapped) > 0:
        print(f"✗ Could not map {len(unmapped)} trait types:")
        for trait in unmapped[:5]:
            print(f"  - '{trait}'")
        if len(unmapped) > 5:
            print(f"  ... and {len(unmapped) - 5} more")
    
    if len(mapped) == 0:
        print(f"\nERROR: No traits were successfully mapped!")
        print(f"\nDebugging info:")
        print(f"Sample trait names in your data (first 3):")
        for i, trait in enumerate(df[trait_col].unique()[:3]):
            print(f"  {i+1}. '{trait}' (length: {len(trait)}, repr: {repr(trait)})")
        sys.exit(1)
    
    df = df[df['trait_std'].notna()]
    
    # Remove missing values
    before = len(df)
    df = df[df['trait_value'].notna()]
    after = len(df)
    if before - after > 0:
        print(f"Removed {before - after} rows with missing trait values")
    
    print(f"✓ Final dataset: {len(df)} rows with mapped traits")
    
    return df


def create_trait_matrix(df):
    """Create species × trait matrix for LES analysis"""
    # Pivot to get species as rows and traits as columns
    if 'sourceName' in df.columns and 'trait_std' in df.columns:
        trait_matrix = df.pivot_table(
            index='sourceName',
            columns='trait_std',
            values='trait_value',
            aggfunc='mean'
        )
    else:
        print("Warning: Required columns not found for trait matrix")
        return None
    
    # Remove species with missing traits
    trait_matrix = trait_matrix.dropna()
    
    print(f"\nTrait matrix: {trait_matrix.shape[0]} species × {trait_matrix.shape[1]} traits")
    print(f"Traits: {list(trait_matrix.columns)}")
    
    return trait_matrix


def plot_trait_correlations(trait_matrix, output_dir):
    """Plot correlation matrix of LES traits"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate correlations
    corr = trait_matrix.corr()
    
    # Create heatmap
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdBu_r', 
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=1, cbar_kws={"shrink": 0.8})
    
    plt.title('Leaf Economic Spectrum - Trait Correlations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/les_trait_correlations.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/les_trait_correlations.png")
    
    # Print correlation statistics
    #print("\n=== Trait Correlations ===")
    print(corr.to_string())
    
    return corr


def plot_trait_pairwise(trait_matrix, output_dir):
    """Create pairwise scatterplots for LES traits"""
    # Create pairplot
    fig = sns.pairplot(trait_matrix, diag_kind='kde', plot_kws={'alpha': 0.6})
    fig.fig.suptitle('Leaf Economic Spectrum - Pairwise Trait Relationships', 
                     y=1.01, fontsize=14, fontweight='bold')
    
    plt.savefig(f'{output_dir}/les_trait_pairwise.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/les_trait_pairwise.png")
    plt.close()


def perform_pca(trait_matrix, output_dir):
    """Perform PCA on LES traits to identify main axes of variation"""
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(trait_matrix)
    
    # Perform PCA
    pca = PCA()
    pca_scores = pca.fit_transform(scaled_data)
    
    # Create DataFrame with PCA scores
    pca_df = pd.DataFrame(
        pca_scores[:, :2],
        index=trait_matrix.index,
        columns=['PC1', 'PC2']
    )
    
    # Plot variance explained
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scree plot
    variance_explained = pca.explained_variance_ratio_ * 100
    ax1.bar(range(1, len(variance_explained) + 1), variance_explained, 
            color='steelblue', alpha=0.8)
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Variance Explained (%)', fontsize=12)
    ax1.set_title('Scree Plot - LES Trait Variation', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(1, len(variance_explained) + 1))
    
    # Add cumulative variance line
    ax1_twin = ax1.twinx()
    cumvar = np.cumsum(variance_explained)
    ax1_twin.plot(range(1, len(variance_explained) + 1), cumvar, 
                  'ro-', linewidth=2, markersize=8)
    ax1_twin.set_ylabel('Cumulative Variance (%)', fontsize=12)
    ax1_twin.grid(False)
    
    # Biplot (PCA scores + loadings)
    ax2.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.6, s=50, color='steelblue')
    
    # Plot loadings as arrows
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    for i, trait in enumerate(trait_matrix.columns):
        ax2.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3, 
                 head_width=0.1, head_length=0.1, fc='red', ec='red', 
                 linewidth=2, alpha=0.7)
        ax2.text(loadings[i, 0]*3.2, loadings[i, 1]*3.2, trait, 
                fontsize=11, fontweight='bold', color='darkred')
    
    ax2.set_xlabel(f'PC1 ({variance_explained[0]:.1f}%)', fontsize=12)
    ax2.set_ylabel(f'PC2 ({variance_explained[1]:.1f}%)', fontsize=12)
    ax2.set_title('PCA Biplot - LES Trait Space', fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/les_pca_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/les_pca_analysis.png")
    
    # Print PCA statistics
    print("\n=== PCA Results ===")
    print(f"PC1 explains {variance_explained[0]:.2f}% of variance")
    print(f"PC2 explains {variance_explained[1]:.2f}% of variance")
    print(f"Total variance explained by PC1+PC2: {cumvar[1]:.2f}%")
    print("\nPrincipal Component Loadings:")
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(pca.components_))],
        index=trait_matrix.columns
    )
    print(loadings_df.to_string())
    
    return pca_df, pca, loadings_df

def classify_les_strategy(trait_matrix):
    """Classify species along the LES continuum"""
    # Use PCA to get LES axis
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(trait_matrix)
    
    pca = PCA(n_components=1)
    les_scores = pca.fit_transform(scaled_data)
    
    # Create DataFrame with LES scores
    les_df = pd.DataFrame({
        'species': trait_matrix.index,
        'LES_score': les_scores[:, 0]
    })
    
    # Orient PC1 so that positive scores = acquisitive (high SLA direction).
    # PCA sign is arbitrary, so we anchor it: if SLA loads negatively on PC1, flip all scores.
    sla_loading = pca.components_[0][list(trait_matrix.columns).index('SLA')]
    if sla_loading < 0:
        les_df['LES_score'] = -les_df['LES_score']
        print("  (PC1 flipped to align positive direction with SLA)")

    # Classify at the origin (zero), with a ±1 SD band around it for Intermediate.
    # This respects the natural centroid of the trait space rather than forcing equal groups.
    sd = les_df['LES_score'].std()
    les_df['strategy'] = pd.cut(
        les_df['LES_score'],
        bins=[-np.inf, -sd, sd, np.inf],
        labels=['Conservative', 'Intermediate', 'Acquisitive']
    )

    print("\n=== LES Strategy Classification ===")
    print(f"  Threshold band: ±{sd:.2f} (1 SD around origin)")
    print(les_df['strategy'].value_counts())
    
    return les_df

'''
def classify_les_strategy(trait_matrix):
    """Classify species along the LES continuum"""
    # Use PCA to get LES axis
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(trait_matrix)
    
    pca = PCA(n_components=1)
    les_scores = pca.fit_transform(scaled_data)
    
    # Create DataFrame with LES scores
    les_df = pd.DataFrame({
        'species': trait_matrix.index,
        'LES_score': les_scores[:, 0]
    })
    
    # Classify into categories based on quantiles
    les_df['strategy'] = pd.cut(
        les_df['LES_score'],
        bins=[-np.inf, les_df['LES_score'].quantile(0.33), 
              les_df['LES_score'].quantile(0.67), np.inf],
        labels=['Conservative', 'Intermediate', 'Acquisitive']
    )
    
    print("\n=== LES Strategy Classification ===")
    print(les_df['strategy'].value_counts())
    
    return les_df

'''

def analyze_metabolites_by_strategy(trait_df, metabolite_df, les_df, species_id_map, output_dir):
    """Link metabolite diversity to LES strategies"""
    
    # Rename 'species' → 'sourceName' in les_df so it can be joined later
    species_les = les_df.rename(columns={'species': 'sourceName'})
    print("\n--- species_les (from LES classification) ---")
    print(species_les.head(3).to_string())
    print(f"  Total species with LES scores: {len(species_les)}")
    
    # Count metabolites per species (keyed on source_wdx)
    if 'source_wdx' in metabolite_df.columns and 'structure_inchikey' in metabolite_df.columns:
        metabolite_counts = metabolite_df.groupby('source_wdx').agg({
            'structure_inchikey': 'nunique',
            'wd_chem': 'nunique'
        }).reset_index()
        metabolite_counts.columns = ['source_wdx', 'n_metabolites', 'n_unique_chem']
    else:
        print("Warning: Required columns not found in metabolite data")
        return
    print(f"\n--- metabolite_counts (unique metabolites per species) ---")
    print(metabolite_counts.head(3).to_string())
    print(f"  Total species with metabolite data: {len(metabolite_counts)}")

    # === KEY FIX: use the pre-filtering species_id_map to attach sourceName ===
    # This map was built from the raw trait data before clean_trait_data dropped rows,
    # so it contains every species that was ever in the trait file.
    metabolite_counts = metabolite_counts.merge(species_id_map, on='source_wdx', how='inner')
    print(f"\n--- after joining species_id_map (source_wdx → sourceName) ---")
    print(metabolite_counts.head(3).to_string())
    print(f"  Species retained after ID→name join: {len(metabolite_counts)}")

    # Merge with LES classification (only species that made it through PCA survive here)
    metabolite_les = metabolite_counts.merge(species_les, on='sourceName', how='inner')
    print(f"\n--- after joining species_les (attaching LES scores) ---")
    print(metabolite_les.head(3).to_string())
    print(f"  Species retained after LES join: {len(metabolite_les)}")

    # Diagnostic summary
    print("\n=== SPECIES OVERLAP DIAGNOSTIC ===")
    print(f"  Raw trait file species (ID map):       {len(species_id_map)}")
    print(f"  Species with metabolite data:          {len(metabolite_counts)}")
    print(f"  Species with LES scores:               {len(species_les)}")
    print(f"  Final overlap (metabolites ∩ LES):     {len(metabolite_les)}")
    if len(metabolite_les) == 0:
        # Show which sourceName values exist on each side so you can spot mismatches
        met_names = set(metabolite_counts['sourceName'].unique())
        les_names = set(species_les['sourceName'].unique())
        print(f"\n  WARNING: zero overlap!")
        print(f"  Example metabolite sourceName values: {list(met_names)[:5]}")
        print(f"  Example LES      sourceName values:   {list(les_names)[:5]}")
        print(f"  Shared names: {met_names & les_names}")
        return

    # Plot metabolite diversity by LES strategy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Boxplot
    sns.boxplot(data=metabolite_les, x='strategy', y='n_metabolites', 
               palette='Set2', ax=ax1)
    ax1.set_xlabel('LES Strategy', fontsize=12)
    ax1.set_ylabel('Number of Metabolites', fontsize=12)
    ax1.set_title('Metabolite Diversity by LES Strategy', fontsize=12, fontweight='bold')
    
    # Scatter plot with LES score
    sns.scatterplot(data=metabolite_les, x='LES_score', y='n_metabolites',
                   hue='strategy', palette='Set2', s=100, alpha=0.7, ax=ax2)
    ax2.set_xlabel('LES Score (Conservative ← → Acquisitive)', fontsize=12)
    ax2.set_ylabel('Number of Metabolites', fontsize=12)
    ax2.set_title('Metabolite Diversity along LES Continuum', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metabolite_les_relationship.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/metabolite_les_relationship.png")
    
    # Statistical test
    print("\n=== Metabolite Diversity by LES Strategy ===")
    for strategy in metabolite_les['strategy'].unique():
        subset = metabolite_les[metabolite_les['strategy'] == strategy]
        print(f"{strategy}: {subset['n_metabolites'].mean():.1f} ± {subset['n_metabolites'].std():.1f} metabolites (n={len(subset)})")
    
    # Correlation test
    if len(metabolite_les) > 3:
        corr, pval = stats.spearmanr(metabolite_les['LES_score'], 
                                     metabolite_les['n_metabolites'])
        print(f"\nSpearman correlation (LES score vs metabolites): r={corr:.3f}, p={pval:.4f}")


def plot_trait_distributions(trait_df, output_dir):
    """Plot distributions of individual LES traits"""
    if 'trait_std' not in trait_df.columns or 'trait_value' not in trait_df.columns:
        print("Warning: Cannot plot trait distributions - missing columns")
        return
    
    traits = trait_df['trait_std'].unique()
    n_traits = len(traits)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, trait in enumerate(traits[:4]):  # Limit to 4 traits
        data = trait_df[trait_df['trait_std'] == trait]['trait_value']
        
        axes[i].hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        axes[i].axvline(data.median(), color='red', linestyle='--', 
                       linewidth=2, label=f'Median: {data.median():.2f}')
        axes[i].set_xlabel('Trait Value', fontsize=11)
        axes[i].set_ylabel('Frequency', fontsize=11)
        axes[i].set_title(f'{trait} Distribution (n={len(data)})', 
                         fontsize=11, fontweight='bold')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/les_trait_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/les_trait_distributions.png")


def main():
    # Get input from command-line arguments
    if len(sys.argv) < 3:
        print("Usage: python CS5_viz.py <traits_file> <metabolites_file> [output_dir]")
        print("\nExample:")
        print("  python CS5_viz.py traits.tsv metabolites.tsv")
        print("  python CS5_viz.py traits.tsv metabolites.tsv results/")
        sys.exit(1)
    
    traits_file = sys.argv[1]
    metabolites_file = sys.argv[2]
    output_dir_name = sys.argv[3] if len(sys.argv) > 3 else 'les_output'
    
    # Create output directory
    output_dir = Path(output_dir_name)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nOutput directory: {output_dir.absolute()}\n")
    
    # Load data
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    trait_df = load_trait_data(traits_file)
    metabolite_df = load_metabolite_data(metabolites_file)
    
    print("\n" + "=" * 60)
    print("CLEANING METABOLITE DATA")
    print("=" * 60)

    for col in metabolite_df.columns:
            # Clean each value individually to ensure it works
            # Remove XML Schema annotations
            metabolite_df[col] = metabolite_df[col].apply(lambda x: str(x).split('^^<http://')[0] if pd.notna(x) and '^^<http://' in str(x) else str(x))
            # Remove surrounding quotes (both single and double)
            metabolite_df[col] = metabolite_df[col].str.strip().str.strip('"').str.strip("'").str.strip()
    #print("\nFirst few rows of metabolite data:")
    #print(metabolite_df.head(3).to_string())



    # Clean and prepare data
    #print("\n" + "=" * 60)
    print("CLEANING TRAIT DATA")
    #print("=" * 60)
    trait_df_clean = clean_trait_data(trait_df)
    #print("\nFirst few rows:")
    #print(trait_df_clean.head(3).to_string())
    
    # Build the species ID → name map BEFORE any filtering, so no species are lost
    species_id_map = (
        trait_df_clean[['source_wdx', 'sourceName']]
        .drop_duplicates()
        .dropna(subset=['source_wdx', 'sourceName'])
    )
    print(f"\nSpecies ID map built from raw trait data: {len(species_id_map)} unique species")
    
    # Create trait matrix
    #print("\n" + "=" * 60)
    print("CREATING TRAIT MATRIX")
    #print("=" * 60)
    trait_matrix = create_trait_matrix(trait_df_clean)
    #print("\nFirst few rows of trait matrix:")
    #print(trait_matrix.head(3).to_string())
    
    if trait_matrix is None or len(trait_matrix) == 0:
        print("Error: Could not create trait matrix. Check your data format.")
        sys.exit(1)
    
    # Plot trait distributions
    #print("\n" + "=" * 60)
    print("PLOTTING TRAIT DISTRIBUTIONS")
    #print("=" * 60)
    plot_trait_distributions(trait_df_clean, output_dir)
    
    # Analyze trait correlations
    #print("\n" + "=" * 60)
    print("ANALYZING TRAIT CORRELATIONS")
    #print("=" * 60)
    corr_matrix = plot_trait_correlations(trait_matrix, output_dir)
    
    # Create pairwise plots
    #print("\n" + "=" * 60)
    print("CREATING PAIRWISE PLOTS")
    #print("=" * 60)
    plot_trait_pairwise(trait_matrix, output_dir)
    
    # Perform PCA
    #print("\n" + "=" * 60)
    print("PERFORMING PCA")
    #print("=" * 60)
    pca_df, pca, loadings = perform_pca(trait_matrix, output_dir)
    
    # Classify LES strategies
    #print("\n" + "=" * 60)
    print("CLASSIFYING LES STRATEGIES")
    #print("=" * 60)
    les_df = classify_les_strategy(trait_matrix)
    
    # Analyze metabolites by strategy
    #print("\n" + "=" * 60)
    print("ANALYZING METABOLITE-LES RELATIONSHIPS")
    #print("=" * 60)
    analyze_metabolites_by_strategy(trait_df_clean, metabolite_df, les_df, species_id_map, output_dir)
    
    # Save processed data
    trait_matrix.to_csv(f'{output_dir}/trait_matrix.csv')
    les_df.to_csv(f'{output_dir}/les_classification.csv', index=False)
    
    #print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    #print("=" * 60)
    print(f"\nAll results saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print(f"  - les_trait_correlations.png (trait correlation heatmap)")
    print(f"  - les_trait_pairwise.png (pairwise scatterplots)")
    print(f"  - les_pca_analysis.png (PCA biplot and variance)")
    print(f"  - les_trait_distributions.png (trait histograms)")
    print(f"  - metabolite_les_relationship.png (metabolite diversity)")
    print(f"  - trait_matrix.csv (species × trait matrix)")
    print(f"  - les_classification.csv (LES strategy classifications)")


if __name__ == '__main__':
    main()
