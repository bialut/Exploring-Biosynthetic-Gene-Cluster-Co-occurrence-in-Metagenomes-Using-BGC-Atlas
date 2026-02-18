"""
MASTER PIPELINE: NETWORK INFERENCE & ANALYSIS

Description: 
    This script performs an end-to-end analysis of GCF (Gene Cluster Family) 
    co-occurrence and co-abundance data. It executes three distinct network 
    inference methods to capture different biological dimensions:
    
Methods:
    - Jaccard Index: Captures binary co-occurrence (Niche sharing).
    - Graphical LASSO: Infers conditional independence (Causal backbone).
    - Spearman: Measures monotonic co-abundance (Functional modules).
   
"""
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import community as community_louvain
from itertools import combinations
from collections import Counter
from sklearn.covariance import GraphicalLasso
import os
import warnings

# CONFIGURATION
FILE_METALOG_GCF = "../data/metalog_bgcs_with_gcf_and_tax.tsv"
FILE_MGNIFY_GCF = "../data/mgnify_bgcs_with_gcf_and_tax.tsv"
FILE_METALOG_SAMPLES = "../data/metalog_samples.tsv"
FILE_MGNIFY_SAMPLES = "../data/mgnify_samples.tsv"
OUTPUT_DIR = "../results_thesis"
GLOBAL_COLOR_MAPS = {}

# Jaccard Parameters
MIN_COOCCURRENCE = 20     
JACCARD_THRESHOLD = 0.30  

# GLASSO & Spearman Parameters
GLASSO_ALPHA = 0.002
CLR_PSEUDOCOUNT = 10
PREVALENCE_THRESHOLD = 0.01

SPEARMAN_THRESHOLD = 0.45

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
warnings.simplefilter(action='ignore')

print(" STARTING PIPELINE")


# HELPER FUNCTIONS

def print_network_summary(G, partition, name="Network"):
    """
    Prints professional statistics for the thesis log.
    Includes Density, Node/Edge counts, and Top 15 Community sizes.
    """
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    density = nx.density(G)
    
    # Count community members
    counts = Counter(partition.values())
    
    print("\n" + "="*45)
    print(f" FINAL SUMMARY: {name.upper()}")
    print("="*45)
    print(f" -> Total Nodes (GCFs): {nodes}")
    print(f" -> Total Edges:        {edges}")
    print(f" -> Network Density:    {density:.5f}")
    print(f" -> Total Communities:  {len(counts)}")

def get_smart_biome(row):
    """
Optimized version: Truncates ENVO tags BEFORE validation to ensure bad_terms are reliably detected.
    """
    bad_terms = {'nan', 'none', '', 'mixed', 'other', 'misc', 'unclassified', 'root:mixed', 'generic', '-', 'null'}

    def is_truly_valid(val):
        if val is None or pd.isna(val):
            return False
        
        # 1. Immediately truncate ENVO tags "Lake [ENVO:123]" -> "Lake"
        s = str(val).split('[')[0]
        # 2. Normalize to lowercase and strip whitespace for comparison
        s = s.strip().lower()
        
        # 3. Check against blacklist
        if not s or s in bad_terms:
            return False
        return True

    def format_output(val):
        # Maintain formatting: Ensures clean names like "Marine" or "Soil"
        s = str(val).split('[')[0].strip()
        return s.title()

    # Priority Chain
    # 1. Feature
    val_feat = row.get('environment_feature')
    if is_truly_valid(val_feat):
        return format_output(val_feat), 'environment_feature'

    # 2. Environment Biome
    val_env = row.get('environment_biome')
    if is_truly_valid(val_env):
        return format_output(val_env), 'environment_biome'

    # 3. Biome Lineage
    val_bio = row.get('biome')
    if is_truly_valid(val_bio):
        raw_s = str(val_bio)
        if ':' in raw_s:
            candidate = raw_s.split(':')[-1].strip()
            if is_truly_valid(candidate):
                return format_output(candidate), 'biome_lineage'
        return format_output(val_bio), 'biome_lineage'

    return "Unknown", "unknown"

def rank_communities_by_size(G, partition, network_name="Network"):
    counts = Counter(partition.values())
    # Sort by size (descending), then by original ID (ascending) for stable results
    sorted_ids = sorted(counts.keys(), key=lambda x: (-counts[x], x))
    
    new_names_map = {old_id: f"Community_{rank}" for rank, old_id in enumerate(sorted_ids, 1)}
    community_dict = {node: new_names_map[old_id] for node, old_id in partition.items()}
    
    nx.set_node_attributes(G, community_dict, 'community')

    # 6. Print Summary for the Thesis Log
    print(f"\n[{network_name}] Top 15 Communities by size:")
    size_series = pd.Series(community_dict).value_counts()

    for i in range(1, 16):
        name = f"Community_{i}"
        if name in size_series:
            print(f"  {name}: {size_series[name]} members")


    return community_dict


def clr_transform(data_matrix, pseudocount=10):
    """
    Performs Centered Log-Ratio (CLR) transformation on abundance data.
    
    Args:
        data_matrix: Array-like (NumPy or Pandas) of shape (n_samples, n_features)
        pseudocount: Value added to handle zeros and stabilize log-transform
    """
    if hasattr(data_matrix, 'values'):
        data = data_matrix.values

    else:
        data = data_matrix

    data_plus_pseudo = data + pseudocount
    log_data = np.log(data_plus_pseudo)

    # Calculate mean and reshape safely
    gmean_log = np.mean(log_data, axis=1).reshape(-1, 1)
    clr_data = log_data - gmean_log

    return clr_data


def analyze_community_composition(G, attribute_name, plot_title, legend_title=None, top_n_categories=20, max_communities=15):
    global GLOBAL_COLOR_MAPS
    data = G.nodes(data=True)
    
    # 1. Data extraction and cleaning
    df = pd.DataFrame([d for n, d in data])
        
    # Standardize NaNs and null values
    df[attribute_name] = df[attribute_name].fillna("Unknown").replace(['0', 0, 'nan', 'None', 'unknown'], "Unknown")

    # 2. Helper for ranking (Extracts X from "Community_X")
    def get_rank(name):
        try:
            return int(str(name).split('_')[1])
        except:
            return 9999

    df_filtered = df[df['community'].apply(get_rank) <= max_communities].copy()
    
    # Category filter (Top N most frequent + Unknown/Other)
    global_counts = df_filtered[attribute_name].value_counts()
    categories_to_keep = global_counts.nlargest(top_n_categories).index.tolist()

    df_filtered[attribute_name] = df_filtered[attribute_name].apply(
        lambda x: x if (x in categories_to_keep or x == "Unknown") else "other"
    )

    # 3. Crosstab & Normalization (Purity)
    contingency = pd.crosstab(df_filtered['community'], df_filtered[attribute_name])
    
    # Stable sorting of rows (1, 2, 3... instead of 1, 10, 11...)
    sorted_rows = sorted(contingency.index, key=get_rank)
    contingency = contingency.reindex(index=sorted_rows)

    # calculate purity
    purity = contingency.div(contingency.sum(axis=1), axis=0).fillna(0) * 100

    # 4. Color managment
    if attribute_name not in GLOBAL_COLOR_MAPS:
        GLOBAL_COLOR_MAPS[attribute_name] = {"Unknown": "#20B2AA", "other": "#000000"}
    
    current_map = GLOBAL_COLOR_MAPS[attribute_name]
    palette = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)

    color_list = []
    for cat in purity.columns:
        if cat not in current_map:
            current_map[cat] = palette[len(current_map) % len(palette)]
        color_list.append(current_map[cat])

    # 5. BAR-CHART PLOTTING
    plt.figure(figsize=(8, 5))
    
    purity.plot(kind='bar', 
                stacked=True, 
                ax=plt.gca(), 
                width=0.6, 
                color=color_list, 
                edgecolor='black', 
                linewidth=0.5)

    plt.title(f"{plot_title} (Top {max_communities})", fontsize=14, fontweight='bold')
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.xlabel("Community ID", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    plt.legend(title=legend_title or attribute_name, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{plot_title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 6. HEATMAP PLOTTING
    plt.figure(figsize=(8, 7))
    sns.heatmap(purity, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={'label': 'Percentage (%)'}, linewidths=.5)
    plt.title(f"{plot_title} Heatmap (Top {max_communities})", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{plot_title.replace(' ', '_')}_Heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_deep_dive_heatmap(G, attribute_name, plot_title, max_communities=15, threshold_percent=20):
    """
    Detailed Heatmap focusing strictly on the Top 15.
    Optimized for stable sorting and focused biological signals.
    """
    data = G.nodes(data=True)
    df = pd.DataFrame([d for n, d in data])
    
    # Helfer für die Sortierung (Community_1 vor Community_10)
    def get_rank(name):
        try:
            return int(str(name).split('_')[1])
        except:
            return 9999    

    # 1. Filter for Top Communities only
    all_communities = df['community'].unique()
    relevant_communities = [c for c in all_communities if get_rank(c) <= max_communities]
    # Stable sort
    sorted_communities = sorted(relevant_communities, key=get_rank)

    # 2. Create Contingency Table
    df_filtered = df[df['community'].isin(sorted_communities)]
    ct = pd.crosstab(df_filtered['community'], df_filtered[attribute_name])
    
    # Ensure all Top-Communities are in the correct order
    ct = ct.reindex(index=sorted_communities).fillna(0)

    # 3. Calculate Purity
    purity = ct.div(ct.sum(axis=1), axis=0) * 100
    
    # 4. Filter: Focus on biological significance
    cols_to_drop = [c for c in purity.columns if any(x in str(c).lower() for x in ['other', 'unknown'])]
    purity = purity.drop(columns=cols_to_drop)
    
    # Remove non-informative columns... Keep only columns exceeding the threshold (e.g., 20%)
    important_cols = purity.columns[(purity >= threshold_percent).any()]
    df_heatmap = purity[important_cols]

    if not df_heatmap.empty:
        sorted_cols = df_heatmap.max().sort_values(ascending=False).index
        df_heatmap = df_heatmap[sorted_cols]

        # 5. Plotting
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_heatmap, 
                    annot=True, 
                    fmt=".0f", 
                    cmap="YlGnBu", 
                    linewidths=.5,
                    vmin=0, vmax=100,
                    cbar_kws={'label': 'Purity (%)'})
        
        plt.title(f"{plot_title}\n(Top {max_communities}, Threshold >{threshold_percent}%)", 
                  fontsize=14, fontweight='bold')
        plt.ylabel("Community ID")
        plt.xlabel(attribute_name.replace('_id', '').replace('_', ' ').title())
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{plot_title.replace(' ', '_')}_DeepDive.png", dpi=300)
        plt.close()

def map_all_metadata(G, df_raw):
    print("Mapping attributes to Graph (Optimized)...")
    
    # 1. Column Name Check
    
    if 'bigslice_gcf_id' in df_raw.columns:
        id_col = 'bigslice_gcf_id'
    elif 'gcf_id' in df_raw.columns:
        id_col = 'gcf_id'
    else:
        print(" ERROR: Neither 'bigslice_gcf_id' nor 'gcf_id' found in DataFrame.")
        return
    
    node_list = list(G.nodes())
    
    # Create a subset copy once to calculate new columns (Biome/Tax)
    # This filters the huge dataframe down to only the GCFs in the graph.
    df_sub = df_raw[df_raw[id_col].isin(node_list)].copy()
    
    # ---------------------------------------------------------
    # PART B: BIOME MAPPING (Using Pre-calculated Smart Biome)
    # ---------------------------------------------------------

    
    if 'dominant_biome' in df_sub.columns:
        # If step 2 returned "Unknown", map to None or "Unclassified" for the graph
        
        # Success report for this specific subgraph
        valid_count = (df_sub['dominant_biome'] != "Unknown").sum()
        total_mapped = len(df_sub)
        success_rate = (valid_count / total_mapped) * 100 if total_mapped > 0 else 0
        
        print(f"[REPORT] Biome mapping in subgraph: {success_rate:.1f}% ({valid_count}/{total_mapped})")

# ---------------------------------------------------------
    # PART C: MAPPING ATTRIBUTES TO NODES
    # ---------------------------------------------------------
    attributes_to_add = [
        {'col': 'product_category', 'attr': 'product_categories'}, 
        {'col': 'product_type',     'attr': 'product_subcategories'},
        {'col': 'tax_family',       'attr': 'family_id'},
        {'col': 'tax_genus',        'attr': 'genus_id'},
        {'col': 'tax_species',      'attr': 'species_id'},
        {'col': 'dominant_biome',   'attr': 'dominant_biome'} 
    ]
    

    for item in attributes_to_add:
        col_name = item['col']
        attr_name = item['attr']
        
        if col_name in df_sub.columns:
            # 1. Prepare data (immediately mark '0' or NaNs as Unknown)
            df_clean = df_sub[[id_col, col_name]].copy()
            df_clean[col_name] = df_clean[col_name].replace(['0', 0, '-1', -1, 'nan', 'None'], np.nan).fillna("Unknown")
            df_clean[col_name] = df_clean[col_name].astype(str).str.replace(r"[\[\]'()\" {}]", "", regex=True).str.strip()
            
            if not df_clean.empty:
                # UNIVERSAL PRIORITY LOGIC
                def get_most_common_priority(series):
                    # Filters all variants of "No Info"
                    valid = series[~series.isin(["Unknown", "unknown", "0", 0, "-1", -1])]
                    if not valid.empty:
                        # If real names exist, the mode wins
                        return valid.mode().iloc[0]
                    return "Unknown"
                
                mapping_dict = df_clean.groupby(id_col)[col_name].agg(get_most_common_priority).to_dict()
                
                # Convert keys to strings for NetworkX and write to graph
                mapping_dict = {str(k): v for k, v in mapping_dict.items()}
                nx.set_node_attributes(G, mapping_dict, attr_name)
                
                print(f"  -> Added '{attr_name}' (Priority Logic) to {len(mapping_dict)} nodes.")
def get_source_efficiency_report(df):

    """Calculates which metadata column provided the final biome info."""
    bad_terms = {'nan', 'none', '', 'mixed', 'other', 'misc', 'unclassified', 'root:mixed', 'generic'}
    stats = {'environment_feature': 0, 'environment_biome': 0, 'biome': 0, 'unknown': 0}
    
    def is_valid(val):
        s = str(val).split('[')[0].strip().lower()
        return s and s not in bad_terms

    for _, row in df.iterrows():
        if is_valid(row.get('environment_feature', '')): stats['environment_feature'] += 1
        elif is_valid(row.get('environment_biome', '')): stats['environment_biome'] += 1
        else:
            raw_b = str(row.get('biome', ''))
            candidate = raw_b.split(':')[-1].strip() if ':' in raw_b else raw_b
            
            if is_valid(candidate):
                stats['biome'] += 1
            else:
                stats['unknown'] += 1
                
    return stats

def validate_community_consistency(G1, G2, name1="Glasso", name2="Spearman"):
    """
    Creates a cross-tabulation heatmap to validate community consistency.
    Maintains the notebook's sorting logic based on the size of shared GCFs.
    """
    # 1. Identify shared nodes
    shared_nodes = set(G1.nodes()) & set(G2.nodes())
    if not shared_nodes:
        print(f"[VALIDATION] No shared nodes between {name1} and {name2}.")
        return

    # 2. Build comparison DataFrame
    df_compare = pd.DataFrame(index=list(shared_nodes))
    df_compare[f'{name1.lower()}_comm'] = pd.Series(nx.get_node_attributes(G1, 'community'))
    df_compare[f'{name2.lower()}_comm'] = pd.Series(nx.get_node_attributes(G2, 'community'))
    df_compare = df_compare.dropna()

    # 3. Create Crosstab
    ct = pd.crosstab(df_compare[f'{name1.lower()}_comm'], df_compare[f'{name2.lower()}_comm'])

    # 4. Sorting logic from notebook: Sort by total frequency in the shared set (Descending)
    top_g1_ids = df_compare[f'{name1.lower()}_comm'].value_counts().nlargest(10).index
    top_g2_ids = df_compare[f'{name2.lower()}_comm'].value_counts().nlargest(10).index

    # Reindex matrix to match the descending size order
    comparison_matrix = ct.loc[top_g1_ids, top_g2_ids]

    # 5. Clean labels for the plot (e.g., "G-C5" instead of "Community_5")
    comparison_matrix.index = [f"{name1[0]}-C{str(i).split('_')[1]}" for i in comparison_matrix.index]
    comparison_matrix.columns = [f"{name2[0]}-C{str(i).split('_')[1]}" for i in comparison_matrix.columns]

    # 6. Plotting
    plt.figure(figsize=(12, 9))
    sns.heatmap(comparison_matrix, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5,
                cbar_kws={'label': 'Number of shared GCFs'})

    plt.title(f'Validation: Community Consistency\n{name1} (G) vs. {name2} (S)', fontsize=16, pad=20)
    plt.xlabel(f'{name2} Communities (Sorted by size)', fontweight='bold')
    plt.ylabel(f'{name1} Communities (Sorted by size)', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Community_Consistency_Heatmap_{name1[0]}{name2[0]}.png", dpi=300)
    plt.close()
    
# MAIN LOGIC
def main():
    # 1. DATA LOADING
    print("\n[Step 1] Loading Primary GCF Tables and External Metadata...")
    cols_gcf = ['analysis_accession', 'gcf_id', 'mmseqs_taxonomy', 'product_category', 'product_type']
    df_gcf = pd.concat([
        pd.read_csv(FILE_METALOG_GCF, sep=r'\t', usecols=lambda x: x in cols_gcf),
        pd.read_csv(FILE_MGNIFY_GCF, sep=r'\t', usecols=lambda x: x in cols_gcf)
    ], ignore_index=True)

    cols_meta = ['external_id', 'environment_feature', 'environment_biome', 'biome']
    df_meta = pd.concat([
        pd.read_csv(FILE_METALOG_SAMPLES, sep='\t', usecols=lambda x: x in cols_meta),
        pd.read_csv(FILE_MGNIFY_SAMPLES, sep='\t', usecols=lambda x: x in cols_meta)
    ], ignore_index=True)

    # 2. CLEANING & NOISE FILTER
    print("[Step 2] Cleaning GCF data ...")
    df_clean = df_gcf.dropna(subset=['gcf_id']).copy()
    df_clean['gcf_id'] = df_clean['gcf_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    
    # COMPREHENSIVE noise filter (includes "" and "unknown")
    noise_list = ["-1", "nan", "None", "", "unknown"]
    df_clean = df_clean[~df_clean["gcf_id"].isin(noise_list)]
    df_clean.rename(columns={'gcf_id': 'bigslice_gcf_id'}, inplace=True)
    
    
    # TAXONOMY
    print(" -> Extracting taxonomy levels (Family, Genus, Species)...")
    if 'mmseqs_taxonomy' in df_clean.columns:
        df_clean['tax_family']  = df_clean['mmseqs_taxonomy'].str.extract(r'f_([^;]+)')
        df_clean['tax_genus']   = df_clean['mmseqs_taxonomy'].str.extract(r'g_([^;]+)')
        df_clean['tax_species'] = df_clean['mmseqs_taxonomy'].str.extract(r's_([^;]+)')

    # 3. PREPARING METADATA MAPPING
    print("[Step 3] Preparing Metadata for mapping (Smart Biome)...")
    biome_stats = df_meta.apply(get_smart_biome, axis=1)
    df_meta['dominant_biome'] = [r[0] for r in biome_stats]
    
    # Unique metadata per sample to prevent row multiplication during merge
    df_meta_unique = df_meta.drop_duplicates(subset=['external_id'])
    df_mapping = pd.merge(df_clean, df_meta_unique, left_on='analysis_accession', right_on='external_id', how='left')
    
    report = get_source_efficiency_report(df_meta)
    print("\n[REPORT] Metadata Source Efficiency (Decision Chain):")
    for source, count in report.items():
        percentage = (count / len(df_meta)) * 100
        print(f"  -> {source}: {count} samples ({percentage:.1f}%)")
        
    # PART A: JACCARD

    print("\n--- PART A: JACCARD PIPELINE ---")

    gcf_freq_dict = df_clean.groupby("bigslice_gcf_id")["analysis_accession"].nunique().to_dict()
    print(f"Individual GCF frequencies calculated.")

    print("Counting pairs (Itertools)...")
    gcfs_per_sample = df_clean.groupby("analysis_accession")["bigslice_gcf_id"].apply(set)
    pair_counts = Counter()

    for gcfs in gcfs_per_sample:
        for g1, g2 in combinations(sorted(list(gcfs)), 2):
            pair_counts[(g1, g2)] += 1

    print(f"Finished! Found {len(pair_counts)} unique pairs.")

    print(f"Filtering pairs with count >= {MIN_COOCCURRENCE}...")
    filtered_data = []
    for (g1, g2), count in pair_counts.items():
        if count >= MIN_COOCCURRENCE:
            f1 = gcf_freq_dict.get(g1, 0)
            f2 = gcf_freq_dict.get(g2, 0)
            denom = (f1 + f2 - count)
            if denom == 0: denom = 1
            jaccard = count / denom
            jaccard = round(jaccard, 8)
            
            if jaccard > JACCARD_THRESHOLD:
                filtered_data.append({
                    'gcf1': g1, 'gcf2': g2, 
                    'count': count, 'jaccard_index': jaccard,
                    'distance': 1.0 - jaccard, 'weight': jaccard
                })
    
    df_coocc = pd.DataFrame(filtered_data)
    df_coocc = df_coocc.sort_values(by=['jaccard_index', 'gcf1', 'gcf2'], ascending=[False, True, True])
    df_coocc = df_coocc.sort_values(by=["jaccard_index", "gcf1", "gcf2"], ascending=[False, True, True]).reset_index(drop=True)
    df_coocc.to_csv(f"{OUTPUT_DIR}/coocc_gcfs_with_jaccard.csv", index=False)

    print(f"\nBuilding Jaccard Graph...")
    G_jaccard = nx.from_pandas_edgelist(
        df_coocc, source="gcf1", target="gcf2", edge_attr=['weight', 'distance'] 
    )
    
    if len(G_jaccard.nodes) > 0:
        print("Calculating Metrics (Betweenness, Harmonic, Neighborhood)...")
        bet = nx.betweenness_centrality(G_jaccard, normalized=True, weight="distance")
        nx.set_node_attributes(G_jaccard, bet, 'betweenness_score')
        
        harm_raw = nx.harmonic_centrality(G_jaccard, distance="distance")
        n = len(G_jaccard)
        nx.set_node_attributes(G_jaccard, {k: v/(n-1) for k,v in harm_raw.items()}, 'harmonic_score')
        
        neigh_raw = nx.average_neighbor_degree(G_jaccard, weight="weight")
        nx.set_node_attributes(G_jaccard, neigh_raw, 'neighborhood_connectivity')

        #  Louvain Community Detection; uses 'weight' and fixed random_state
        part_jaccard = community_louvain.best_partition(G_jaccard, weight='weight', random_state=42, resolution=2)
        rank_communities_by_size(G_jaccard, part_jaccard, "Jaccard")
        mod_jaccard = community_louvain.modularity(part_jaccard, G_jaccard, weight='weight')
        print(f"[METRIC] Jaccard Network Modularity (Q): {mod_jaccard:.4f}")
        
        map_all_metadata(G_jaccard, df_mapping)
        nx.write_graphml(G_jaccard, f"{OUTPUT_DIR}/network_jaccard.graphml")

        print("Generating Jaccard Plots...")
        analyze_community_composition(G_jaccard, 'dominant_biome', 'Jaccard Ecological Composition (Biome)')
        analyze_community_composition(G_jaccard, 'product_categories', 'Jaccard Product', 'Product')
        analyze_community_composition(G_jaccard, 'product_subcategories', 'Jaccard Sub Product', 'Sub Type')
        analyze_community_composition(G_jaccard, 'genus_id', 'Jaccard Taxonomic Composition (Genus Level)')
        analyze_community_composition(G_jaccard, 'family_id', 'Jaccard Taxonomic Composition (Family Level)')
        analyze_community_composition(G_jaccard, 'species_id', 'Jaccard Taxonomic Composition (Species Level)')
        plot_deep_dive_heatmap(G_jaccard, 'dominant_biome', 'Jaccard Ecological Composition (Biome)', threshold_percent=20)
        plot_deep_dive_heatmap(G_jaccard, 'product_categories', 'Jaccard Deep Dive', threshold_percent=10)
        plot_deep_dive_heatmap(G_jaccard, 'product_subcategories', 'Jaccard SubCat Deep Dive', threshold_percent=10)
        plot_deep_dive_heatmap(G_jaccard, 'genus_id', 'Jaccard Genus Deep Dive', threshold_percent=10)
        plot_deep_dive_heatmap(G_jaccard, 'family_id', 'Jaccard Family Deep Dive', threshold_percent=10)
        plot_deep_dive_heatmap(G_jaccard, 'species_id', 'Jaccard Species Deep Dive', threshold_percent=5)



    # PART B: GLASSO

    # 4. Calculate Prevalence
    print("\n--- PART B: GLASSO PIPELINE ---")
    gcf_prevalence = df_clean.groupby('bigslice_gcf_id')['analysis_accession'].nunique()
    total_samples = df_clean['analysis_accession'].nunique()
    min_samples = int(PREVALENCE_THRESHOLD * total_samples)
    
    passing_gcfs = gcf_prevalence[gcf_prevalence >= min_samples].index
    df_filtered = df_clean[df_clean['bigslice_gcf_id'].isin(passing_gcfs)]
    print(f" -> GCFs passing threshold: {len(passing_gcfs)}")

    # 5. CREATE ABUNDANCE MATRIX, which is done before metadata merge
    df_counts = df_filtered.groupby(['analysis_accession', 'bigslice_gcf_id']).size()
    df_abundanz = df_counts.unstack(fill_value=0)
    df_abundanz = df_abundanz.reindex(sorted(df_abundanz.columns), axis=1)
    df_abundanz = df_abundanz.sort_index()

    
    # 6. Remove duplicates (Collinearity)
    df_final = df_abundanz.T.drop_duplicates().T
    print(f" -> Final Matrix Shape: {df_final.shape}")

    # GLASSO SPECIFIC
    gcf_names = df_final.columns
    abundance_matrix = df_final.values
    clr_data = clr_transform(abundance_matrix, pseudocount=CLR_PSEUDOCOUNT)

    print(f"Fitting GLASSO model with Alpha={GLASSO_ALPHA}...")
    try:
        model = GraphicalLasso(alpha=GLASSO_ALPHA, mode='cd', tol=1e-4, max_iter=1000)
        model.fit(clr_data)
        precision_matrix = model.precision_
        
        G_glasso = nx.Graph()
        G_glasso.add_nodes_from(gcf_names)
        
        rows, cols = np.where(np.triu(precision_matrix, k=1) != 0)
        for i, j in zip(rows, cols):
            raw_val = precision_matrix[i, j]
            
            # 1. Flip sign (Negative entry = Positive correlation)
            association = -raw_val
            
            # 2. FILTER: Allow only positive associations (cooperation)
            if association > 0:
                strength = association
                
                dist = 1.0 / strength if strength != 0 else 1.0
                
                G_glasso.add_edge(gcf_names[i], gcf_names[j], weight=association, strength=strength, distance=dist)
            
        # Extract Core
        connected_components = list(nx.connected_components(G_glasso))
        connected_components.sort(key=len, reverse=True)
        
        # =============================================================================
        # --- GLASSO: CORE EXTRACTION & STABLE RECONSTRUCTION ---
        # =============================================================================
        connected_components = list(nx.connected_components(G_glasso))
        connected_components.sort(key=len, reverse=True)
        
        if connected_components:
            # 1. Retrieve nodes of the largest component and sort ALPHABETICALLY
            giant_component_nodes = connected_components[0]
            sorted_nodes = sorted(list(giant_component_nodes))

            # 2. Extract edges from subgraph and sort STABLY (Source, Target)
            temp_subgraph = G_glasso.subgraph(sorted_nodes)
            sorted_edges = sorted(temp_subgraph.edges(data=True), key=lambda x: (str(x[0]), str(x[1])))

            # 3. Reconstruct G_core from scratch (important for reproducibility)
            G_core = nx.Graph()
            G_core.add_nodes_from(sorted_nodes)
            G_core.add_edges_from(sorted_edges)

            print(f"Core Graph reconstructed (Stable Sort for 100% Notebook Match).")
            print(f"Nodes: {G_core.number_of_nodes()}, Edges: {G_core.number_of_edges()}")

            # 4. Calculate metrics
            bet = nx.betweenness_centrality(G_core, normalized=True, weight="distance")
            nx.set_node_attributes(G_core, bet, 'betweenness_score')
            
            harm_raw = nx.harmonic_centrality(G_core, distance="distance")
            nx.set_node_attributes(G_core, {k: v/(len(G_core)-1) for k,v in harm_raw.items()}, 'harmonic_score')
        
            neigh_raw = nx.average_neighbor_degree(G_core, weight="weight")
            nx.set_node_attributes(G_core, neigh_raw, 'neighborhood_connectivity')
            
            # 5. Louvain
            part_glasso = community_louvain.best_partition(G_core, weight='weight', random_state=42, resolution=4.0)
            
            # 6. Ranking & Mapping
            rank_communities_by_size(G_core, part_glasso, "GLASSO")
            mod_glasso = community_louvain.modularity(part_glasso, G_core, weight='weight')
            print(f"[METRIC] GLASSO Network Modularity (Q): {mod_glasso:.4f}")
            
            map_all_metadata(G_core, df_mapping)
            nx.write_graphml(G_core, f"{OUTPUT_DIR}/network_glasso.graphml")
            
            # 7. Plots generieren
            analyze_community_composition(G_core, 'dominant_biome', 'Glasso Ecological Composition (Biome)')
            analyze_community_composition(G_core, 'product_categories', 'Glasso Product', 'Product')
            analyze_community_composition(G_core, 'product_subcategories', 'Glasso Sub Product', 'Sub Type')
            analyze_community_composition(G_core, 'genus_id', 'Glasso Taxonomic Composition (Genus Level)')
            analyze_community_composition(G_core, 'family_id', 'Glasso Taxonomic Composition (Family Level)')
            analyze_community_composition(G_core, 'species_id', 'Glasso Taxonomic Composition (Species Level)')
            plot_deep_dive_heatmap(G_core, 'dominant_biome', 'Glasso Ecological Deep Dive', threshold_percent=20)
            plot_deep_dive_heatmap(G_core, 'product_categories', 'Glasso Product Heatmap', threshold_percent=10)
            plot_deep_dive_heatmap(G_core, 'product_subcategories', 'Glasso SubCat Heatmap', threshold_percent=10)
            plot_deep_dive_heatmap(G_core, 'genus_id', 'Glasso Genus Deep Dive', threshold_percent=10)
            plot_deep_dive_heatmap(G_core, 'family_id', 'Glasso Family Deep Dive', threshold_percent=10)
            plot_deep_dive_heatmap(G_core, 'species_id', 'Glasso Species Deep Dive', threshold_percent=10)
            
    except Exception as e:
        print(f"ERROR in GLASSO Pipeline: {e}")



    # PART C: SPEARMAN
  
    print("\n--- PART C: SPEARMAN PIPELINE ---")
    
    print("Calculating Spearman Correlation...")
    spearman_corr = df_final.corr(method='spearman')
    
    spearman_corr.index.name = 'gcf1'
    spearman_corr.columns.name = 'gcf2'
    
    links = spearman_corr.stack().reset_index()
    links.columns = ['var1', 'var2', 'value']
    

    links_filtered = links.loc[
        (links['value'] >= SPEARMAN_THRESHOLD) & 
        (links['var1'] != links['var2'])
    ]
    
    G_spearman = nx.from_pandas_edgelist(links_filtered, 'var1', 'var2', edge_attr='value')
    print(f"Spearman Graph: {G_spearman.number_of_nodes()} nodes")
    
    for u, v, d in G_spearman.edges(data=True):
        d['weight'] = abs(d['value'])
        d['distance'] = 1.0 - abs(d['value'])
        
    if len(G_spearman.nodes) > 0:
        print("   Metrics...")
        bet = nx.betweenness_centrality(G_spearman, normalized=True, weight="distance")
        nx.set_node_attributes(G_spearman, bet, 'betweenness_score')
        
        harm_raw = nx.harmonic_centrality(G_spearman, distance="distance")
        nx.set_node_attributes(G_spearman, {k: v/(len(G_spearman)-1) for k,v in harm_raw.items()}, 'harmonic_score')
        
        neigh_raw = nx.average_neighbor_degree(G_spearman, weight="weight")
        nx.set_node_attributes(G_spearman, neigh_raw, 'neighborhood_connectivity')
        
        # Louvain
        part_spearman = community_louvain.best_partition(G_spearman, weight='weight',random_state=42, resolution=4)
        rank_communities_by_size(G_spearman, part_spearman, "Spearman")
        mod_spearman = community_louvain.modularity(part_spearman, G_spearman, weight='weight')
        print(f"[METRIC] Spearman Network Modularity (Q): {mod_spearman:.4f}")
        
        map_all_metadata(G_spearman, df_mapping)
        nx.write_graphml(G_spearman, f"{OUTPUT_DIR}/network_spearman.graphml")
        
        # Plots
        analyze_community_composition(G_spearman, 'dominant_biome', 'Spearman Ecological Composition (Biome)')
        analyze_community_composition(G_spearman, 'product_categories', 'Spearman Product', 'Product')
        analyze_community_composition(G_spearman, 'product_subcategories', 'Spearman Sub Product', 'Sub Type')
        analyze_community_composition(G_spearman, 'genus_id', 'Spearman Taxonomic Composition (Genus Level)')
        analyze_community_composition(G_spearman, 'family_id', 'Spearman Taxonomic Composition (Family Level)')
        analyze_community_composition(G_spearman, 'species_id', 'Spearman Taxonomic Composition (Species Level)')
        plot_deep_dive_heatmap(G_spearman, 'dominant_biome', 'Spearman Ecological Deep Dive', threshold_percent=20)
        plot_deep_dive_heatmap(G_spearman, 'product_categories', 'Spearman Product Heatmap', threshold_percent=10)
        plot_deep_dive_heatmap(G_spearman, 'product_subcategories', 'Spearman Sub Product Heatmap', threshold_percent=10)
        plot_deep_dive_heatmap(G_spearman, 'genus_id', 'Spearman Genus Deep Dive', threshold_percent=20)
        plot_deep_dive_heatmap(G_spearman, 'family_id', 'Spearman Family Deep Dive', threshold_percent=20)
        plot_deep_dive_heatmap(G_spearman, 'species_id', 'Spearman Species Deep Dive', threshold_percent=15)

    #summary of metrics
    print_network_summary(G_jaccard, part_jaccard, "JACCARD")

    print_network_summary(G_core, part_glasso, "GLASSO")

    print_network_summary(G_spearman, part_spearman, "SPEARMAN")
    
    print("CROSS-METHOD VALIDATION")
    validate_community_consistency(G_core, G_spearman, "Glasso", "Spearman")
    
    print("MASTER PIPELINE DONE")
if __name__ == "__main__":
     main()