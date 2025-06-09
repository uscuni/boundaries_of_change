import geopandas as gpd
import pandas as pd
from scipy.spatial import distance

def get_buildings(path_to_buildings):
    """Returns building centroids with their cluster assignment at various levels of aggregation, and a list of these levels."""
    #Read buildings dataset
    buildings = gpd.read_parquet(path_to_buildings)
    #Read cluster mapping
    final_clusters = pd.read_parquet(
        "/data/uscuni-ulce/processed_data/clusters/cluster_mapping_v10.pq")
    
    level_columns = []
    #Remap building centroids to clusters on levels in 'final_clusters' (max level 7)
    for level in final_clusters.columns:
        buildings[f"level_{level}"] = buildings.final_without_noise.map(
            final_clusters[level]
        )
        level_columns.append(f"level_{level}")
    #Rename lowest level to level 7
    level_7 = buildings.pop("final_without_noise")
    buildings["level_7"] = level_7
    level_columns.append("level_7")
    return buildings, level_columns

def get_building_counts_per_region(buildings, boundary, level="level_4"):
    """Returns count table of buildings per region in a specified layer."""
    joined = gpd.sjoin(buildings, boundary, how="left", predicate="within")
    grouped = joined.groupby("NAME")[level].value_counts()
    cnt_table = pd.DataFrame(grouped.unstack().fillna(0).T)
    
    return cnt_table

def normalize_building_counts(count_table):
    """Returns normalized count table."""
    region_building_sums = count_table.sum(axis=0)
    cnt_table_norm = count_table.div(region_building_sums, axis=1)
    return cnt_table_norm.T

def compute_bc_matrix(count_table):
    """Computes Bray-Curtis dissimilarity matrix from input table."""
    bray_curtis_dist = distance.pdist(count_table, metric='braycurtis')
    bray_curtis_matrix = distance.squareform(bray_curtis_dist)
    bray_curtis_df = pd.DataFrame(bray_curtis_matrix, index=count_table.T.columns, columns=count_table.T.columns)
    return bray_curtis_df