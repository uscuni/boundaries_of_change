import geopandas as gpd
import pandas as pd
from scipy.spatial import distance

def get_buildings(path_to_buildings):
    buildings = gpd.read_parquet(path_to_buildings)
    
    final_clusters = pd.read_parquet(
        "/data/uscuni-ulce/processed_data/clusters/cluster_mapping_v3.pq")
    
    level_columns = []
    
    for level in final_clusters.columns:
        buildings[f"level_{level}"] = buildings.final_without_noise.map(
            final_clusters[level]
        )
        level_columns.append(f"level_{level}")
    
    buildings = buildings.rename(columns={"final_without_noise":"level_7"})
    level_columns.append("level_7")
    return buildings, level_columns

def get_building_counts_per_region(buildings, boundary, level="level_4"):
    joined = gpd.sjoin(buildings, boundary, how="left", predicate="within")
    grouped = joined.groupby("NAME")[level].value_counts()
    cnt_table = pd.DataFrame(grouped.unstack().fillna(0).T)
    
    return cnt_table

def normalize_building_counts(count_table):
    region_building_sums = count_table.sum(axis=0)
    cnt_table_norm = count_table.div(region_building_sums, axis=1)
    return cnt_table_norm.T

def compute_bc_matrix(count_table):
    bray_curtis_dist = distance.pdist(count_table, metric='braycurtis')
    bray_curtis_matrix = distance.squareform(bray_curtis_dist)
    bray_curtis_df = pd.DataFrame(bray_curtis_matrix, index=count_table.T.columns, columns=count_table.T.columns)
    return bray_curtis_df

def generate_bc_matrices(boundary_list, level_list):
    for bnd in boundary_list:
        boundary = gpd.read_file("/data/uscuni-ulce/boundaries_of_change/impact_boundaries.gpkg", layer=bnd)
        
        for level in level_list:
            building_counts = get_building_counts_per_region(buildings, boundary, level)
            building_counts_norm = normalize_building_counts(building_counts)
            
            bray_curtis_matrix = compute_bc_matrix(building_counts_norm)
    
            bray_curtis_matrix.to_parquet(f'/data/uscuni-ulce/boundaries_of_change/bc_matrices/bc_{bnd}_{level}.pq')