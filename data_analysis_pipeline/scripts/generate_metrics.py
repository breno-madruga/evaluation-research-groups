##########################################################################################################
######################## Generating CNA metrics datasets from collaboration nets #########################
##########################################################################################################

########################################################################
# Importing the required libraries.
import csv, pandas as pd, networkx as nx, glob
########################################################################

########################################################################
# 1. Getting and preprocessing the data
########################################################################

# Function to calculate the metrics of research group.
def get_records_per_metric(metric_name, metric_per_nodes, nodes_attributes):
    records = []
    for id, metric in metric_per_nodes.items():
        record = {}
        record["id"] = id
        record["metric_value"] = metric
        record["metric_name"] = metric_name
        record["complete_name"] = nodes_attributes[id]["complete_name"]
        record["h_index"] = nodes_attributes[id]["h_index"]
        record["is_permanent"] = nodes_attributes[id]["is_permanent"]
        record["research_line"] = nodes_attributes[id]["research_line"] if record["is_permanent"] else None
        records.append(record)
    return pd.DataFrame(records).sort_values("complete_name")

# Function to generate the metrics dataset.
def get_metrics_complete(networks, year_papers, name_dataset):
    list_result = []
    for net in networks:
        df_result = pd.DataFrame()
        G = nx.read_gexf(net)
        temp = G.subgraph(max(nx.connected_components(G), key=len))
        metrics = {"degree": nx.degree_centrality(G),
                   "betweenness": nx.betweenness_centrality(G, normalized=True, weight="num_paper"),
                   "eigenvector": nx.eigenvector_centrality(G, max_iter=2000, weight="num_paper"),
                   "closeness": nx.closeness_centrality(G, distance="num_paper", wf_improved=True),
                   "clustering": nx.clustering(G, weight="num_paper"),
                   "num_cliques": nx.number_of_cliques(G),
                   "eccentricity_scc": nx.eccentricity(temp)
        }
        for metric_name, metric_values in metrics.items():
            records = get_records_per_metric(metric_name, metric_values, dict(
                G.nodes(data=True) if not metric_name == "eccentricity_scc" else temp.nodes(data=True)))
            df_result = pd.concat([df_result, records], ignore_index=True)
        df_result.loc[:, "year"] = net.split("/")[-1].split("_")[0]
        if "window" in net:
            step = int(net.split("/")[-1].split("_")[-2])
            step -= 1
        else:
            step = None
        df_result.loc[:, "total_num_paper"] = year_papers[year_papers == int(df_result.loc[0, "year"]) \
            if "cumulative" not in net else \
            year_papers <= int(df_result.loc[0, "year"]) \
            if pd.isnull(step) else \
            (year_papers >= int(df_result.loc[0, "year"]) - step) & (year_papers <= int(df_result.loc[0, "year"]))].count()
        records = df_result[df_result.metric_name == "eccentricity_scc"].copy()
        records.loc[:, "metric_name"] = "eccentricity_scc_normed"
        records.loc[:, "metric_value"] = records["metric_value"] / temp.number_of_nodes() - 1
        df_result = pd.concat([df_result, records], ignore_index=True)
        list_result.append(df_result.copy())

    pd.concat(list_result, ignore_index=True).to_csv(name_dataset, index=False, quoting=csv.QUOTE_ALL)

# Getting the data.
df_data = pd.read_csv("data_analysis_pipeline/data/prepared/production_members_final.csv", header=0, index_col=False)

# Preprocessing the data.
df_data = df_data[["members_name", "year"]]
df_data.members_name = df_data.members_name.apply(lambda x: eval(x) if x else None)

########################################################################
# 2. Generating and saving the CNA metrics datasets
########################################################################

# Getting the metrics.
networks = sorted(glob.glob("data_analysis_pipeline/data/networks/*network.gexf"))
get_metrics_complete(networks, df_data.year, "data_analysis_pipeline/data/prepared/metrics_isolated.csv")

networks = sorted(glob.glob("data_analysis_pipeline/data/networks/*network_cumulative.gexf"))
get_metrics_complete(networks, df_data.year, "data_analysis_pipeline/data/prepared/metrics_cumulative.csv")

networks = sorted(glob.glob("data_analysis_pipeline/data/networks/*network_cumulative*_4_window.gexf"))
get_metrics_complete(networks, df_data.year, "data_analysis_pipeline/data/prepared/metrics_cumulative_4_window.csv")

networks = sorted(glob.glob("data_analysis_pipeline/data/networks/*network_cumulative*_2_window.gexf"))
get_metrics_complete(networks, df_data.year, "data_analysis_pipeline/data/prepared/metrics_cumulative_2_window.csv")