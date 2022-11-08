##########################################################################################################
############## Generating the Temporal Networks from the Production data and Members' stats ##############
##########################################################################################################

########################################################################
# 1. Importing the libraries
########################################################################

# Importing the required libraries.
import numpy as np, pandas as pd, networkx as nx, os
from itertools import combinations

# Creating the "networks" directory.
os.makedirs("data_analysis_pipeline/data/networks/", exist_ok=True)

########################################################################
# 2. Getting the data
########################################################################

########################################################################
# 2.1. Manuscripts dataset
########################################################################

# Getting the data.
df_data = pd.read_csv("data_analysis_pipeline/data/prepared/production_members_final.csv", header=0, dtype={"id": str})

# Checking some information about the dataset.
print(df_data.info())

########################################################################
# 2.2. Members dataset
########################################################################

# Getting the data.
df_members = pd.read_csv("data_analysis_pipeline/data/prepared/members_stats_final.csv", header=0)

# Checking some information about the dataset.
print(df_members.info())

########################################################################
# 3. Preprocessing the datasets
########################################################################

########################################################################
# 3.1. Members' data
########################################################################

# Changing the "NaN" values by "None" ones.
df_members.replace({np.nan: None}, inplace=True)

# Changing the type of features with composite data.
df_members.loc[:, ["identifiers", "subject_areas", "affiliation_current", "affiliation_history"]] = \
df_members.loc[:, ["identifiers", "subject_areas", "affiliation_current", "affiliation_history"]].apply(
    lambda x: x.apply(lambda y: eval(y) if y else None))

########################################################################
# 3.2. Manuscripts data
########################################################################

# Changing the "NaN" values by "None" ones.
df_data.replace({np.nan: None}, inplace=True)

# Defining "zero" value to the empty records.
df_data.citation_num.fillna(0, inplace=True)
df_data.ref_count.fillna(0, inplace=True)

# Changing the type of features with composite data.
df_data.loc[:, ["members_name", "auth_keywords", "index_terms", "affiliations",
    "subject_areas", "authors", "author_affil", "references"]] = \
df_data.loc[:, ["members_name", "auth_keywords", "index_terms", "affiliations",
    "subject_areas", "authors", "author_affil", "references"]].apply(
        lambda x: x.apply(lambda y: eval(y) if y else None))

# Normalizing the feature "authors".
columns = ["identifiers", "complete_name", "h_index", "is_permanent", "research_line"]
df_data.authors = [[
    (author["id"], {"complete_name": author["name"], "h_index": 1, "is_permanent": False}) \
    if df_members[[author["id"] in id_scopus for id_scopus in df_members.identifiers]].empty else
    df_members.loc[[author["id"] in id_scopus for id_scopus in df_members.identifiers], columns].apply(
        lambda x: (x[columns[0]][0], x[columns[1:]].to_dict()), axis=1).iloc[0] \
    if df_members.is_permanent[df_members[
        [author["id"] in id_scopus for id_scopus in df_members.identifiers]].index].iloc[0] else
    df_members.loc[[author["id"] in id_scopus for id_scopus in df_members.identifiers], columns[:-1]].apply(
        lambda x: (x[columns[0]][0], x[columns[1:-1]].to_dict()), axis=1).iloc[0]
    for author in authors] for authors in df_data.authors]

# Normalizing the feature "affiliations".
df_data.affiliations = [[(affil["id"], {"complete_name": affil["affiliation"],
    "country": affil["country"]} if affil["country"] else {"complete_name": affil["affiliation"]})
                         for affil in affils if affil["id"] and affil["affiliation"]] \
        if affils else None for affils in df_data.affiliations]

# Listing the five highest number of authors.
print(df_data.authors.apply(lambda x: len(x)).sort_values(ascending=False).head(10))
print(df_data.id[df_data.title.str.lower().str.contains("editorial") |
                 df_data.title.str.lower().str.contains("preface")])

# Ignoring the manuscript with the highest number of authors ("Editorial/Preface").
df_data = df_data[~df_data.id.isin(["79551633927", "85111369731", "85050805722", "84893597292",
                                    "85082195656", "85125064739", "85061298248",
                                    "85126284305", "85110504864", "85112304772",
                                    "85029483948", "85100291392", "85126083408"])]

# Checking some information about the dataset after the preprocessing.
print(df_data.info())

########################################################################
# 4. Generating and saving the Temporal Networks
########################################################################

def create_temporal_nets(df, feature, flag=False, step=None):
    if step and step > 1:
        step -= 1
    # Creating the collaboration networks for isolated (flag = False) and cumulative (flag = True) contexts.
    for year in sorted(df.year.unique().tolist()):
        # Creating a collabotation network (undirected graph).
        G = nx.Graph()
        temp = df.loc[(df[feature].notnull()) & (df.year == year), [feature, "citation_num"]].copy() \
            if not flag and pd.isnull(step) else \
                df.loc[(df[feature].notnull()) & (df.year <= year), [feature, "citation_num"]].copy() \
                    if flag and pd.isnull(step) else \
                        df.loc[(df[feature].notnull()) & (df.year >= year - step) & (df.year <= year),
                               [feature, "citation_num"]].copy()
        for idx, paper in temp.iterrows():
            # Adding the nodes, along with their attributes, to the graph.
            G.add_nodes_from(list(paper[feature]))

            # Getting the list of nodes' IDs.
            nodes = [item[0] if type(item) == tuple else item for item in paper[feature]]

            # Adding the edges to the graph (include the auto-loops).
            edges = set(combinations(nodes, 2)) if len(nodes) > 1 else [(nodes[0],)*2]
            for edge in edges:
                if G.has_edge(*edge):
                    G.edges[edge]["num_paper"] += 1
                    G.edges[edge]["citation_num"] += paper.citation_num
                else:
                    G.add_edge(*edge, num_paper=1)
                    G.add_edge(*edge, num_paper=1, citation_num=paper.citation_num)

        # Saving the graph.
        nx.write_gexf(G, "data_analysis_pipeline/data/networks/{}_{}_network.gexf".format(
                year, feature) if not flag and pd.isnull(step) \
            else "data_analysis_pipeline/data/networks/{}_{}_network_cumulative.gexf".format(
                year, feature) if flag and pd.isnull(step) \
            else "data_analysis_pipeline/data/networks/{}_{}_network_cumulative_{}_window.gexf".format(
                year, feature, step+1))

# Generating the temporal authorship networks.
create_temporal_nets(df_data, "authors")
create_temporal_nets(df_data, "authors", True)
create_temporal_nets(df_data, "authors", True, 4)
create_temporal_nets(df_data, "authors", True, 2)