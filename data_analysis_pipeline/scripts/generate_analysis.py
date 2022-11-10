##########################################################################################################
########## Generating Insights for Temporal Evaluation and Cohesion Analysis of Research Groups ##########
##########################################################################################################

########################################################################
# 1. Importing the libraries
########################################################################

# Importing required libraries.
import os, csv, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import networkx as nx, glob
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from node2vec import Node2Vec
from karateclub import Graph2Vec

# Creating the required directories.
os.makedirs("data_analysis_pipeline/data/output", exist_ok=True)

# Defining a function to config Matplotlib.
def set_config_plt():
    sns.reset_orig()
    plt.style.use("ggplot")
    plt.rcParams.update({"figure.figsize": (9, 7), "figure.autolayout": True,
        "figure.dpi": 180, "font.size": 18, "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.edgecolor": "black"})

# Configuring the Matplotlib.
set_config_plt()

########################################################################
# 2. Getting and checking the information dataset
########################################################################

########################################################################
# 2.1. Isolated Metrics of Research Group
########################################################################

# Importing the data.
df_metrics_iso = pd.read_csv("data_analysis_pipeline/data/prepared/metrics_isolated.csv", delimiter=",", header=0)

# Checking some information about the data.
print(df_metrics_iso.info())

########################################################################
# 2.2. Cumulative Sliding Window for Metrics of Research Group
########################################################################

# Importing the data.
df_metrics_agg = pd.read_csv("data_analysis_pipeline/data/prepared/metrics_cumulative.csv", delimiter=",", header=0)

# Checking some information about the data.
print(df_metrics_agg.info())

########################################################################
# 2.3. Cumulative Metrics of Research Group with 4-window
########################################################################

# Importing the data.
df_metrics_4_window = pd.read_csv("data_analysis_pipeline/data/prepared/metrics_cumulative_4_window.csv", delimiter=",", header=0)

# Checking some information about the data.
print(df_metrics_4_window.info())

########################################################################
# 2.4. Cumulative Metrics of Research Group with 2-window
########################################################################

# Importing the data.
df_metrics_2_window = pd.read_csv("data_analysis_pipeline/data/prepared/metrics_cumulative_2_window.csv", delimiter=",", header=0)

# Checking some information about the data.
print(df_metrics_2_window.info())

########################################################################
# 2.5. Members' Isolated Metrics
########################################################################

# Getting the data.
df_members_iso = df_metrics_iso[df_metrics_iso.is_permanent == True]

# Visualizing the information about the data.
print(df_members_iso.info())

########################################################################
# 2.6. Members' Metrics for Cumulative Sliding Window
########################################################################

# Getting the data.
df_members_agg = df_metrics_agg[df_metrics_agg.is_permanent == True]

# Visualizing the information about the data.
print(df_members_agg.info())

########################################################################
# 2.7. Members' 4-window Metrics
########################################################################

# Getting the data.
df_members_4_window = df_metrics_4_window[df_metrics_4_window.is_permanent == True]

# Visualizing the information about the data.
print(df_members_4_window.info())

########################################################################
# 2.8. Members' 2-window Metrics
########################################################################

# Getting the data.
df_members_2_window = df_metrics_2_window[df_metrics_2_window.is_permanent == True]

# Visualizing the information about the data.
print(df_members_2_window.info())

########################################################################
# 2.9. Research Group's Production data
########################################################################

# Importing the data.
df_data = pd.read_csv("data_analysis_pipeline/data/prepared/production_members_final.csv", index_col=False, header=0, dtype=object)

# Visualizing the information about the data.
print(df_data.info())

########################################################################
# 2.10. Researchers' Stats and Data
########################################################################

# Importing the data.
df_members = pd.read_csv("data_analysis_pipeline/data/prepared/members_stats_final.csv", index_col=False, header=0)

# Visualizing the information about the data.
print(df_members.info())

########################################################################
# 2.11. Research Group's Temporal Nets
########################################################################

# Getting the data.
net_isolated = sorted(glob.glob("data_analysis_pipeline/data/networks/*network.gexf"))
net_cumulative = sorted(glob.glob("data_analysis_pipeline/data/networks/*network_cumulative.gexf"))
net_4_window = sorted(glob.glob("data_analysis_pipeline/data/networks/*network_cumulative_4_window.gexf"))
net_2_window = sorted(glob.glob("data_analysis_pipeline/data/networks/*network_cumulative_2_window.gexf"))

########################################################################
# 3. Cleaning the dataframes
########################################################################

# Changing the invalid values to "None".
df_metrics_iso.replace({np.nan: None}, inplace=True)
df_metrics_agg.replace({np.nan: None}, inplace=True)
df_metrics_4_window.replace({np.nan: None}, inplace=True)
df_metrics_2_window.replace({np.nan: None}, inplace=True)
df_members_iso.replace({np.nan: None}, inplace=True)
df_members_agg.replace({np.nan: None}, inplace=True)
df_members_4_window.replace({np.nan: None}, inplace=True)
df_members_2_window.replace({np.nan: None}, inplace=True)
df_data.replace({np.nan: None}, inplace=True)
df_members.replace({np.nan: None}, inplace=True)

# Reading each network.
net_isolated = dict(zip(sorted(df_metrics_iso.year.unique()),
                        [nx.read_gexf(graph) for graph in net_isolated]))
net_cumulative = dict(zip(sorted(df_metrics_iso.year.unique()),
                          [nx.read_gexf(graph) for graph in net_cumulative]))
net_4_window = dict(zip(sorted(df_metrics_iso.year.unique()),
                      [nx.read_gexf(graph) for graph in net_4_window]))
net_2_window = dict(zip(sorted(df_metrics_iso.year.unique()),
                      [nx.read_gexf(graph) for graph in net_2_window]))

# Removing unnecessary columns.
columns = ["id", "h_index", "is_permanent", "research_line"]
df_members_iso.drop(columns=columns, inplace=True)
df_members_agg.drop(columns=columns, inplace=True)
df_members_4_window.drop(columns=columns, inplace=True)
df_members_2_window.drop(columns=columns, inplace=True)

# Aggregating the data.
columns_agg = {"metric_value": "mean", "total_num_paper": "first"}
columns = ["metric_name", "complete_name", "year"]
df_members_iso = df_members_iso.groupby(columns).agg(columns_agg).reset_index().rename(
    columns={"complete_name": "researcher"})
df_members_agg = df_members_agg.groupby(columns).agg(columns_agg).reset_index().rename(
    columns={"complete_name": "researcher"})
df_members_4_window = df_members_4_window.groupby(columns).agg(columns_agg).reset_index().rename(
    columns={"complete_name": "researcher"})
df_members_2_window = df_members_2_window.groupby(columns).agg(columns_agg).reset_index().rename(
    columns={"complete_name": "researcher"})

# Changing the type of numeric columns.
df_data.year = df_data.year.apply(lambda x: int(x) if not pd.isnull(x) else None)
df_data.citation_num = df_data.citation_num.apply(lambda x: int(x) if not pd.isnull(x) else None)
df_data.month = df_data.month.apply(lambda x: int(x) if not pd.isnull(x) else None)
df_data.ref_count = df_data.ref_count.apply(lambda x: int(float(x)) if not pd.isnull(x) else None)

# Converting from the "str" type to the "list" type of some columns of data.
df_data.members_name = df_data.members_name.apply(lambda x: eval(x) if x else None)
df_data.auth_keywords = df_data.auth_keywords.apply(lambda x: eval(x) if x else None)
df_data.index_terms = df_data.index_terms.apply(lambda x: eval(x) if x else None)
df_data.affiliations = df_data.affiliations.apply(lambda x: eval(x) if x else None)
df_data.subject_areas = df_data.subject_areas.apply(lambda x: eval(x) if x else None)
df_data.authors = df_data.authors.apply(lambda x: eval(x) if x else None)
df_data.author_affil = df_data.author_affil.apply(lambda x: eval(x) if x else None)
df_data.references = df_data.references.apply(lambda x: eval(x) if x else None)

# Defining the number of articles per year for each researcher.
df_members_iso.loc[:, "num_paper"] = df_members_iso[["researcher", "year"]].apply(lambda row: df_data[
    (df_data.year == row.year) &
    ([row.researcher in members for members in df_data.members_name])].id.size, axis=1)
df_members_agg.loc[:, "num_paper"] = df_members_agg[["researcher", "year"]].apply(lambda row: df_data[
    (df_data.year <= row.year) &
    ([row.researcher in members for members in df_data.members_name])].id.size, axis=1)
df_members_4_window.loc[:, "num_paper"] = df_members_4_window[["researcher", "year"]].apply(lambda row: df_data[
    (df_data.year >= row.year - 3) & (df_data.year <= row.year) &
    ([row.researcher in members for members in df_data.members_name])].id.size, axis=1)
df_members_2_window.loc[:, "num_paper"] = df_members_2_window[["researcher", "year"]].apply(lambda row: df_data[
    (df_data.year >= row.year - 1) & (df_data.year <= row.year) &
    ([row.researcher in members for members in df_data.members_name])].id.size, axis=1)

# Creating the feature "percentual_num_paper".
df_members_iso["percentual_num_paper"] = (df_members_iso["num_paper"] / df_members_iso["total_num_paper"]) * 100
df_members_agg["percentual_num_paper"] = (df_members_agg["num_paper"] / df_members_agg["total_num_paper"]) * 100
df_members_4_window["percentual_num_paper"] = (df_members_4_window["num_paper"] / df_members_4_window["total_num_paper"]) * 100
df_members_2_window["percentual_num_paper"] = (df_members_2_window["num_paper"] / df_members_2_window["total_num_paper"]) * 100

# Converting from the "str" type to the "list" type of some columns of data.
df_members.identifiers = df_members.identifiers.apply(lambda x: eval(x) if x else None)
df_members.subject_areas = df_members.subject_areas.apply(lambda x: eval(x) if x else None)
df_members.affiliation_current = df_members.affiliation_current.apply(lambda x: eval(x) if x else None)
df_members.affiliation_history = df_members.affiliation_history.apply(lambda x: eval(x) if x else None)

########################################################################
# 4. Utils functions
########################################################################

########################################################################
# 4.1. Visualization methods
########################################################################

def plot_single_line_chart(x_values, y_values, x_label, y_label, title, name_fig=None, pos_vline=None, text_vline=None):
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, marker="o")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x_values)
    if pos_vline:
        ax.axvline(pos_vline, ymax=0.92, color="#400000", linewidth=4)
        ax.text(pos_vline + 0.1, y_values[pos_vline - 1] + 0.1, text_vline, color="black")
    ax.set_ylim([0, max(y_values) * 1.1])
    fig.tight_layout()
    if name_fig:
        plt.savefig(name_fig)
    plt.close()


def plot_multiple_line_chart(dataframe, x_value, y_values, y_labels, title, xaxis_label, yaxis_label,
                             title_legend=None, name_fig=None, y_is_int=False, external_legend=True,
                             has_legend=True):
    fig, ax = plt.subplots()
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xaxis_label)
    ax.set_ylabel(yaxis_label)
    for idx in range(len(y_values)):
        ax.plot(dataframe[x_value], dataframe[y_values[idx]], marker="o", label=y_labels[idx])
    ax.set_xticks(dataframe[x_value])
    ax.set_xticklabels(ax.get_xticks(), rotation = 45)
    if y_is_int:
        ax.set_yticks([item for item in np.arange(0, dataframe[y_values].max().max() + 1, 2)])
    else:
        ax.set_yticks([float("{:.3f}".format(item)) for item in np.linspace(0, dataframe[y_values].max().max(), 10)])
    ax.set_ylim([0, dataframe[y_values].max().max() * 1.1])
    plt.axvspan(2019, 2021, color='gray', alpha=0.3)
    if has_legend:
        ax.legend(title=title_legend, fontsize="medium")
    if external_legend and has_legend:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    if name_fig:
        plt.savefig(name_fig)
    plt.close()


def plot_multiple_line_chart_sns(dataframe, x_value, y_value, y_value_style, title, xaxis_label, yaxis_label,
                                 y_lim_max=1, has_error=False, has_legend=True, title_legend=None, labels_legend=None,
                                 external_legend=False, highlight_pandemic=True, name_fig=None):

    colors = ["#0e77ab","#39c8c6","#e16670","#7d2b32","#e6c7ca",
              "#9b5de5","#f15bb5","#fee440","#00bbf9","#00f5d4", "#a9bac7",
              "#a3cef1","#8b8c89", "#ff9770","#ffd670","#adcc00", "#f992ad",
              "#fab4c8","#f8a0d6","#d4b0f9", "#1c3144"][:dataframe[y_value_style].unique().size]

    ax = sns.lineplot(data=dataframe, x=x_value, y=y_value, hue=y_value_style, style=y_value_style, markers=True,
                      legend="auto" if has_legend else has_legend, ci="sd" if has_error else None,
                      palette=colors)
    ax.set_title(title, fontsize=16)
    ax.set_ylabel(yaxis_label)
    ax.set_ylim([0, y_lim_max])
    ax.set_xlabel(xaxis_label)
    ax.set_xticks(sorted(dataframe[x_value].unique()))
    ax.set_xticklabels(ax.get_xticks(), rotation = 45)
    if highlight_pandemic:
        plt.axvspan(2019, 2021, color="gray", alpha=0.2)
    if external_legend and has_legend:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="medium")
    if title_legend and has_legend:
        ax.get_legend().set_title(title_legend)
    if labels_legend and has_legend:
        for t, l in zip(ax.get_legend().texts, labels_legend):
            t.set_text(l)
    plt.tight_layout()
    if name_fig:
        plt.savefig(name_fig)
    plt.close()


def plot_clusters(dataframe, x_value, y_value, class_value, size_value, label_value, x_label, y_label, title, num_bins=5,
                  space_legend=2, name_fig=None):
    pallete = ['#4198B7', '#39C8C6', '#D3500C', '#FFB139',
               '#0E77AB', '#1E3544', '#041434', '#12B5C4',
               '#630C3A']
    fig, ax = plt.subplots(figsize=(12,7))
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    sc = ax.scatter(dataframe[x_value], dataframe[y_value], s=dataframe[size_value] * 10,
                    c=dataframe[class_value].apply(lambda x: pallete[int(x)]))
    for idx, record in dataframe.iterrows():
        ax.annotate(record[label_value], (record[x_value], record[y_value]),
                    xytext=(0, 1), textcoords="offset points", ha="center", va="center", fontsize=13)
    handles, labels = sc.legend_elements(prop="sizes", alpha=0.5, num=num_bins)
    labels = np.linspace(dataframe[size_value].min(), dataframe[size_value].max(), num_bins, dtype=int)
    ax.legend(handles, labels, title="Nº. of Manuscripts", fontsize="small",
              title_fontsize="medium", loc='center left', bbox_to_anchor=(1, 0.5),
              borderpad=1, handletextpad=2, labelspacing=space_legend, frameon=False)
    fig.tight_layout()
    if name_fig:
        plt.savefig(name_fig)
    plt.close()


def plot_overlapping_kdes(dataframe, x_value, hue_value, x_label, xlim=None, name_fig=None):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    # Initializing the FacetGrid object.
    pal = sns.cubehelix_palette(13, rot=-.25, light=.7)
    ax = sns.FacetGrid(dataframe, row=hue_value, hue=hue_value, aspect=15, height=.5, palette=pal)

    # Drawing the densities in a few steps.
    ax.map(sns.kdeplot, x_value, bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    ax.map(sns.kdeplot, x_value, clip_on=False, color="w", lw=2, bw_adjust=.5)

    # Passing color=None to refline() uses the hue mapping.
    ax.refline(y=0, linewidth=1, linestyle="-", color=None, clip_on=False)

    # Defining and using a simple function to label the plot in axes coordinates.
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)
    ax.map(label, x_value)

    # Setting the subplots to overlap.
    ax.figure.subplots_adjust(hspace=-.25)

    # Removing axes details that don't play well with overlap.
    ax.set_titles("")
    ax.set(yticks=[], ylabel="")
    ax.despine(bottom=True, left=True)
    ax.set_xlabels("")
    ax.set(xlim=(None, xlim))
    if name_fig:
        plt.savefig(name_fig)
    plt.close()


def plot_confusion_matrix(y_real, y_pred, name_fig=None):
    ConfusionMatrixDisplay.from_predictions(y_real, y_pred, colorbar=False)
    if name_fig:
        plt.savefig(name_fig)
    plt.close()


########################################################################
# 4.2. Clustering Methods
########################################################################

# Defining the function to cluster the data.
def clustering_kmeans(dataframe, n_clusters):
    # Preprocessing the data.
    ss = StandardScaler()
    X = ss.fit_transform(dataframe)

    # Training the model and clustering the data.
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X)
    y_pred = model.predict(X)

    return y_pred


# Defining the Elbow method.
def elbow_method(dataframe, max_k):
    # Preprocessing the data.
    ss = StandardScaler()
    X = ss.fit_transform(dataframe)

    sse = dict()
    for k in range(1, max_k + 1):
        temp = pd.DataFrame(data=X, columns=dataframe.columns.tolist())
        sse[k] = pd.Series([0] * temp.columns.size, index=temp.columns)
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(X)
        temp["label"] = model.predict(X)
        for c in range(k):
            sse[k] += np.sum(np.power(
              temp.iloc[[c == cluster for cluster in temp.label], :-1] - model.cluster_centers_[c], 2))
        sse[k] = sse[k].sum()
    return pd.Series(sse)


# Function to generate a net with the cluster of each node.
def generate_net_cluster(graph, dataframe, name_file):
    G = nx.Graph(graph)
    attrs = {row.id: {"cluster": row.cluster,
                      "num_paper": row.num_paper,
                      "id_alt": row.id_alt,
                      "pred_rl": row.pred_rl}
             for row in dataframe.itertuples()}
    nx.set_node_attributes(G, attrs)
    nx.write_gexf(G, name_file)


# Function to discretize the research line for each member.
def discretize_research_line(research_line):
    if research_line == "Automation and Systems":
        return 1
    elif research_line == "Computer Engineering":
        return 2
    elif research_line == "Telecommunication":
        return 3
    else:
        return None


# Function to define the research line from the cluster.
def set_research_line_from_cluster(dataframe, num_members_per_line):
    dataframe["pred_rl"] = None
    for c in sorted(dataframe.cluster.unique()):
        temp = dataframe.real_rl[dataframe.cluster == c].value_counts()
        temp = temp / num_members_per_line[temp.index.tolist()]
        main_line = temp[temp == temp.max()].index.tolist()
        if len(main_line) == 1:
            dataframe.loc[dataframe.cluster == c, "pred_rl"] = main_line[0]
        elif len(main_line) > 1:
            temp = dataframe.loc[
                (dataframe.cluster == c) & (dataframe.real_rl.isin(main_line)),
                ["num_paper", "real_rl"]
            ].groupby("real_rl").num_paper.sum().sort_values(ascending=False)
            dataframe.loc[dataframe.cluster == c, "pred_rl"] = temp.index.tolist()[0]
    dataframe["pred_rl"] = dataframe["pred_rl"].astype(np.int32)
    dataframe["pred_rl"] = dataframe["pred_rl"].astype("category")
    return dataframe.pred_rl.copy()


########################################################################
# 4.3. Embedding Methods
########################################################################

# Function to create the embedding vector from a graph.
def get_vector_from_graph(graph, num_dimension=2, num_window=10):
    node2vec = Node2Vec(graph, dimensions=num_dimension)
    model = node2vec.fit(window=num_window)
    data = []
    for node in graph.nodes(data=True):
        vector = model.wv.get_vector(str(node[0]))
        vector = {"column_{}".format(n + 1): vector[n] for n in range(num_dimension)}
        data.append({**node[1], **vector})
    return pd.DataFrame(data).rename(columns={"label": "id_scopus"})


# Function to create the embedding vectors from a list of graph.
def get_graph_embedding(graphs, num_dimension=2):
    graphs = {k: nx.convert_node_labels_to_integers(G) for k, G in graphs.items()}
    model = Graph2Vec(dimensions=num_dimension)
    model.fit(list(graphs.values()))
    embeddings = model.get_embedding()
    data = pd.DataFrame(columns=["year", *["column_{}".format(n + 1) for n in range(num_dimension)]])
    data["year"] = list(graphs.keys())
    data.iloc[:, -2:] = embeddings
    return data


########################################################################
# 5. Exploratory Data Analysis
########################################################################

########################################################################
# 5.1. Research Group's Analysis
########################################################################

########################################################################
# 5.1.1. Evaluation of Reseach Group by Metrics' Temporal Analysis
########################################################################

# Getting the data.
df = df_metrics_iso[df_metrics_iso.metric_name.isin(
    ["degree", "betweenness", "eigenvector", "closeness"])].copy()

# Plotting the data.
plot_multiple_line_chart_sns(df, "year", "metric_value", "metric_name", None,
                             "Year", "Metric Value", 0.25, True, True, "Metric",
                             ["Degree", "Betweenness", "Eigenvector", "Closeness"],
                             name_fig="data_analysis_pipeline/data/output/01_centrality_metrics_iso.png")

# Getting the data.
df = df_metrics_agg[df_metrics_agg.metric_name.isin(
    ["degree", "betweenness", "eigenvector", "closeness"])].copy()

# Plotting the data.
plot_multiple_line_chart_sns(df, "year", "metric_value", "metric_name", None,
                             "Year", "Metric Value", 0.25, True, False,
                             name_fig="data_analysis_pipeline/data/output/02_centrality_metrics_agg.png")

# Getting the data.
df = df_metrics_4_window[df_metrics_4_window.metric_name.isin(
    ["degree", "betweenness", "eigenvector", "closeness"])].copy()

# Plotting the data.
plot_multiple_line_chart_sns(df, "year", "metric_value", "metric_name", None,
                             "Year", "Metric Value", 0.25, True, False,
                             name_fig="data_analysis_pipeline/data/output/03_centrality_metrics_4_window.png")

########################################################################
# 5.1.2. Cohesion Analysis by Connected Components (CC) Analysis
########################################################################

# Function to generate the histogram data.
def get_data_hist_cc(networks, only_members=True):
    records = []
    for year, G in networks.items():
        if only_members:
            members = [id for id, node in G.nodes(data=True) if node["is_permanent"]]
            nodes = [n for m in members for n in G.neighbors(m)]
            nodes = list(set([*members, *nodes]))
            G = nx.subgraph(G, nodes)
        for idx, cc in enumerate(nx.connected_components(G)):
            record = {}
            record["year"] = year
            record["id_cc"] = idx + 1
            record["number_nodes_cc"] = len(cc)
            records.append(record)
    return pd.DataFrame(records)

########################################################################
# 5.1.2.1. Distribution of Number of Nodes in CC (Isolated Scenario)
########################################################################

# Getting the data.
df = get_data_hist_cc(net_isolated)
df = df[["year", "number_nodes_cc"]]
data = df.year.value_counts().to_frame("isolated")
df = df[df.number_nodes_cc < 130]

# Plotting the data.
plot_overlapping_kdes(df, "number_nodes_cc", "year",
                      "Nº. of Nodes in Connected Components", 150,
                      "data_analysis_pipeline/data/output/04_cc_analysis_iso.png")

# Restoring the config of Matplotlib.
set_config_plt()

########################################################################
# 5.1.2.2. Time Series of Number of Connected Components
########################################################################

# Getting and listing the data.
data["cumulative"] = get_data_hist_cc(net_cumulative)[["year", "number_nodes_cc"]].year.value_counts()
data["window_4"] = get_data_hist_cc(net_4_window)[["year", "number_nodes_cc"]].year.value_counts()
data = data.reset_index().rename(columns={"index": "year"}).sort_values("year").reset_index(drop=True)
print(data)

# Plotting the data.
plot_multiple_line_chart(data, data.columns[0], data.columns[1:].tolist(),
                         ["Isolated", "Cumulative", "4-window"], None,
                         "Year", "Nº. of CC", None, y_is_int=True, external_legend=False,
                         name_fig="data_analysis_pipeline/data/output/05_time_series_num_cc.png")

########################################################################
# 5.1.3. Clustering Analysis of Research Group's Temporal Nets
########################################################################

########################################################################
# 5.1.3.1. Isolated Scenario
########################################################################

# Getting the data.
df = df_metrics_iso[["year", "total_num_paper"]].groupby("year").first().reset_index()
data = get_graph_embedding(dict(net_isolated))
data["num_paper"] = df["total_num_paper"].astype(np.int64)
data.year = data.year.apply(lambda x: str(int(x)))

# Trying to discovery the optimal k.
sse = elbow_method(data.iloc[:, 1:3].copy(), 10)
plot_single_line_chart(sse.index, sse.values, "Nº. of Clusters", "W(C)", None,
                       "data_analysis_pipeline/data/output/06_elbow_k_optimal_iso.png",
                       4, "best k-clusters")

# Clustering the data.
data["cluster"] = clustering_kmeans(data.iloc[:, 1:3].copy(), 4)

# Plotting the clustering result.
plot_clusters(data, "column_1", "column_2", "cluster", "num_paper", "year",
              "Embedding Column 1", "Embedding Column 2", None,
              name_fig="data_analysis_pipeline/data/output/07_clustering_embeddings_iso.png")

########################################################################
# 5.1.3.2. 4-Window Cumulative Scenario
########################################################################

# Getting the data.
df = df_metrics_4_window[["year", "total_num_paper"]].groupby("year").first().reset_index()
data = get_graph_embedding(dict(net_4_window))
data["num_paper"] = df["total_num_paper"].astype(np.int64).values
data.year = data.year.apply(lambda x: str(int(x)))

# Trying to discovery the optimal k.
sse = elbow_method(data.iloc[:, 1:3].copy(), 10)
plot_single_line_chart(sse.index, sse.values, "Nº. of Clusters", "W(C)", None,
                       "data_analysis_pipeline/data/output/08_elbow_k_optimal_window_emb.png",
                       4, "best k-clusters")

# Clustering the data.
data["cluster"] = clustering_kmeans(data.iloc[:, 1:3].copy(), 4)

# Plotting the clustering result.
plot_clusters(data, "column_1", "column_2", "cluster", "num_paper", "year",
              "Embedding Column 1", "Embedding Column 2", None, space_legend=3.5,
              name_fig="data_analysis_pipeline/data/output/09_clustering_embeddings_window.png")

########################################################################
# 5.2. Members' Analysis
########################################################################

########################################################################
# 5.2.1. Clustering Analysis of Members for Discovery of Research Lines
########################################################################

# Getting the number of members per research line.
num_members_per_rl = df_members.research_line.value_counts()
num_members_per_rl.index = [discretize_research_line(idx) for idx in num_members_per_rl.index]

########################################################################
# 5.2.1.1. CNA Metrics
########################################################################

# Function to generate the data for specific year.
def get_data_researcher(dataframe, year):
    columns = ["metric_name", "researcher", "metric_value"]
    columns_agg = ["mean", "min", "max", "std"]
    metrics = ["betweenness", "closeness", "eigenvector", "degree",
               "eccentricity_scc", "clustering", "num_cliques"]
    data = dataframe.loc[(dataframe.year == year) & (dataframe.metric_name == "degree"),
                         ["researcher", "num_paper"]].reset_index(drop=True).sort_values("researcher")
    df_temp = dataframe[columns].groupby(["metric_name", "researcher"]).agg(columns_agg).reset_index()
    df_temp.columns = ["metric_name", "researcher", "mean", "min", "max", "std"]
    for member in data.researcher.values:
        for m in metrics:
            data.loc[data.researcher == member, ["{}_{}".format(m, c) for c in columns_agg]] = df_temp.loc[
                (df_temp.metric_name == m) & (df_temp.researcher == member), columns_agg].values \
                    if not df_temp.loc[(df_temp.metric_name == m) & (df_temp.researcher == member),
                        columns_agg].empty else [None] * len(columns_agg)
    data.fillna(0, inplace=True)
    data["id"] = data.researcher.apply(lambda x: df_members.loc[df_members.complete_name == x, "identifiers"].iloc[0][0])
    data = data[[*data.columns.tolist()[-1:], *data.columns.tolist()[:-1]]]
    data = data.reset_index().rename(columns={"index": "id_alt"})
    data["id_alt"] += 1
    return data


# Function to cluster the researchers.
def clustering_researchers(dataframe, year):
    # Gettind and preprocessing the data.
    df = get_data_researcher(dataframe, year)

    # Clustering the data.
    df["cluster"] = clustering_kmeans(df.iloc[:, 4:].copy(), 3)
    df["cluster"] += 1
    df["cluster"] = df["cluster"].astype("category")
    df["real_rl"] = df.researcher.apply(lambda x: discretize_research_line(
        df_members.research_line[df_members.complete_name == x].iloc[0]))
    df["pred_rl"] = set_research_line_from_cluster(df[["num_paper", "cluster", "real_rl"]].copy(),
                                                   num_members_per_rl)

    # Showing the classification result.
    plot_confusion_matrix(df["real_rl"], df["pred_rl"],
                          "data_analysis_pipeline/data/output/conf_matrix_metrics_{}.png".format(year))
    print(classification_report(df["real_rl"], df["pred_rl"]))

    # Generating the net for visualization of nodes and saving the data.
    generate_net_cluster(net_cumulative[2022],
        df[["id", "id_alt", "num_paper", "cluster", "pred_rl"]].copy(),
        name_file="data_analysis_pipeline/data/output/cluster_metrics_{}.gexf".format(year))
    df.to_csv("data_analysis_pipeline/data/output/cluster_metrics_{}.csv".format(year),
              quoting=csv.QUOTE_ALL, index=False)


# Clustering the members.
dataframes = [df_members_4_window, df_members_2_window]
for year, idx in {2012: 0, 2016: 0, 2020: 0, 2022: 1}.items():
    clustering_researchers(dataframes[idx].copy(), year)


########################################################################
# 5.2.1.2. Node Embeddings
########################################################################

# Function to generate the data for specific year.
def get_data_researcher(network, dataframe, year, dimensions=10):
    data = get_vector_from_graph(network, dimensions)
    data = data.loc[data.is_permanent == True].reset_index(drop=True)
    data = data.drop(columns=["h_index", "is_permanent", "research_line"])
    data = data.sort_values("complete_name").reset_index(drop=True)
    data = data.reset_index().rename(columns={"index": "id_alt", "id_scopus": "id"})
    data["id_alt"] += 1
    data.loc[:, "num_paper"] = data.complete_name.apply(
        lambda x: dataframe.loc[
            (dataframe.year == year) & (dataframe.researcher == x), "num_paper"].iloc[0])
    data = data[[*data.columns.tolist()[:3], *data.columns.tolist()[-1:], *data.columns.tolist()[3:-1]]]
    return data


# Function to cluster the researchers.
def clustering_researchers(network, dataframe, year):
    # Gettind and preprocessing the data.
    data = get_data_researcher(network, dataframe, year)

    # Clustering the data.
    data.loc[:, "cluster"] = clustering_kmeans(data[data.columns.tolist()[4:]].copy(), 3)
    data["cluster"] += 1
    data["cluster"] = data["cluster"].astype("category")
    data["real_rl"] = data.complete_name.apply(lambda x: discretize_research_line(
        df_members.research_line[df_members.complete_name == x].iloc[0]))
    data["pred_rl"] = set_research_line_from_cluster(data[["num_paper", "cluster", "real_rl"]].copy(),
                                                     num_members_per_rl)

    # Showing the classification result.
    plot_confusion_matrix(data["real_rl"], data["pred_rl"],
                          "data_analysis_pipeline/data/output/conf_matrix_emb_{}.png".format(year))
    print(classification_report(data["real_rl"], data["pred_rl"]))

    # Generating the net for visualization of nodes and saving the data.
    generate_net_cluster(net_cumulative[2022],
        data[["id", "id_alt", "num_paper", "cluster", "pred_rl"]].copy(),
        name_file="data_analysis_pipeline/data/output/cluster_emb_{}.gexf".format(year))
    data.to_csv("data_analysis_pipeline/data/output/cluster_emb_{}.csv".format(year),
                quoting=csv.QUOTE_ALL, index=False)


# Clustering the members.
nets = [net_4_window, net_2_window]
dataframes = [df_members_4_window, df_members_2_window]
for year, idx in {2012: 0, 2016: 0, 2020: 0, 2022: 1}.items():
    clustering_researchers(nets[idx][year], dataframes[idx].copy(), year)