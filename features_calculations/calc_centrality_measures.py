import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import networkx as nx


def prepare_features_inter(otu):
    """
    Fix for intersection
    :param otu: Pandas dataframe with log abundances
    :return: Fixed otu and fixed tag
    """
    dict_dfs = dict()
    for p,df in otu.groupby("person"):
        dict_dfs[p] = df
    list_df_of_tags = list()
    for person in dict_dfs.keys():
        df_of_tags = pd.DataFrame(index=[f"{person}-{t}" for t in (dict_dfs[person]["time"])], columns=otu.columns[:-3])
        del dict_dfs[person]["person"]
        dict_dfs[person].index = dict_dfs[person]["time"]
        dict_dfs[person] = dict_dfs[person].sort_index()
        del dict_dfs[person]["time"]
        try:
            for row in range(len(dict_dfs[person].index) - 1):
                tag = dict_dfs[person].iloc[row + 1] - dict_dfs[person].iloc[row]
                df_of_tags.loc[f"{person}-{list(dict_dfs[person].index)[row]}"] = tag
            df_of_tags = df_of_tags.dropna()
            list_df_of_tags.append(df_of_tags)
        except:
            pass

    all_tags = pd.concat(list_df_of_tags)
    all_tags = all_tags.dropna()
    otu = otu.loc[all_tags.index]
    del otu["time"]
    del otu["person"]
    return all_tags,otu


def prepare_interaction_network_input_output(TAX,NAME,NO_LOG):
    """
    Prepare the input  and output for the Lasso prediction of differences in next time step
    :param TAX: taxonomy level to work at (int)
    :param NAME: Dataset name (str)
    :param NO_LOG: Boollian whether we log-normalize the data
    :return:
    """
    if TAX == 7 or TAX ==8:
        if TAX == 7:
            otu = pd.read_csv(f"{NAME}/tax{TAX}/tax7_log_subpca.csv",index_col=0)
        else:
            otu = pd.read_csv(f"{NAME}/tax8/tax8_log_sub PCA.csv",index_col=0)
        if NO_LOG:
            otu =10**(otu.astype(float))-0.1
        tag = pd.read_csv(f"{NAME}/tax7/tag.csv",index_col=0)
        otu["person"] = [i.split("_")[0] for i in tag["Sample Name"]]
        otu["time"] = [i.split("_")[-1] for i in tag["Sample Name"]]
        # Create a dictionary to map the values - specific to PRJEB6456
        mapping = {'B': 0, 'M': 1, '4M': 4, '12M': 12}

        # Replace the values in the "time" column
        otu['time'] = otu['time'].replace(mapping)
        tag["name"] = [str(otu["person"].loc[i])+ "-" +str(otu["time"].loc[i]) for i in otu.index]
        otu.index=tag["name"]
        tag_for_learn, otu_for_learn = prepare_features_inter(otu)
        return otu_for_learn,tag_for_learn

def make_network_coef(otu_for_learn,tag_for_learn):
    """
    Predicting the difference of each taxon according to the log abundances of the previous time step
    :param otu_for_learn: Pandas dataframe with all the log abundances
    :param tag_for_learn: Pandas dataframe with all the differences to predict
    :return:
    """
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(otu_for_learn, tag_for_learn, test_size=0.2)
    # dict_spearman = dict()
    coef = pd.DataFrame(index=y_test.columns, columns=y_test.columns)
    for bact in tqdm(y_train.columns, desc="Processing bacteria"):
        clf = linear_model.Lasso(alpha=0.1)
        clf.fit(x_train, y_train[bact])
        coef.loc[bact] = clf.coef_[:-1]
    return coef

def calc_centrality_measures(coef):
    """
    Calculating the centrality measures of - in degree, out degree, betweenness and closeness
    :param coef: Pandas dataframe of the predicted Lasso coefficients
    :return: Pandas dataframe of the centrality measures of each taxon
    """
    missing = list(coef.index.difference(coef.columns))
    missing_df = pd.DataFrame(data=np.zeros((len(coef.index), len(missing))), index=coef.index, columns=missing)
    coef = pd.concat([coef, missing_df], axis=1)
    coef = coef.sort_index()
    coef = coef.T
    coef = coef.sort_index()
    coef = coef.T
    coef = abs(coef)
    p = np.percentile(coef.to_numpy().flatten(), 0.99)
    coef = coef > p
    A = coef.to_numpy()
    G = nx.from_numpy_matrix(A, parallel_edges=True, create_using=nx.DiGraph())
    # CALCULATE CENTRALITY MEASURES
    all_centrality_measures = pd.DataFrame(index=list(range(len(coef.index))),
                                           columns=["In-degree", "Out-degree", "Betweeness", "Closeness"])
    list_in = list()
    list_out = list()
    for node in G.nodes:
        list_in.append(G.in_degree(node))
        list_out.append(G.out_degree(node))

    all_centrality_measures["Betweeness"] = pd.Series(nx.betweenness_centrality(G))
    all_centrality_measures["Closeness"] = pd.Series(nx.closeness_centrality(G))
    all_centrality_measures["In-degree"] = list_in
    all_centrality_measures["Out-degree"] = list_out
    all_centrality_measures.index = coef.index
    return all_centrality_measures



if __name__ == "__main__":
    # An example of building interaction network for the PRJEB6456 study
    TAX =8
    NO_LOG = False
    NAME = 'PRJEB6456'
    otu_for_learn,tag_for_learn = prepare_interaction_network_input_output(TAX,NAME,NO_LOG)
    lasso_coefs = make_network_coef(otu_for_learn,tag_for_learn)
    centrality_features = calc_centrality_measures(lasso_coefs)


