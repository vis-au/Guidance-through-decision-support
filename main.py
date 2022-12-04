# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import csv
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd



from sklearn import cluster, datasets, mixture, datasets
from sklearn.neighbors import kneighbors_graph

from sklearn.utils import shuffle
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


from bokeh.plotting import figure, show
from bokeh.transform import factor_cmap, factor_mark
from bokeh.palettes import Spectral6, Category20_20
from bokeh.models import ColumnDataSource, Select, Slider
from bokeh.io import output_notebook
from bokeh.layouts import column, row
from bokeh.io import curdoc
from bokeh.models import CustomJS, RadioButtonGroup
from bokeh.models import ColumnDataSource, TableColumn, DataTable, NumberFormatter, HTMLTemplateFormatter, SingleIntervalTicker
from bokeh.transform import linear_cmap 
from bokeh.palettes import Spectral6
from bokeh.palettes import Viridis256

#from parallel_plot import parallel_plot
#from parallel_plot import update_parallel_plot


# %%
#Define all the parameters that can be changed 
dr_al = "PCA"
cluster_al = "ac"
scaler = "standard"
order = "cluster"

#parameters for k-means
k_cluster_init = "k-means++" 
k_cluster_num = 3
k_cluster_n_init = 20

#parameters for meanshift
ms_bandwidth = None

#parameters for Affinity Propagation 
ap_damping = 0.6

#parameters for Spectral Clustering
s_cluster_num = 3

#parameters for AgglomerativeClustering
ac_cluster_num = 3
ac_linkage = "ward"

#parameters for PCA
component_num = 2
if_whiten = True
svd_solver = "full"
pca_tol = 0.0
pca_iterated_power = "auto"
pca_random_state = 0

# only used clustering algorithms with the parameter of number
clustering_algorithms= [
    'MiniBatchKMeans',
#    'AffinityPropagation',
#    'MeanShift',
    'SpectralClustering',
    'Ward',
    'AgglomerativeClustering',
#    'DBSCAN',
    'Birch'
]

datasets_names = [
    'Noisy Circles',
    'Noisy Moons',
    'Blobs',
    'No Structure'
]

dr_algorithms=[
    'PCA'
]

# %%
# Define all the functions of processing the data - scaling, cluster and dimension reduction


def get_dataset(dataset, n_samples):
    if dataset == 'Noisy Circles':
        return datasets.make_circles(n_samples=n_samples,
                                    factor=0.5,
                                    noise=0.05)

    elif dataset == 'Noisy Moons':
        return datasets.make_moons(n_samples=n_samples,
                                   noise=0.05)

    elif dataset == 'Blobs':
        return datasets.make_blobs(n_samples=n_samples,
                                   centers = 6,
                                   random_state=8)

    elif dataset == "No Structure":
        return np.random.rand(n_samples, 2), None

def dimension_reduction(X, dr_al):

    if dr_al == "PCA":
        if svd_solver == "arpack":
            model = PCA(n_components=component_num, whiten=if_whiten, svd_solver=svd_solver, tol=pca_tol, random_state=pca_random_state)
        elif svd_solver == "randomized":
            model = PCA(n_components=component_num, whiten=if_whiten, svd_solver=svd_solver, iterated_power=pca_iterated_power, random_state=pca_random_state)
        else:
            model = PCA(n_components=component_num, whiten=if_whiten, svd_solver=svd_solver)
    
        X = model.fit_transform(X)
        return X

# define some helper functions
def clustering(X, algorithm, n_clusters):
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)




    # Generate the new colors:
    if algorithm=='MiniBatchKMeans':
        model = cluster.MiniBatchKMeans(n_clusters=n_clusters)

    elif algorithm=='Birch':
        model = cluster.Birch(n_clusters=n_clusters)

    elif algorithm=='DBSCAN':
        model = cluster.DBSCAN(eps=.2)

    elif algorithm=='AffinityPropagation':
        model = cluster.AffinityPropagation(damping=.9,
                                            preference=-200)

    elif algorithm=='MeanShift':
        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)
        model = cluster.MeanShift(bandwidth=bandwidth,
                                  bin_seeding=True)

    elif algorithm=='SpectralClustering':
        model = cluster.SpectralClustering(n_clusters=n_clusters,
                                           eigen_solver='arpack',
                                           affinity="nearest_neighbors")

    elif algorithm=='Ward':
        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)
        model = cluster.AgglomerativeClustering(n_clusters=n_clusters,
                                                linkage='ward',
                                                connectivity=connectivity)

    elif algorithm=='AgglomerativeClustering':
        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)
        model = cluster.AgglomerativeClustering(linkage="average",
                                                n_clusters=n_clusters,
                                                connectivity=connectivity)

    model.fit(X)

    if hasattr(model, 'labels_'):
            y_pred = model.labels_.astype(int)
    else:
            y_pred = model.predict(X)

    return X, y_pred


# def choose_scaler(scaler):
#     if scaler == "standard":
#         return StandardScaler()

# def choose_cluster_al(cluster_al):
#     if cluster_al == "kmeans":
#         return cluster.KMeans(init=k_cluster_init, n_clusters=k_cluster_num, n_init=k_cluster_n_init)
#     elif cluster_al == "mean_shift":
#         return cluster.MeanShift(bandwidth=ms_bandwidth)
#     elif cluster_al == "ap": 
#         return cluster.AffinityPropagation(damping=ap_damping)
#     elif cluster_al == "spectral":
#         return cluster.SpectralClustering(n_clusters=s_cluster_num)
#     elif cluster_al == "ac":
#         return cluster.AgglomerativeClustering(n_clusters=ac_cluster_num, linkage=ac_linkage)

# def choose_dr_al(dr_al):
#     if dr_al == "PCA":
#         if svd_solver == "arpack":
#             return PCA(n_components=component_num, whiten=if_whiten, svd_solver=svd_solver, tol=pca_tol, random_state=pca_random_state)
#         elif svd_solver == "randomized":
#             return PCA(n_components=component_num, whiten=if_whiten, svd_solver=svd_solver, iterated_power=pca_iterated_power, random_state=pca_random_state)
#         else:
#             return PCA(n_components=component_num, whiten=if_whiten, svd_solver=svd_solver)
# #Change the input choices name to the algorithms through the functions 
# scaler = choose_scaler(scaler)
# cluster_al = choose_cluster_al(cluster_al)
# dr_al = choose_dr_al(dr_al)


# set up initial data 
n_samples = 2000
n_clusters = 2
algorithm = 'MiniBatchKMeans'
dataset = 'Blobs'
dr_algorithm = "PCA"


X, y = get_dataset(dataset, n_samples)
#X = dimension_reduction(X, dr_algorithm)
start_time = time.time()
X, y_pred = clustering(X, algorithm, n_clusters)
cluster_time = time.time() - start_time
spectral = np.hstack([Category20_20])
colors = [spectral[i] for i in y_pred]


n_matrix = []

al_matrix = []


alt = []
alt_source = []
for al in clustering_algorithms: 
    start_time = time.time()
    X, y_pred = clustering(X, al, n_clusters)
    cluster_time = time.time() - start_time
    
    labels_true = y
    labels_pred = y_pred 
    
    r_score = metrics.adjusted_rand_score(labels_true, labels_pred)
    s_score = metrics.silhouette_score(X, labels_pred, metric='euclidean')
    ch_score = metrics.calinski_harabasz_score(X, labels_pred)
    db_score = metrics.davies_bouldin_score(X, labels_pred)
    time_score = cluster_time 
    
    colors = [spectral[i] for i in y_pred]
    alt_source.append(ColumnDataSource(data=dict(x=X[:, 0], y=X[:, 1], colors=colors)))
    alt.append(figure(title=al))
    alt[-1].circle('x', 'y', fill_color='colors', line_color=None, source=alt_source[-1])
    al_matrix.append([r_score, s_score, ch_score, db_score, time_score])


import skcriteria as skc
import matplotlib.pyplot as plt
from skcriteria.madm import electre
from skcriteria.pipeline import mkpipe
from skcriteria.madm import simple
from skcriteria.madm import electre
from skcriteria.preprocessing import invert_objectives, scalers

al_name = clustering_algorithms
n_name = ['2','3','4','5','6','7','8','9','10']
c_name = ["Pre", "S", "CH", "DB", "Time"]

objectives = [max, max, max, min, min]
al_weight = [5, 1, 1, 1, 2]
n_weight = [3, 2, 2, 2, 1]


al_dm = skc.mkdm(
    al_matrix,
    objectives,
    weights=al_weight,
    alternatives=al_name,
    criteria= c_name,
)


inverter = invert_objectives.InvertMinimize()
al_dmt = inverter.transform(al_dm)
scaler = scalers.SumScaler(target="both")
al_dmt = scaler.transform(al_dmt)
dec = simple.WeightedSumModel()
al_rank = dec.evaluate(al_dmt)
#al_dmt.plot()
rank = al_rank.rank_
df_aldm = al_dm.to_dataframe()
df_aldm = df_aldm.drop(["objectives","weights"])
df_aldm.insert(0, "rank", rank)
source_aldm = ColumnDataSource(df_aldm)

r_alt = [None] * 5
flag = 0
for n in list(df_aldm['rank']):
    r_alt[n-1] = alt[flag]
    flag = flag + 1
    
TOOLS = "pan,wheel_zoom,box_select,tap,poly_select,lasso_select,reset" 

for p in range(5):
    if p == 0 :
        r_alt[p].width = 600
        r_alt[p].height = 600
        # r_alt[p].active_inspect = hover_tool
        # r_alt[p].active_drag = pan
        # r_alt[p].active_drag = zoom
    else: 
        r_alt[p].width = 300
        r_alt[p].height = 300
        r_alt[p].toolbar_location=None
        r_alt[p].x_range = r_alt[0].x_range
        r_alt[p].y_range = r_alt[0].y_range
    

template="""
            <div style="background:<%= 
                (function colorfromint(){
                    if(rank == 1){
                        return("red")}
                    }()) %>; 
            color: <%= 
                (function colorfromint(){
                    if(rank == 1){
                        return("white")}
                    }()) %>;"> 
            <%= value %>
            </div>
            """
formatter =  HTMLTemplateFormatter(template=template)


columns = [
        TableColumn(field="index", title="Algorithms", formatter = formatter),
        TableColumn(field="rank", title="Ranking"),
        TableColumn(field="Pre", title="Preference", formatter=NumberFormatter(format="0.000")),
        TableColumn(field="S", title="Silhouette", formatter=NumberFormatter(format="0.000")),
        TableColumn(field="CH", title="Calinski-Harabasz", formatter=NumberFormatter(format="0.000")),
        TableColumn(field="DB", title="Davies-Bouldin", formatter=NumberFormatter(format="0.000")),
        TableColumn(field="Time", title="Time", formatter=NumberFormatter(format="0.000")),
    ]

myTable = DataTable(source=source_aldm, columns=columns, width=400, height=175, index_position=None)

algorithm = df_aldm.loc[df_aldm["rank"] == 1].index.values[0]

for k in range(2, 11):
    start_time = time.time()
    X, y_pred = clustering(X, algorithm, k)
    cluster_time = time.time() - start_time
    
    labels_true = y
    labels_pred = y_pred 
    r_score = metrics.adjusted_rand_score(labels_true, labels_pred)
    s_score = metrics.silhouette_score(X, labels_pred, metric='euclidean')
    ch_score = metrics.calinski_harabasz_score(X, labels_pred)
    db_score = metrics.davies_bouldin_score(X, labels_pred)
    time_score = cluster_time 
    n_matrix.append([r_score, s_score, ch_score, db_score, time_score])



n_dm = skc.mkdm(
    n_matrix,
    objectives,
    weights=n_weight,
    alternatives=n_name,
    criteria= c_name, 
)

inverter = invert_objectives.InvertMinimize()
n_dmt = inverter.transform(n_dm)
scaler = scalers.SumScaler(target="both")
n_dmt = scaler.transform(n_dmt)
dec = simple.WeightedSumModel()
n_rank = dec.evaluate(n_dmt)
#n_dm.to_dataframe()
n_rank.rank_
n_x = [2,3,4,5,6,7,8,9,10]
n_y = n_rank.e_.score
n_data = {
    "n_x":n_x,
    "n_y":n_y
}
n_source = ColumnDataSource(n_data)
mapper = linear_cmap(field_name='n_y', palette=Spectral6 ,low=min(n_y) ,high=max(n_y))
n_plot = figure(toolbar_location=None, width=400, height=300, title="Number of Clusters Evaluation", x_axis_label='Number of Clusters', y_axis_label='Evaluation')
n_plot.line("n_x", "n_y", line_color='rgba(108, 122, 137, 0.8)', line_width=1, source = n_source)
n_plot.circle("n_x", "n_y", line_color='rgba(108, 122, 137, 0.8)', color=mapper, fill_alpha=1, size=8, source = n_source)
n_plot.xaxis.ticker = SingleIntervalTicker(interval=1)
# pipe_vector = mkpipe(
#     invert_objectives.InvertMinimize(),
#     scalers.VectorScaler(target="matrix"),  # this scaler transform the matrix
#     scalers.SumScaler(target="weights"),  # and this transform the weights
#     electre.ELECTRE2(p0=0.65, p1=0.5, p2=0.35, q0=0.65, q1=0.35),
# )

# kernel_al = pipe_vector.evaluate(al_dm)
# kernal_n = pipe_vector.evaluate(n_dm) 
# al_dm.plot()
# kernel_al.plot()

#rank
#rank.rank_
#rank.alternatives
#rank.e_.score


#%%

# set up plot (styling in theme.yaml)


# plot = figure(tools=TOOLS, width=600, height=600, title=algorithm)
# source = ColumnDataSource(data=dict(x=X[:, 0], y=X[:, 1], colors=colors))
# plot.circle('x', 'y', fill_color='colors', line_color=None, source=source)

# the "saved" graph - will change it to other graphs that help with directing
# df = pd.DataFrame(data= X, columns=['x1', 'x2'])
#saved = parallel_plot(df=df, color=df[df.columns[0]], palette=Viridis256)

#figure(tools=TOOLS, width=400, height=400, title="Criteria Comparisons")
#saved.circle('x', 'y', fill_color='colors', line_color=None, source=source)

# a1 = figure(toolbar_location=None, x_range=plot.x_range, y_range=plot.y_range, width=300, height=300, title="Alternative 1 View")
# a1.circle('x', 'y', fill_color='colors', line_color=None, source=source)

# a2 = figure(toolbar_location=None, x_range=plot.x_range, y_range=plot.y_range, width=300, height=300, title="Alternative 2 View")
# a2.circle('x', 'y', fill_color='colors', line_color=None, source=source)

# a3 = figure(toolbar_location=None, x_range=plot.x_range, y_range=plot.y_range, width=300, height=300, title="Alternative 3 View")
# a3.circle('x', 'y', fill_color='colors', line_color=None, source=source)

# a4 = figure(toolbar_location=None, x_range=plot.x_range, y_range=plot.y_range, width=300, height=300, title="Alternative 3 View")
# a4.circle('x', 'y', fill_color='colors', line_color=None, source=source)
# set up widgets


#ORDER_OPTIONS =["Dimension Reduction First", "Clustering First"]
#order_button = RadioButtonGroup(labels=ORDER_OPTIONS, active=0, width=400)
#order_button.js_on_click(CustomJS(code="""
#    console.log('order_button: active=' + this.active, this.toString())
#"""))

# dr_algorithm_select = Select(value='PCA',
#                           title='Dimension reduction algorithm:',
#                           width=200,
#                           options=dr_algorithms)

algorithm_select = Select(value=algorithm,
                          title='Clustering algorithm:',
                          width=200,
                          options=clustering_algorithms)


dataset_select = Select(value='Noisy Circles',
                        title='Dataset:',
                        width=200,
                        options=datasets_names)

samples_slider = Slider(title="Number of samples",
                        value=2000.0,
                        start=1000.0,
                        end=10000.0,
                        step=100,
                        width=200
                        )

clusters_slider = Slider(title="Number of clusters",
                         value=2.0,
                         start=2.0,
                         end=10.0,
                         step=1,
                         width=200
                         )

al_list = clustering_algorithms            
# set up callbacks
def update_algorithm(attrname, old, new):

        
    #repalce the underlying data with the algorithm list by getting the name and change the order in the algorithm name list to ensure the corresponding relatationship
    algorithm = algorithm_select.value
    n_clusters = int(clusters_slider.value)

    top_al = r_alt[0].title.text
    top_index = al_list.index(top_al)
    rep_index = al_list.index(algorithm)

    data_save = dict(alt_source[top_index].data)
    alt_source[top_index].data = dict(alt_source[rep_index].data)
    alt_source[rep_index].data = data_save 
    
    al_list[top_index] = algorithm
    al_list[rep_index] = top_al

    
    # alt_source[0].data = alt_source[al_list.index(algorithm)].data
    # alt_source[al_list.index(algorithm)].data = data_save

   
    #after the data updated, replace the title with the new selected algorithm
    new_index = 0
    for r in r_alt:
        if r.title.text == algorithm:
            break
        new_index  = new_index  + 1
    
#    new_index = df_aldm.loc[df_aldm.index ==  algorithm, "rank"][0] - 1

    if new_index != 0:
        r_alt[new_index].title.text = r_alt[0].title.text
        r_alt[0].title.text = algorithm 
        
    global X, y, n_source
    
#    dataset = dataset_select.value
    algorithm = algorithm_select.value
#    n_samples = int(samples_slider.value)
#    X, y = get_dataset(dataset, n_samples)
    
    n_matrix = []
    
    for k in range(2, 11):
        start_time = time.time()
        X, y_pred = clustering(X, algorithm, k)
        cluster_time = time.time() - start_time
        
        labels_true = y
        labels_pred = y_pred 
        r_score = metrics.adjusted_rand_score(labels_true, labels_pred)
        s_score = metrics.silhouette_score(X, labels_pred, metric='euclidean')
        ch_score = metrics.calinski_harabasz_score(X, labels_pred)
        db_score = metrics.davies_bouldin_score(X, labels_pred)
        time_score = cluster_time 
        n_matrix.append([r_score, s_score, ch_score, db_score, time_score])
        
    n_dm = skc.mkdm(
        n_matrix,
        objectives,
        weights=n_weight,
        alternatives=n_name,
        criteria= c_name, 
    )
    inverter = invert_objectives.InvertMinimize()
    n_dmt = inverter.transform(n_dm)
    scaler = scalers.SumScaler(target="both")
    n_dmt = scaler.transform(n_dmt)
    dec = simple.WeightedSumModel()
    n_rank = dec.evaluate(n_dmt)
    #n_dm.to_dataframe()
    n_rank.rank_
    n_x = [2,3,4,5,6,7,8,9,10]
    n_y = n_rank.e_.score
    n_data = {
        "n_x":n_x,
        "n_y":n_y
    }
    n_source.data = n_data

    
def update_cluster(attrname, old, new):
    dataset = dataset_select.value
    n_clusters = int(clusters_slider.value)
    n_samples = int(samples_slider.value)
    global X, y, al_matrix, alt, alt_source, r_alt, al_list, source_aldm
    
    al_list = clustering_algorithms 
    #X, y = get_dataset(dataset, n_samples)
    al_matrix = []

    al_loop = 0
    for al in al_list: 
        start_time = time.time()
        X, y_pred = clustering(X, al, n_clusters)
        cluster_time = time.time() - start_time
        
        labels_true = y
        labels_pred = y_pred 
        
        r_score = metrics.adjusted_rand_score(labels_true, labels_pred)
        s_score = metrics.silhouette_score(X, labels_pred, metric='euclidean')
        ch_score = metrics.calinski_harabasz_score(X, labels_pred)
        db_score = metrics.davies_bouldin_score(X, labels_pred)
        time_score = cluster_time 
        
        colors = [spectral[i] for i in y_pred]
        alt_source[al_loop].data = dict(x=X[:, 0], y=X[:, 1], colors=colors)
        
        al_matrix.append([r_score, s_score, ch_score, db_score, time_score])
        al_loop = al_loop + 1
    
    al_dm = skc.mkdm(
        al_matrix,
        objectives,
        weights=al_weight,
        alternatives=al_name,
        criteria= c_name,
    )
    
    inverter = invert_objectives.InvertMinimize()
    al_dmt = inverter.transform(al_dm)
    scaler = scalers.SumScaler(target="both")
    al_dmt = scaler.transform(al_dmt)
    dec = simple.WeightedSumModel()
    al_rank = dec.evaluate(al_dmt)
    #al_dmt.plot()
    rank = al_rank.rank_
    df_aldm = al_dm.to_dataframe()
    df_aldm = df_aldm.drop(["objectives","weights"])
    df_aldm.insert(0, "rank", rank)
    source_aldm.data = dict(ColumnDataSource(df_aldm).data)
    
    for n in list(df_aldm['rank']):
        r_alt[n-1].title.text = df_aldm.loc[df_aldm["rank"] == n].index.values[0]
    

    
    

def update_samples_or_dataset(attrname, old, new):
    global X, y

    dataset = dataset_select.value
    algorithm = algorithm_select.value
    n_clusters = int(clusters_slider.value)
    n_samples = int(samples_slider.value)
    
    

#     X, y = get_dataset(dataset, n_samples)
#     X, y_pred = clustering(X, algorithm, n_clusters)
#     colors = [spectral[i] for i in y_pred]

#     source.data = dict(colors=colors, x=X[:, 0], y=X[:, 1])
    
#     df = pd.DataFrame(data= X, columns=['x1', 'x2'])
# #    update_parallel_plot(p=p, df=df, color=df[df.columns[0]], palette=Viridis256)
    
#     plot.title.text = algorithm

#def update_order(active):
#    if active==0:
    
#    if active==1:

algorithm_select.on_change('value', update_algorithm)
clusters_slider.on_change('value_throttled', update_cluster)

dataset_select.on_change('value', update_algorithm)
samples_slider.on_change('value_throttled', update_samples_or_dataset)




#%%
#evaluate algoithms when number of clusters, sample or dataset change
def evaluate_algorithms():
    global al_matrix, alt, alt_source, r_alt, al_list, source_aldm
    
    al_list = clustering_algorithms 
    #X, y = get_dataset(dataset, n_samples)
    al_matrix = []

    al_loop = 0
    for al in al_list: 
        start_time = time.time()
        X, y_pred = clustering(X, al, n_clusters)
        cluster_time = time.time() - start_time
        
        labels_true = y
        labels_pred = y_pred 
        
        r_score = metrics.adjusted_rand_score(labels_true, labels_pred)
        s_score = metrics.silhouette_score(X, labels_pred, metric='euclidean')
        ch_score = metrics.calinski_harabasz_score(X, labels_pred)
        db_score = metrics.davies_bouldin_score(X, labels_pred)
        time_score = cluster_time 
        
        colors = [spectral[i] for i in y_pred]
        alt_source[al_loop].data = dict(x=X[:, 0], y=X[:, 1], colors=colors)
        
        al_matrix.append([r_score, s_score, ch_score, db_score, time_score])
        al_loop = al_loop + 1
    
    al_dm = skc.mkdm(
        al_matrix,
        objectives,
        weights=al_weight,
        alternatives=al_name,
        criteria= c_name,
    )
    
    inverter = invert_objectives.InvertMinimize()
    al_dmt = inverter.transform(al_dm)
    scaler = scalers.SumScaler(target="both")
    al_dmt = scaler.transform(al_dmt)
    dec = simple.WeightedSumModel()
    al_rank = dec.evaluate(al_dmt)
    #al_dmt.plot()
    rank = al_rank.rank_
    df_aldm = al_dm.to_dataframe()
    df_aldm = df_aldm.drop(["objectives","weights"])
    df_aldm.insert(0, "rank", rank)
    source_aldm.data = dict(ColumnDataSource(df_aldm).data)
    
    for n in list(df_aldm['rank']):
        r_alt[n-1].title.text = df_aldm.loc[df_aldm["rank"] == n].index.values[0]
    

    
#evaluate number of clusters when algorithm, sample or dataset change
def evaluate_numbers():
    global n_source
    
#    dataset = dataset_select.value
    algorithm = algorithm_select.value
#    n_samples = int(samples_slider.value)
#    X, y = get_dataset(dataset, n_samples)
    
    n_matrix = []
    
    for k in range(2, 11):
        start_time = time.time()
        X, y_pred = clustering(X, algorithm, k)
        cluster_time = time.time() - start_time
        
        labels_true = y
        labels_pred = y_pred 
        r_score = metrics.adjusted_rand_score(labels_true, labels_pred)
        s_score = metrics.silhouette_score(X, labels_pred, metric='euclidean')
        ch_score = metrics.calinski_harabasz_score(X, labels_pred)
        db_score = metrics.davies_bouldin_score(X, labels_pred)
        time_score = cluster_time 
        n_matrix.append([r_score, s_score, ch_score, db_score, time_score])
        
    n_dm = skc.mkdm(
        n_matrix,
        objectives,
        weights=n_weight,
        alternatives=n_name,
        criteria= c_name, 
    )
    inverter = invert_objectives.InvertMinimize()
    n_dmt = inverter.transform(n_dm)
    scaler = scalers.SumScaler(target="both")
    n_dmt = scaler.transform(n_dmt)
    dec = simple.WeightedSumModel()
    n_rank = dec.evaluate(n_dmt)
    #n_dm.to_dataframe()
    n_rank.rank_
    n_x = [2,3,4,5,6,7,8,9,10]
    n_y = n_rank.e_.score
    n_data = {
        "n_x":n_x,
        "n_y":n_y
    }
    n_source.data = n_data



# %%
# set up layout
#selects = row(dataset_select, algorithm_select, width=420)
#row(dataset_select, samples_slider)
inputs = column(row(algorithm_select, clusters_slider))
main_view= row(column(inputs, myTable, n_plot), r_alt[0])
# add to document
alternatives = row(r_alt[1],r_alt[2],r_alt[3],r_alt[4])
curdoc().add_root(column(main_view, alternatives))
curdoc().title = "Clustering"




# %%
