# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 09:40:33 2022

@author: IanChen
"""
import copy
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from .PdGeom import PdGeom
from .GetPlaneEq import GetPlaneEq
from .PdProject import PdProject
from .PdBatch import *
from .draw import *
def TargetProfile(df, p1, p2, p3, len_local_space):
    geometry = PdGeom(df)
    geometry.estimate(20)  
    df_plan, df_corr = geometry.planarize(0.6)

    df_corr_xyz = df_corr[["X", "Y", "Z"]]
    df_corr_xyz.insert(3, "Const", 1)
    
    a1, b1, c1, d1 = GetPlaneEq(p1, p2, p3) 
    abcd = np.asarray([a1, b1, c1, d1])
    dist_flag = 0
    df_plane_num = df_corr_xyz.dot(abcd)
    df_plane_num = df_plane_num.abs()
    
    hist, bin_edges = np.histogram(df_plane_num)

#%%
    f=np.array(df_plane_num)[:, None]
    print("f :", f)
    gmm = GaussianMixture(n_components=3,covariance_type='full')
    gmm.fit(f)
    weights = gmm.weights_
    means = gmm.means_
    covars = gmm.covariances_
    stds = np.sqrt(covars)
    results = gmm.fit_predict(np.array(df_plane_num)[:, None])

#%%
    index_max = np.argmax(means)
    idx_0 = np.where(results == index_max)[0]
    
    """The component closer to the hyperplane is what we want"""    
    index_min = np.argmin(means)
    idx_1 = np.where(results == index_min)[0]    

    index_list = [index_max, index_min]
    idx_middle = None
    for i in range(3):
        if i not in index_list:
            idx_middle = i
            break
   
    idx_rest = np.where(results == idx_middle)[0]

    df_corr_xyz_ = df_corr.iloc[idx_1] 
    df_corr_drop = df_corr.iloc[idx_0]

    # draw_pd_three_result(df_corr_xyz_, df_corr_drop, df_corr.iloc[idx_rest],
    #                      [0, 1, 0], [1, 165/255, 0], [1, 0, 0])
#%%
#     df_plane_num.name = "The distance from the point to the hyperplane (mm)"
#     sns.distplot(df_plane_num, rug=False, hist=True)
#     plt.rcParams.update({'font.family':'Times New Roman'})
#     sns.set(font="Times New Roman")
#     plt.show()

#     def plot_mixture(gmm, X, show_legend=True, ax=None):
#         if ax is None:
#             ax = plt.gca()
#             # Compute PDF of whole mixture
#         x = np.linspace(0, len_local_space, len_local_space*5)
#         logprob = gmm.score_samples(x.reshape(-1, 1))
#         pdf = np.exp(logprob)
#         # Compute PDF for each component
#         responsibilities = gmm.predict_proba(x.reshape(-1, 1))
#         pdf_individual = responsibilities * pdf[:, np.newaxis]
#         # Plot data histogram
#         ax.hist(X, 30, density=True, histtype='stepfilled', alpha=0.4, label
#         ='Data')
#         # Plot PDF of whole model
#         ax.plot(x, pdf, '-k', label='Mixture PDF')
#         # Plot PDF of each component
#         ax.plot(x, pdf_individual, '--', label='Component PDF')
#         ax.set_xlabel("The distance from the point to the hyperplane (mm)")
#         ax.set_ylabel("Density")
#         if show_legend:
#             ax.legend()
    
# #%%
#     """Plot of One Component"""
#     k_arr = np.asarray([3])
#     models = [
#     GaussianMixture(n_components=k).fit(f)
#     for k in k_arr
#     ]

#     plt.figure()
#     ax = plt.axes()

#     plot_mixture(models[0], df_plane_num, show_legend=False, ax=ax)
#     ax.set_title(f'Number of components ={gmm.n_components}')
#     plt.tight_layout()
#     plt.rcParams.update({'font.family':'Times New Roman'})
# #%%
#     """Plot of All Components"""
#     k_arr = np.arange(10) + 1
#     models = [
#     GaussianMixture(n_components=k).fit(f)
#     for k in k_arr
#     ]
#     # Show all models for n_components 1 to 9
#     _, axes = plt.subplots(3, 3, figsize=np.array([3,3])*3, dpi=100)
#     for gmm, ax in zip(models, axes.ravel()):
#         plot_mixture(gmm, df_plane_num, show_legend=False, ax=ax)
#         ax.set_title(f'Number of components ={gmm.n_components}')

#     plt.tight_layout()
#     plt.rcParams.update({'font.family':'Times New Roman'})

    return df_corr_drop, df_corr_xyz_