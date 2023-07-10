# Identification of 3D Burrs on Irregular-Shaped Workpieces with Novel Computational Intelligence based on Laser Profiler Data 

## Motivation 

Nowadays, manual inspection of cutting part burrs is a time-consuming process. Furthermore, the human inspection method is subjective, as it is influenced by individual standards and techniques, thereby affecting the results. Moreover, existing robotic deburring systems treat all burrs equally regardless of their size, leading to inefficient processing. 

In contrast to previous studies that primarily focused on defect positions and classifications in two-dimensional space, this research proposes a novel computational intelligence algorithm to address the identification of burrs in three-dimensional space, particularly on irregularly shaped surfaces.The findings of this study indicate that the proposed method can be utilized to determine the sizes and relative positions of the burrs, thereby guiding the robotic manipulator to execute different processing strategies more effectively.


## Getting Started 

- [second_contour.ipynb](https://github.com/IanChen0718/burr-identification/blob/main/second_contour.ipynb). This notebook presents the step-by-step process of burr identification.

- There are two irregular contours for identification purposes, respectively [first_contour.py](https://github.com/IanChen0718/burr-identification/blob/main/first_contour.py) and [second_contour.py](https://github.com/IanChen0718/burr-identification/blob/main/second_contour.py).

## Local Area Attention

First of all, the plane where the burrs are located, also known as the processing plane, is referred to as the **_hyperplane_**. To optimize the analysis, the hyperplane is utilized to extract the area of the workpiece surface.

## Contour Extraction

Contour extraction in a point cloud often involves evaluating the degree of planarity, which is a commonly employed method. However, it is worth noting that there are contours that do not contain any burrs.

To address this issue, the hyperplane and Gaussian Mixture Model \(GMM\) are introduced as a means to remove such contours.

![GMM](/images/gmm.png)

## Feature Extraction
### Dimension reduction with hyperplane
The process of feature extraction involves the planar projection of the point cloud using a hyperplane. This projection allows the contours in the three-dimensional space to be represented using new two-dimensional coordinates.

## Burr Identification 
### Segmentation wih contours

A novel algorithm has been developed to differentiate whether the point cloud represents a burr or not. In this algorithm, the contour of the target point cloud is considered as a one-dimensional manifold embedded in a two-dimensional space. Consequently, the linear segmentation can be achieved by unfolding the contour of the target point cloud.

![Segmentation](/images/segmentation.png)

## Burr Size Measurement

The burr size is defined as the vertical distance between the source and the target. However, since the function of the target is unknown, the vertical distance cannot be directly obtained. Therefore, a cost function has been developed to calculate the burr size. 

**Cost function**
$$\Phi = distance \left( t_j^{i}, s_{i} \right) \cdot f^{'} \left( t_j^{i} \right)$$

$$ argmin \left( \Phi \left( t_j^{i} \right) \right), ~ subject ~ to \left \| f^{'}\left( t_j^{i}\right) \right \| = 1 $$

where the source $S = \\{ s_i \mid 1 \leq i \leq m \\}$, the target $T^{i} = \\{ t_j^{i} \mid 1 \leq j \leq k \\}$ in the neighbourhood of $s_i$
















