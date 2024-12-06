# Optimization of the Ho-Kashyap Classification Algorithm Using Appropriate Learning Samples

This repository contains the implementation of the paper **"Optimization of the Ho-Kashyap Classification Algorithm Using Appropriate Learning Samples"**. The paper introduces an optimized classification approach by combining the Ho-Kashyap algorithm with Multi-Class Instance Selection (MCIS). This README provides an overview of the paper, its methodology, datasets used, and the Python implementation.

### Link to the paper:
- https://ieeexplore.ieee.org/abstract/document/7777760
---

## Table of Contents
- [Introduction](#introduction)
- [Key Contributions](#key-contributions)
- [Datasets Used](#datasets-used)
- [Methodology](#methodology)
  - [Multi-Class Instance Selection (MCIS)](#multi-class-instance-selection-mcis)
  - [Ho-Kashyap Algorithm](#ho-kashyap-algorithm)
  - [Pipeline Overview](#pipeline-overview)
- [Implementation](#implementation)
- [Results](#results)
- [How to Use This Repository](#how-to-use-this-repository)
- [References](#references)

---

## Introduction

Classification is a fundamental task in machine learning, with applications in various domains such as medicine, agriculture, and industry. However, noisy or far-from-center data can degrade the performance of classifiers. This paper proposes an approach to mitigate these challenges by:
1. Selecting representative and boundary data points using the **Multi-Class Instance Selection (MCIS)** method.
2. Applying the **Ho-Kashyap classification algorithm** to the optimized dataset to enhance accuracy and reduce runtime.

---

## Key Contributions

The paper's main contributions include:
1. **Improved Accuracy**: The combination of MCIS and Ho-Kashyap reduces the influence of noisy and far-from-center data, improving classification accuracy.
2. **Reduced Runtime**: By focusing only on essential data points, the runtime of the classification process is significantly reduced.
3. **General Applicability**: The method is validated on multiple datasets, showcasing its robustness across domains.

---

## Datasets Used

The methodology is tested on several publicly available datasets:
1. **Pima Indians Diabetes**:
   - 8 Features, 2 Classes, 768 Samples
2. **Ionosphere**:
   - 33 Features, 2 Classes, 351 Samples
3. **Breast Cancer Wisconsin**:
   - 10 Features, 2 Classes, 699 Samples
4. **Haberman**:
   - 3 Features, 2 Classes, 306 Samples

This repository focuses on implementing the method using the **Pima Indians Diabetes dataset**.

---

## Methodology

### Multi-Class Instance Selection (MCIS)

MCIS is a preprocessing method to optimize training datasets by selecting boundary instances:
1. **Cluster Positive Instances**: Use K-Means clustering to group positive class instances.
2. **Identify Boundary Instances**: Calculate the distance between negative class instances and cluster centers of the positive class. Mark instances within a specified radius as boundary data.
3. **Iterate for All Classes**: Repeat the process for all classes to identify representative samples.

### Ho-Kashyap Algorithm

The Ho-Kashyap algorithm is a linear classification technique used to solve classification problems by iteratively adjusting a separating hyperplane. In this paper, it is combined with MCIS for improved performance.

### Pipeline Overview

1. **Preprocessing**:
   - Handle missing values and scale the dataset features.
2. **Instance Selection**:
   - Apply MCIS to filter and optimize the dataset.
3. **Classification**:
   - Use the Ho-Kashyap algorithm or another classifier (e.g., SVM, KNN) for final classification.

---

## Implementation

The implementation in this repository follows the steps outlined in the paper:
1. **Load the dataset** and preprocess it by handling missing values and scaling features.
2. **Apply MCIS** to identify boundary and representative instances.
3. **Train an SVM classifier** using the processed dataset.
4. **Evaluate performance** using metrics such as accuracy, precision, recall, and F1-score.

---
![res](https://github.com/user-attachments/assets/14df9c15-b2cf-4b8a-92ca-fea80dabf933)
---


### Please cite:
- Dezfoulian, Mir Hossein, et al. "Optimization of the Ho-Kashyap classification algorithm using appropriate learning samples." 2016 Eighth International Conference on Information and Knowledge Technology (IKT). IEEE, 2016.
- DOI: https://doi.org/10.1109/IKT.2016.7777760
