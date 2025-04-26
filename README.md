# DyStaBind：Dynamic and static characterization for predicting protein-RNA interactions with multi-scale cross attention

Codes for our paper *Dynamic and static characterization for predicting protein-RNA interactions with multi-scale cross attention*

## Overview

![StaticPrediction](.\assets\ComparisonStudy(Static).png)

RNA-binding proteins (RBPs) play indispensable roles in fundamental biological processes including RNA splicing, sequence editing and translational control. These proteins bind to RNA through a sophisticated interplay of sequence specificity, structural complementarity, and dynamic conformational adaptations, mediated by specialized binding domains such as RNA recognition motifs (RRMs) and K homology (KH) domains. The binding modalities of RBPs exhibit functional diversity, encompassing both stable binding and dynamic binding. Investigating RBPs' regulatory mechanisms and binding rules is essential for learning their biological roles. To address the challenges posed by the vast RBP repertoire (representing ~10\% of the human proteome), high-throughput experimental and computational approaches have emerged as powerful tools for binding site characterization. Although the experimental approaches offer precise binding site detection, their utility in large-scale studies is constrained by substantial time investments and prohibitive costs. 

Based on the cellular environmental conditions, protein-RNA interactions (PRIs) can be categorized into two distinct classes: static PRIs, which occur under steady-state cellular conditions without environmental perturbations, and dynamic PRIs, which arise in response to specific cellular stimuli or contextual changes. Consequently, computational methods for studying these interactions are divided into two categories. The first type of method does not take into account the changes in cellular conditions. The second type of method takes into account the dynamic changes in cellular conditions. However, these methods do not effectively combine static and dynamic features, resulting in insufficient utilization of cellular conditions and a certain deviation from physiological reality.

## Architecture

We proposes a multi-view framework that combines multi-scale CNNs with cross-attention mechanisms to unify RNA's dynamic and static characteristics for PRIs prediction. The framework characterizes RNA features through three complementary perspectives: dynamic sequence embeddings, dynamic structural representations, and static conformational attributes. These multi-view features are processed in parallel via three convolutional neural networks with distinct kernel sizes to extract multi-scale latent motifs. Cross-attention modules subsequently perform iterative pairwise interaction modeling among the three sets of CNN-processed features. The integrated representations are then fed into a pyramid-shaped context-aware classifier for the final prediction of PRIs.

![DyStaBind](.\assets\DyStaBind.png)

## Release

[2025/04/27]  We first release our code (including training and evaluation scripts).

## Contents

- [Install](https://github.com/chengqian-lu/DyStaBind#install)
- [Dataset](https://github.com/chengqian-lu/DyStaBind#Dataset)
- [Train](https://github.com/chengqian-lu/DyStaBind#train)
- [StaticPrediction](https://github.com/chengqian-lu/DyStaBind#StaticPrediction)
- [DynamicPrediction](https://github.com/chengqian-lu/DyStaBind#DynamicPrediction)

## Install

```
conda create -n DyStaBind python=3.10
conda activate DyStaBind
pip install -r requirements.txt
```

## Dataset

- TODO: coming soon

## Train

```python
python ./main.py --data_file TITA_Hela --data_path ./dataset --model_save_path ./results/model --output_dir ./results/output --BERT_model_path ./BERT_Model --z_curve --icSHAPE --BERTEmbedding --train
```

## StaticPrediction

```python
python ./main.py --data_file TITA_Hela --data_path ./dataset --model_save_path ./results/model --output_dir ./results/staticPrediction --BERT_model_path ./BERT_Model --z_curve --icSHAPE --BERTEmbedding --validate
```

## DynamicPrediction

```python
python ./main.py --data_file TITA_Hela --data_path ./dataset --model_save_path ./results/model --output_dir ./results/dynamicPrediction --BERT_model_path ./BERT_Model --z_curve --icSHAPE --BERTEmbedding --dynamic_validate
```

