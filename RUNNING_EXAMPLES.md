# Running Examples

> Datasets and test problems reused across chapters. When an existing running example
> fits, the agent must use it rather than inventing a new one (STEERING.md §6.5).

## Datasets

| Name | Source | Rows | Features | Description | Chapters Used |
|------|--------|------|----------|-------------|---------------|
| California Housing | `sklearn.datasets.fetch_california_housing` | 20,640 | 8 | Median house value in California districts | Ch. 1 |
| Iris (versicolor vs virginica) | `sklearn.datasets.load_iris` (classes 1,2 only) | 100 | 2 (sepal length, width) | Binary classification, classes overlap | Ch. 1 |

## Optimization Test Problems

| Name | Description | Chapters Used |
|------|-------------|---------------|
| | | |

## Synthetic Generators

| Name | Generator | Purpose | Chapters Used |
|------|-----------|---------|---------------|
| Separable blobs | `make_blobs(centers=2, cluster_std=0.5, random_state=42)` | Perfect separation diagnostic failure | Ch. 1 |
