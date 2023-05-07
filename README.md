# README

## Data

The data comes from this url: [医疗费用个人数据集 数据集 - DataFountain](https://www.datafountain.cn/datasets/31). Much thank to the contributer : )

## Installation

Please run:  `pip install -r requirements.txt`

## Benchmarks

To run different algorithms on this dataset. We use the commands in the following table:

| Algorithms            | Commands                           | Performances(r square in test dataset, 10-fold) |
| --------------------- | ---------------------------------- | ----------------------------------------------- |
| NN (3-layer MLP)      | `python nn_main.py`                | 0.460±0.051                                     |
| Linear Regression     | `python linear_regression_main.py` | 0.744±0.051                                     |
| SVM                   | `python svm_main.py`               | 0.775±0.031                                     |
| Decision Tree         | `python decision_tree_main.py`     | 0.683±0.069                                     |
| XGBoost               | `python xgboost_main.py`           | 0.806±0.059                                     |
| Random Forest         | `python random_foreset_main.py`    | 0.833±0.048                                     |
| LightGBM              | `python lightgbm_main.py`          | 0.858±0.046                                     |
| Bayesian Method(ours) | `python bayesian_main.py`          | **0.845±0.064**                                 |

For visualization, run `python visualize.py`

## Citation

```
@misc{medical-bayesian-analysis,
    title={DataAnalysis},
    author={kxzxvbk},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/kxzxvbk/DataAnalysis}},
    year={2023},
}
```

## License

DataAnalysis released under the Apache 2.0 license.