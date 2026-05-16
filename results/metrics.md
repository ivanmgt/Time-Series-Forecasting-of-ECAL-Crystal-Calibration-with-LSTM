## Evaluation Metrics

To evaluate the performance of our predictive models for individual crystals, we use the **MAPE (Mean Absolute Percentage Error)** metric. MAPE measures, on average, the magnitude of the absolute prediction error relative to the observed value, expressing it as a percentage. This provides a direct and intuitive interpretation of the error, indicating how large the average deviation of the model is with respect to the true values in relative terms.

Formally, MAPE is defined as:

$$
MAPE=\frac{100}{n}\sum_{i=1}^{n}
\left|
\frac{y_i-\hat{y}_i}{y_i}
\right|
$$

One of the main advantages of MAPE is its scale independence, making it particularly useful for comparing the performance of different models trained on different datasets or variables. In practical applications, lower MAPE values indicate a better model fit, whereas higher values suggest larger discrepancies between predictions and observed data.

However, MAPE has some limitations that must be carefully considered. In particular, the metric is undefined when any observed value $y_i$ equals zero and can become extremely large when $y_i$ approaches zero, potentially distorting the evaluation. In our case, this issue does not arise because calibration values equal to zero are considered invalid measurements and are removed during preprocessing.

For models trained and evaluated across all crystals within a ring, we use a weighted version of MAPE called **WMAPE (Weighted Mean Absolute Percentage Error)**:

$$
WMAPE=
\frac{\sum_i(MAPE_i\times N_i)}
{\sum_iN_i}
$$

where $N_i$ represents the number of records associated with crystal $i$.
