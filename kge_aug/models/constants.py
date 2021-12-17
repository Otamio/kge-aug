from typing import Final

epochs: Final = 1000
dimension: Final = 200

# stopper
metric: Final = "mean_reciprocal_rank"
patience: Final = 3
frequency_short: Final = 5
delta: Final = 0.000000001
relative_delta: Final = 0.001

# dataset locations
dataset_fb15k237: Final = "datasets/fb15k237"
dataset_nations: Final = "datasets/nations"
dataset_yago15k: Final = "datasets/yago15k"

# dataset positions
link_prediction_path: Final = "data"
numeric_prediction_path: Final = "numeric"
link_prediction_path_processed: Final = "processed"