import config
from CXRMetric.run_eval import calc_metric
from CXRMetric.run_eval import CompositeMetric
from transformers import AutoTokenizer, AutoModel

''''
CUDA_VISIBLE_DEVICES=6 python test_metric.py
'''


gt_reports = config.GT_REPORTS
predicted_reports = config.PREDICTED_REPORTS
out_file = config.OUT_FILE
use_idf = config.USE_IDF

if __name__ == "__main__":
    calc_metric(gt_reports, predicted_reports, out_file, use_idf)
