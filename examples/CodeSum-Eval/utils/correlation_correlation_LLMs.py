import random
import numpy as np
from scipy import stats
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
import warnings
from scipy.stats.mstats import gmean

warnings.filterwarnings('ignore')
import logging.config

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            'datefmt': '%m/%d/%Y %H:%M:%S'}},
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout'}},
    'loggers': {'': {'handlers': ['default']}}
}
logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)


def kendalltau_way1(x, y):
    count = 0
    Lens = len(x)
    for i in range(Lens - 1):
        for j in range(i + 1, Lens):
            count = count + np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
    kendallCorrelation = count / ((Lens * (Lens - 1)) / 2)
    return kendallCorrelation


def calculate_correlation(human_corpus_score, aggregrated_bleu_score, fc_bleu):
    for key in list(diff_score.columns):

        print(key)
        latex_result = ""
        if key == "BLEU-FC":
            result = kendalltau_way1(human_corpus_score, fc_bleu)
            _, pvalue = stats.kendalltau(human_corpus_score, fc_bleu)
            latex_result = latex_result + " &" + str(round(result, 2))
            print("KendalltauResult(correlation=%s pvalue=%s) " % (result, pvalue))

            # spearmanr
            result = stats.spearmanr(human_corpus_score, fc_bleu)
            latex_result = latex_result + " &" + str(round(result[0], 2))
            print(result)
        else:
            result = kendalltau_way1(human_corpus_score, aggregrated_bleu_score[key])
            _, pvalue = stats.kendalltau(human_corpus_score, fc_bleu)
            latex_result = latex_result + " &" + str(round(result, 2))
            print("KendalltauResult(correlation=%s pvalue=%s) " % (result, pvalue))

            result = stats.spearmanr(human_corpus_score, aggregrated_bleu_score[key])
            latex_result = latex_result + " &" + str(round(result[0], 2))
            print(result)

        # print(latex_result)


def sample_and_aggregrate_score(sample_num, iter_time, aggregrate_way="arithmetic_mean"):
    assert aggregrate_way in aggregrate_ways, RuntimeError(" aggregrate_ways must be in %s" % (aggregrate_ways))
    human_corpus_score = []
    aggregrated_bleu_score = {}
    fc_bleu = []
    # aggregrate_ways = ["arithmetic_mean", "geometric_mean"]
    # ori_target = human_annotation["Target"].tolist()
    for seed in range(iter_time):
        random.seed(seed)
        random_index = random.sample(range(0, 291), sample_num)
        if aggregrate_way == "arithmetic_mean":
            human_score = [human_avg[idx] for idx in random_index]
            human_corpus_score.append(np.mean(human_score))
        elif aggregrate_way == "geometric_mean":
            human_score = [human_avg[idx]+1 for idx in random_index]
            human_corpus_score.append(gmean(human_score))
        for key in list(diff_score.columns):
            score = diff_score[key].tolist()
            score_list = [score[idx] for idx in random_index]
            if key not in aggregrated_bleu_score:
                aggregrated_bleu_score[key] = []
            aggregrated_bleu_score[key].append(np.mean(score_list))

        sample_target = [target[idx] for idx in random_index]
        sample_predict = [predict[idx] for idx in random_index]
        c_bleu4 = corpus_bleu(sample_target, sample_predict, weights=(0.25, 0.25, 0.25, 0.25))
        c_bleu4 = round(c_bleu4 * 100, 4)
        fc_bleu.append(c_bleu4)
    print(50 * "*")
    calculate_correlation(human_corpus_score, aggregrated_bleu_score, fc_bleu)


if __name__ == '__main__':
    aggregrate_ways = ["arithmetic_mean"]
        # , "geometric_mean"]
    human_annotation = pd.read_excel("../dataset/human_evaluation/RQ1-2/human_evaluation.xlsx")
    target = human_annotation["Target"].tolist()
    predict = human_annotation['Generated'].tolist()
    target = [[item.split()] if type(item) != float else [""] for item in target]
    predict = [item.split() if type(item) != float else [""] for item in predict]
    diff_score_raw = pd.read_excel("../RQ1/evaluated_291_by_text-davinci-003_reference0_final.xlsx")
    for j in range(4):
        for i in range(4):
            diff_score = diff_score_raw.iloc[:, diff_score_raw.shape[1]-(i+1+j*4):diff_score_raw.shape[1]-(i+1+j*4)+1]
            name = diff_score_raw.columns[diff_score_raw.shape[1]-(i+1+j*4)]
            print(name)
            human_avg = human_annotation.iloc[:,
                        human_annotation.shape[1] - (i+1):human_annotation.shape[1] - (i+1) + 1].squeeze().tolist()
            name = human_annotation.columns[human_annotation.shape[1]-(i+1)]
            print(name)
            human_avg = list(human_avg)
            for aggregrate_way in aggregrate_ways:
                # print(100 * "*")
                # print("Aggregation way: %s"%aggregrate_way)
                for sample_num in [1]:
                    # print(50 * "-")
                    # print("corpus size %d" %sample_num)
                    sample_and_aggregrate_score(sample_num, iter_time=5000, aggregrate_way=aggregrate_way)
