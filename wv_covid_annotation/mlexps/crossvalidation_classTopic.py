import sys
import math
from WvMLLib import BatchIterBert, DictionaryProcess
from WvMLLib import WVvaeDataIter as WVdataIter
from WvMLLib import ModelUltiClass as modelUlti
from WvMLLib import ReaderPostProcessorVAE as ReaderPostProcessor
from WvMLLib.models import ClassTopicModel_v2 as NVDM
from configobj import ConfigObj
import torch
import argparse
import copy
from sklearn.model_selection import KFold
import random
import os
from pathlib import Path
from gensim.corpora.dictionary import Dictionary

def xonlyBatchProcessor(x, y):
    ss = [s[1] for s in x]
    return ss[0]


def get_average_fmeasure_score(results_dict, field):
    t=0
    score = 0
    for class_field in results_dict['f-measure']:
        score += sum(results_dict['f-measure'][class_field][field])
        t += len(results_dict['f-measure'][class_field][field])
    return score/t

def get_micro_fmeasure(results_dict, num_field, de_field):
    score = 0
    for class_field in results_dict['f-measure']:
        numerator = sum(results_dict['f-measure'][class_field][num_field])
        denominator = sum(results_dict['f-measure'][class_field][de_field])
        if denominator != 0:
            score += numerator/denominator
    t = len(results_dict['f-measure'])

    return score/t


def maskedBertBatchProcessor(raw_x, y):
    x = [s[0] for s in raw_x]
    idded_words = [s[1] for s in raw_x]

    y_class = y
    return torch.tensor(x), torch.tensor(idded_words), torch.tensor(y_class)


def reconstruct_ids(each_fold, all_ids):
        output_ids = [[],[]] #[train_ids, test_ids]
        for sp_id in range(len(each_fold)):
            current_output_ids = output_ids[sp_id]
            current_fold_ids = each_fold[sp_id]
            for doc_id in current_fold_ids:
                current_output_ids.append(all_ids[doc_id])
        return output_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("merged_json_file", help="Merged Json file")
    parser.add_argument("--configFile", help="config files if needed")
    parser.add_argument("--cachePath", help="save models")
    parser.add_argument("--nFold", type=int, default=10, help="config files if needed")
    parser.add_argument("--randomSeed", type=int, help="randomSeed for reproduction")
    parser.add_argument("--preEmbd", default=False, action='store_true', help="calculate embedding before training")
    parser.add_argument("--dynamicSampling", default=False, action='store_true', help="sample based on class count")
    args = parser.parse_args()

    merged_json_file = args.merged_json_file
    config_file = args.configFile
    cache_path = args.cachePath
    if args.randomSeed:
        random.seed(args.randomSeed)


    config = ConfigObj(config_file)
    postProcessor = ReaderPostProcessor(tokenizer='bert', config=config, word2id=True, remove_single_list=False, add_spec_tokens=True)
    trainDataIter = WVdataIter(merged_json_file, postProcessor=postProcessor, config=config, shuffle=True)
    batchiter = BatchIterBert(trainDataIter, filling_last_batch=False, postProcessor=xonlyBatchProcessor, batch_size=1)
    common_dictionary = Dictionary(batchiter)
    dictProcess = DictionaryProcess(common_dictionary)
    postProcessor.dictProcess = dictProcess
    trainDataIter.count_samples()

    # precalculate bert embedding, speed up experiment
    print('pre calculating embedding')
    #batchIter = BatchIterBert(trainDataIter, filling_last_batch=False, postProcessor=maskedBertBatchProcessor)
    if args.preEmbd:
        net = NVDM(len(dictProcess), config)
        mUlti = modelUlti(net, gpu=True)
        trainDataIter.preCalculateEmbed(mUlti.net.bert_embedding, 0)

    #batchIter.dataIter.preCalculateEmbed(mUlti.net.bert_embedding, 0)



    # perpare split
    all_ids = copy.deepcopy(trainDataIter.all_ids)
    random.shuffle(all_ids)

    kf = KFold(n_splits=args.nFold)
    fold_index = 1
    results_dict = {}
    results_dict['accuracy'] = []
    results_dict['f-measure'] = {}

    testDataIter = copy.deepcopy(trainDataIter)

    for each_fold in kf.split(all_ids):
        print('running fold', str(fold_index))

        # reconstruct train and test ids in data reader
        train_val_ids, test_ids = reconstruct_ids(each_fold, all_ids)
        train_ids = train_val_ids
        trainDataIter.all_ids = train_ids
        testDataIter.all_ids = test_ids

        if args.dynamicSampling:
            print('get training data sample weights')
            trainDataIter.cal_sample_weights()
        else:
            trainDataIter.count_samples()
        testDataIter.count_samples()

        trainBatchIter = BatchIterBert(trainDataIter, filling_last_batch=True, postProcessor=maskedBertBatchProcessor)
        testBatchIter = BatchIterBert(testDataIter, filling_last_batch=False, postProcessor=maskedBertBatchProcessor, batch_size=32)


        net = NVDM(len(dictProcess), config)
        mUlti = modelUlti(net, gpu=True)

        fold_cache_path = os.path.join(cache_path, 'fold'+str(fold_index))
        path = Path(fold_cache_path)
        path.mkdir(parents=True, exist_ok=True)
        mUlti.train(trainBatchIter, cache_path=fold_cache_path, num_epohs=150, valBatchIter=testBatchIter, patience=15)
        results = mUlti.eval(testBatchIter)
        print(results)

        results_dict['accuracy'].append(results['accuracy'])
        for f_measure_class in results['f-measure']:
            if f_measure_class not in results_dict['f-measure']:
                results_dict['f-measure'][f_measure_class] = {'precision':[], 'recall':[], 'f-measure':[], 'total_pred':[], 'total_true':[], 'matches':[]}
            results_dict['f-measure'][f_measure_class]['precision'].append(results['f-measure'][f_measure_class][0])
            results_dict['f-measure'][f_measure_class]['recall'].append(results['f-measure'][f_measure_class][1])
            results_dict['f-measure'][f_measure_class]['f-measure'].append(results['f-measure'][f_measure_class][2])
            results_dict['f-measure'][f_measure_class]['total_pred'].append(results['f-measure'][f_measure_class][3])
            results_dict['f-measure'][f_measure_class]['total_true'].append(results['f-measure'][f_measure_class][4])
            results_dict['f-measure'][f_measure_class]['matches'].append(results['f-measure'][f_measure_class][5])
        #print(results)

        fold_index += 1
        #break
    print(results_dict)
    overall_accuracy = sum(results_dict['accuracy'])/len(results_dict['accuracy'])



    macro_precision = get_average_fmeasure_score(results_dict, 'precision')
    macro_recall = get_average_fmeasure_score(results_dict, 'recall')
    macro_fmeasure = get_average_fmeasure_score(results_dict, 'f-measure')


    micro_precision = get_micro_fmeasure(results_dict, 'matches', 'total_pred')
    micro_recall = get_micro_fmeasure(results_dict, 'matches', 'total_true')
    micro_fmeasure = 2*((micro_precision*micro_recall)/(micro_precision+micro_recall))





    print('accuracy: ', overall_accuracy)
    print('micro_precision: ', micro_precision)
    print('micro_recall: ', micro_recall)
    print('micro_f-measure: ', micro_fmeasure)
    print('macro_precision: ', macro_precision)
    print('macro_recall: ', macro_recall)
    print('macro_f-measure: ', macro_fmeasure)




