import sys
import math
from WvMLLib import WVdataIter, ReaderPostProcessor, BatchIterBert, modelUlti
from WvMLLib.models import BERT_Simple
from configobj import ConfigObj
import torch
import argparse
import copy
from sklearn.model_selection import KFold
import random
import os
from pathlib import Path



def get_average_fmeasure_score(results_dict, field):
    t=0
    score = 0
    for class_field in results_dict['f-measure']:
        score += sum(results_dict['f-measure'][class_field][field])
        t += len(results_dict['f-measure'][class_field][field])
    return score/t

def maskedBertBatchProcessor(x, y):
    word_ids = [s[0] for s in x]
    mask = [s[1] for s in x]
    y = y
    return torch.tensor(word_ids), torch.tensor(y), torch.tensor(mask)


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
    args = parser.parse_args()

    merged_json_file = args.merged_json_file
    config_file = args.configFile
    cache_path = args.cachePath
    if args.randomSeed:
        random.seed(args.randomSeed)


    config = ConfigObj(config_file)
    postProcessor = ReaderPostProcessor(tokenizer='bert', config=config, word2id=True, return_mask=True, remove_single_list=True)
    dataIter = WVdataIter(merged_json_file, postProcessor=postProcessor.postProcess, config=config)

    all_ids = copy.deepcopy(dataIter.all_ids)
    random.shuffle(all_ids)

    kf = KFold(n_splits=args.nFold)
    fold_index = 1
    results_dict = {}
    results_dict['accuracy'] = []
    results_dict['f-measure'] = {}
    for each_fold in kf.split(all_ids):
        print('running fold', str(fold_index))
        #print(each_fold[1])
        train_val_ids, test_ids = reconstruct_ids(each_fold, all_ids)
        #print(test_links[0])

        random.shuffle(train_val_ids)
        top90_train = math.floor(len(train_val_ids) * 0.9)
        train_ids = train_val_ids[:top90_train]
        val_ids = train_val_ids[top90_train:]

        trainDataIter = copy.deepcopy(dataIter)
        valDataIter = copy.deepcopy(dataIter)
        testDataIter = copy.deepcopy(dataIter)

        trainDataIter.all_ids = train_ids
        valDataIter.all_ids = val_ids
        testDataIter.all_ids = test_ids
        trainBatchIter = BatchIterBert(trainDataIter, filling_last_batch=True, postProcessor=maskedBertBatchProcessor)
        testBatchIter = BatchIterBert(testDataIter, filling_last_batch=False, postProcessor=maskedBertBatchProcessor)
        valBatchIter = BatchIterBert(valDataIter, filling_last_batch=False, postProcessor=maskedBertBatchProcessor)
        net = BERT_Simple(config)
        mUlti = modelUlti(net, gpu=True)
        fold_cache_path = os.path.join(cache_path, 'fold'+str(fold_index))
        path = Path(fold_cache_path)
        path.mkdir(parents=True, exist_ok=True)
        mUlti.train(trainBatchIter, valBatchIter=valBatchIter, cache_path=fold_cache_path, num_epohs=150)
        results = mUlti.eval(testBatchIter)

        results_dict['accuracy'].append(results['accuracy'])
        for f_measure_class in results['f-measure']:
            if f_measure_class not in results_dict['f-measure']:
                results_dict['f-measure'][f_measure_class] = {'precision':[], 'recall':[], 'f-measure':[]}
            results_dict['f-measure'][f_measure_class]['precision'].append(results['f-measure'][f_measure_class][0])
            results_dict['f-measure'][f_measure_class]['recall'].append(results['f-measure'][f_measure_class][1])
            results_dict['f-measure'][f_measure_class]['f-measure'].append(results['f-measure'][f_measure_class][2])



        print(results)

        fold_index += 1
    print(results_dict)
    overall_accuracy = sum(results_dict['accuracy'])/len(results_dict['accuracy'])
    overall_precision = get_average_fmeasure_score(results_dict, 'precision')
    overall_recall = get_average_fmeasure_score(results_dict, 'recall')
    overall_fmeasure = get_average_fmeasure_score(results_dict, 'f-measure')

    print('accuracy: ', overall_accuracy)
    print('precision: ', overall_precision)
    print('recall: ', overall_recall)
    print('f-measure: ', overall_fmeasure)





    


    #batchIter = BatchIterBert(dataIter, filling_last_batch=True, postProcessor=maskedBertBatchProcessor)
    #net = BERT_Simple(config)
    #mUlti = modelUlti(net, gpu=True)
    #mUlti.train(batchIter)









