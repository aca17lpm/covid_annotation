import random
import json

class WVdataIter:
    def __init__(self, merged_json, postProcessor=None, shuffle=False, config=None):
        self.label_count_dict = {}
        self._readConfigs(config)
        self.shuffle = shuffle
        self._initReader(merged_json)
        self._reset_iter()
        self.postProcessor = postProcessor
        self.goPoseprocessor = True

    def _readConfigs(self, config):
        self.target_labels = None
        if config:
            if 'TARGET' in config:
                self.target_labels = config['TARGET'].get('labels')
                print(self.target_labels)


    def _initReader(self, merged_json):
        with open(merged_json, 'r') as f_json:
            merged_data = json.load(f_json)
        self.all_ids = []
        self.data_dict = {}
        

        for item in merged_data:
            select = True
            annotation = item['selected_label']
            if self.target_labels:
                if annotation not in self.target_labels:
                    #self.all_ids.append(item['unique_wv_id'])
                    #self.data_dict[item['unique_wv_id']] = item
                    select = False

            if select:
                self.all_ids.append(item['unique_wv_id'])
                self.data_dict[item['unique_wv_id']] = item



    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.all_ids)
        self._reset_iter()
        return self

    def __next__(self):
        if self.current_sample_idx < len(self.all_ids):
            current_sample = self._readNextSample()
            self.current_sample_idx += 1
            if self.postProcessor and self.goPoseprocessor:
                return self.postProcessor.postProcess(current_sample)
            else:
                return current_sample

        else:
            self._reset_iter()
            raise StopIteration


    def _readNextSample(self):
        current_id = self.all_ids[self.current_sample_idx]
        current_sample = self.data_dict[current_id]
        return current_sample


    def __len__(self):
        return len(self.all_ids)

    def _reset_iter(self):
        self.current_sample_idx = 0

    def count_samples(self):
        self.goPoseprocessor = False
        self.label_count_dict = {}
        for item in self:
            #print(item)
            annotation = item['selected_label']
            if annotation not in self.label_count_dict:
                self.label_count_dict[annotation] = 0
            self.label_count_dict[annotation] += 1
        print(self.label_count_dict)
        self.goPoseprocessor = True



