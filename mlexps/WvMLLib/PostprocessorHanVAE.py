import nltk
from nltk.corpus import stopwords
import os
from .PostprocessorVAE import ReaderPostProcessorVAE
from nltk.tokenize import sent_tokenize
import re


class ReaderPostProcessorHanVAE(ReaderPostProcessorVAE):
    def __init__(self, use_source=False, max_doc_len=30, **kwargs):
        super().__init__(**kwargs)
        self.max_doc_len = max_doc_len
        self.use_source = use_source


    def clean_source(self,source_text):
        split_lines = source_text.split('\n')
        added_sent = []
        for splited_line in split_lines:
            keep = True
            line_tok = splited_line.split(' ')
            if len(line_tok) < 3:
                keep=False
            
            if keep:
                all_sents_split = sent_tokenize(splited_line)
                added_sent += all_sents_split

        return added_sent



    def postProcess(self, sample):
        #print(sample)
        split_x = []
        for x_field in self.x_fields:
            current_rawx = sample[x_field]
            if self.keep_case == False:
                current_rawx = current_rawx.lower()
            split_x.append(current_rawx)
        if self.x_output_mode == 'concat':
            split_x = ' '.join(split_x)

        sent_splited = sent_tokenize(split_x)

        if self.use_source:
            source_text = sample['Source_PageTextEnglish']
            if self.keep_case == False:
                source_text = source_text.lower()
            source_text = self.clean_source(source_text)
            sent_splited += source_text
        #print(sent_splited)


        ## Bert toknise for hidden layers. add_special_tokens not added, additional attention will be applied on token level (CLS not used)
        if self.embd_ready:
            current_x = sample['embd']
        else:
            current_x = []
            for each_sent in sent_splited:
                each_sent = re.sub('\n', '', each_sent)
                each_sent = re.sub('\t', '', each_sent)
                current_x.append(self.x_pipeline(each_sent, add_special_tokens=self.add_spec_tokens))
            current_x = self._padding_x(current_x)

        #print(current_x)



        ## NLTK tokenise and remove stopwords for topic modelling
        current_x_nltk_tokened = self.nltkTokenizer(' '.join(sent_splited))
        #current_x_nltk_tokened = self.nltkTokenizer(split_x)
        current_x_nltk_tokened = self._remove_stop_words(current_x_nltk_tokened)
        #print(current_x_nltk_tokened)
        if self.dictProcess:
            current_x_nltk_tokened = self.dictProcess.doc2countHot(current_x_nltk_tokened)

        x=[current_x, current_x_nltk_tokened]

        current_y = sample['selected_label']
        current_y = self.label2ids(current_y)
        y = current_y

        if self.remove_single_list:
            x = self._removeSingleList(x)
            y = self._removeSingleList(y)
        return x, y


    def _padding_x(self, current_x):
        padding_value = [0]*self.max_sent_len
        if len(current_x) > self.max_doc_len:
            current_x = current_x[:self.max_doc_len]
        while len(current_x) < self.max_doc_len:
            current_x.append(padding_value)
        return current_x










