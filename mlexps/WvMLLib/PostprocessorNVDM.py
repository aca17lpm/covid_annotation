import nltk
from nltk.corpus import stopwords
import os

class ReaderPostProcessorNVDM:
    def __init__(self, tokenizer='nltk', 
            x_fields=['Claim', 'Explaination'], 
            x_output_mode='concat', 
            y_fields=['selected_label'], 
            y_output_mode='high_conf', 
            keep_case=False, 
            label2id=True, 
            config=None,
            word2id=False,
            exteralWord2idFunction=None,
            return_mask=False,
            remove_single_list=True,
            ):
        self._init_defaults()
        self.tokenizer = tokenizer
        self.x_fields = x_fields
        self.x_output_mode = x_output_mode
        self.y_fields = y_fields
        self.y_output_mode = y_output_mode
        self.keep_case = keep_case
        self.label2id = label2id
        self.word2id = word2id
        self.exteralWord2idFunction = exteralWord2idFunction
        self.config = config
        self.return_mask = return_mask
        self.remove_single_list = remove_single_list
        self.initProcessor()

    def _init_defaults(self):
        self.labelsFields = ['PubAuthAction', 'CommSpread', 'GenMedAdv', 'PromActs', 'Consp', 'VirTrans', 'VirOrgn', 'PubPrep', 'Vacc', 'Prot', 'None']
        self.stop_words = set(stopwords.words('english'))
        self.dictProcess = None

    def initProcessor(self):
        if self.tokenizer == 'nltk':
            self.tokenizerProcessor = self.nltkTokenizer

        if self.tokenizer == 'bert':
            from transformers import BertTokenizer
            bert_tokenizer_path = os.path.join(self.config['BERT'].get('bert_path'), 'tokenizer')
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)
            self.tokenizerProcessor = self.bertTokenizer
            self.word2idProcessor = self.bertWord2id

        if 'TARGET' in self.config:
            self.labelsFields = self.config['TARGET'].get('labels')
            print(self.labelsFields)
            description_file = self.config['TARGET'].get('description_file')
            self._readDescription(description_file)

    def _readDescription(self, description_file):
        self.desctiptionDict = {}
        with open(description_file, 'r') as fi:
            for line in fi:
                lineTok = line.split('\t')
                self.desctiptionDict[lineTok[0]] = lineTok[1]


    def _remove_stop_words(self, tokened):
        remain_list = []
        for token in tokened:
            keep = True
            if token in self.stop_words:
                keep = False
            elif len(token) == 1:
                keep = False

            if keep == True:
                remain_list.append(token)
        return remain_list


    def postProcess(self, sample):
        split_x = []
        for x_field in self.x_fields:
            current_rawx = sample[x_field]
            if self.keep_case == False:
                current_rawx = current_rawx.lower()
            split_x.append(current_rawx)
        if self.x_output_mode == 'concat':
            split_x = [' '.join(split_x)]

        processed_x = []
        processed_x_token_only = []
        for current_rawx in split_x:
            current_x = self.x_pipeline(current_rawx, add_special_tokens=False)
            current_x_nltk_tokened = self.nltkTokenizer(current_rawx)
            current_x_nltk_tokened = self._remove_stop_words(current_x_nltk_tokened)
            ## remove stopwords
            if self.dictProcess:
                current_x_nltk_tokened = self.dictProcess.doc2countHot(current_x_nltk_tokened)
            processed_x_token_only.append(current_x_nltk_tokened)
            processed_x.append(current_x)

        x=processed_x
        #x_nltk = current_x_nltk_tokened
        x.append(current_x_nltk_tokened)

        


        current_y = sample['selected_label']
        current_y_description = self.desctiptionDict[current_y]
        current_y_description = self.x_pipeline(current_y_description, max_length=100)
        current_y = self.label2ids(current_y)
        y = [current_y, current_y_description]

        if self.remove_single_list:
            x = self._removeSingleList(x)
            y = self._removeSingleList(y)
        #print(x, y)
        return x, y

    def _removeSingleList(self, y):
        if len(y) == 1:
            return y[0]
        else:
            return y

    def label2ids(self, label):
        label_index = self.labelsFields.index(label)
        return label_index


    def x_pipeline(self, raw_x, max_length=300, add_special_tokens=True):
        if self.tokenizer:
            raw_x = self.tokenizerProcessor(raw_x)
        if self.word2id:
            raw_x = self.word2idProcessor(raw_x, max_length=max_length, add_special_tokens=add_special_tokens)
        return raw_x

    def nltkTokenizer(self, text):
        return nltk.word_tokenize(text)

    def bertTokenizer(self, text):
        tokened = self.bert_tokenizer.tokenize(text)
        #print(tokened)
        #ided = self.bert_tokenizer.encode_plus(tokened, max_length=100, pad_to_max_length=True, is_pretokenized=True, add_special_tokens=True)['input_ids']
        #print(ided)
        return tokened

    def bertWord2id(self,tokened, max_length=300, add_special_tokens=True):
        encoded = self.bert_tokenizer.encode_plus(tokened, max_length=max_length, pad_to_max_length=True, is_pretokenized=True, add_special_tokens=add_special_tokens)
        #print(encoded)
        ided = encoded['input_ids']
        if self.return_mask:
            mask = encoded['attention_mask']
            return ided, mask
        else:
            return ided

    def get_label_desc_ids(self):
        label_desc_list = []
        for label in self.labelsFields:
            label_desc = self.desctiptionDict[label]
            current_desc_ids = self.x_pipeline(label_desc, max_length=100)
            label_desc_list.append(current_desc_ids)
        label_ids = [s[0] for s in label_desc_list]
        label_mask_ids = [s[1] for s in label_desc_list]
        return label_ids, label_mask_ids







