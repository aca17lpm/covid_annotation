import nltk
import os

class ReaderPostProcessor:
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

    #def _init_config(self, config):
    #    if 'TARGET' in config:
    #        self.labelsFields = config['TARGET'].get('labels')
    #        print(self.target_labels)

        
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


    def postProcess(self, sample):
        split_x = []
        for x_field in self.x_fields:
            current_rawx = sample[x_field]
            if self.keep_case == False:
                current_rawx = current_rawx.lower()
            split_x.append(current_rawx)
        if self.x_output_mode == 'concat':
            split_x = [' '.join(split_x)]
            #print(split_x)

        processed_x = []
        for current_rawx in split_x:
            current_x = self.x_pipeline(current_rawx)
            processed_x.append(current_x)

        x=processed_x


        split_y = []
        for y_field in self.y_fields:
            current_y = sample[y_field]
            current_y = self.label2ids(current_y)
            split_y.append(current_y)
        y = split_y

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

    def select_y(self, current_y):
        #print(current_y)
        selected_y = None
        if self.y_output_mode == 'high_conf':
            sorted_current_y = sorted(current_y, key=lambda s:s[1], reverse=True)
            #print(sorted_current_y)
            selected_y = sorted_current_y[0][0]
        return selected_y


    def label2ids(self, label):
        label_index = self.labelsFields.index(label)
        return label_index


    def x_pipeline(self, raw_x):
        if self.tokenizer:
            raw_x = self.tokenizerProcessor(raw_x)
        if self.word2id:
            raw_x = self.word2idProcessor(raw_x)
        return raw_x

    def nltkTokenizer(self, text):
        return nltk.word_tokenize(text)

    def bertTokenizer(self, text):
        tokened = self.bert_tokenizer.tokenize(text)
        #print(tokened)
        #ided = self.bert_tokenizer.encode_plus(tokened, max_length=100, pad_to_max_length=True, is_pretokenized=True, add_special_tokens=True)['input_ids']
        #print(ided)
        return tokened

    def bertWord2id(self,tokened):
        encoded = self.bert_tokenizer.encode_plus(tokened, max_length=300, pad_to_max_length=True, is_pretokenized=True, add_special_tokens=True)
        #print(encoded)
        ided = encoded['input_ids']
        if self.return_mask:
            mask = encoded['attention_mask']
            return ided, mask
        else:
            return ided






