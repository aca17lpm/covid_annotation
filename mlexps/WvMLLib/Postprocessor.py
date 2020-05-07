import nltk
from nltk.corpus import stopwords
import os
from .PostprocessorVAE import ReaderPostProcessorVAE

class PostProcessorBaseline(ReaderPostProcessorVAE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def postProcess(self, sample):
        split_x = []
        for x_field in self.x_fields:
            current_rawx = sample[x_field]
            if self.keep_case == False:
                current_rawx = current_rawx.lower()
            split_x.append(current_rawx)
        if self.x_output_mode == 'concat':
            split_x = ' '.join(split_x)

        current_rawx = split_x
        ## Bert toknise for hidden layers. add_special_tokens not added, additional attention will be applied on token level (CLS not used)
        if self.embd_ready:
            current_x = sample['embd']
        else:
            current_x = self.x_pipeline(current_rawx, add_special_tokens=self.add_spec_tokens)

        x=[current_x, None]

        current_y = sample['selected_label']
        current_y = self.label2ids(current_y)
        y = current_y

        if self.remove_single_list:
            x = self._removeSingleList(x)
            y = self._removeSingleList(y)
        return x, y
