# wv_covid_annotation_scripts

This is the privat repository for WeVerify EUvsVirus Hackathon Data and scripts

## Structure:

* README.md: This file
* source_data: The folder contains the unannotated source data
* annotated_data: The folder contains all annotated data
* mergedData: The folder contains the merged data
  * merged_all.json/tsv: Is merged data from all annotated files (exclude SocAlrm)
    * 1480 annotated samples
    * overall agreement:  0.5145888594164456
    * overall kappa:  0.4660477453580901
    * total pair compareed:  1131

  * merged_clean.json/tsv: Is the selected merged data
    * 1293 annotated samples
    * overall agreement:  0.7312348668280871
    * overall kappa:  0.7043583535108958
    * total pair compareed:  413

* mergeAnnos.py: Python script for data merge and agreement calcucation
* WvLibs: Folder contains WeVerify Covid data reader library
* run.sh: The bash script generate merged_all.json/tsv
* run_clean.sh: The bash script generate merged_clean.json/tsv


## mergeAnnos.py:
* Requirement: Python 3
* Useage: `python mergeAnnos.py raw_json_dir annoed_json_dir merged_json --output2csv merged_tsv [options]`
  * raw_json_dir: The folder contains the unannotated source data (source_data in this folder)
  * annoed_json_dir: The folder contains all annotated data (annotated_data in this folder)
  * merged_json: file path to merged json file
  * merged_tsv: file path to merged tsv file (This is required for accurate calculate kappa)
* Options inlcude:
`
optional arguments:
  -h, --help            show this help message and exit
  --ignoreLabel IGNORELABEL
                        ignore label, splited by using ,
  --ignoreUser IGNOREUSER
                        ignore user, splited by using ,
  --min_anno_filter MIN_ANNO_FILTER
                        min annotation frequence
  --min_conf_filter MIN_CONF_FILTER
                        min confident
  --output2csv OUTPUT2CSV
                        output to csv
  --transfer_label TRANSFER_LABEL
                        trasfer label to another category, in format:
                        orilabel1:tranlabel1,orilabel2:tranlabel2
  --cal_agreement       calculate annotation agreement
  --logging_level LOGGING_LEVEL
                        logging level, default warning, other option inlcude
                        info and debug
  --user_conf USER_CONF
                        User level confident cutoff threshold, in format:
                        user1:thres1,user2:thres2
  --set_reverse         reverse the selection condition, to check what
                        discared
`
