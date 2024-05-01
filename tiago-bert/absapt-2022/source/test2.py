# https://colab.research.google.com/drive/1oDgxmc9aYNjBbzIQcv_g7XZax8On8zma?usp=sharing
# https://huggingface.co/neuralmind/bert-large-portuguese-cased
# https://github.com/neuralmind-ai/portuguese-bert

"""
pip3 install pyabsa=='1.9.6'
"""

import os
import findfile

from pyabsa.functional import ATEPCModelList
from pyabsa.functional import Trainer, ATEPCTrainer
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import ATEPCConfigManager
from pyabsa.functional.dataset import DatasetItem
from pyabsa import ATEPCCheckpointManager

atepc_config = ATEPCConfigManager.get_atepc_config_template()

atepc_config.dataset_name = 'test2_dataset'
atepc_config.learning_rate = 0.03 # default 0.00003
atepc_config.num_epoch = 8 # default 10
atepc_config.batch_size = 16 # default 16
atepc_config.evaluate_begin = 0 # default 4
atepc_config.log_step = 50 # default 100
atepc_config.patience = 6 # default 99999
atepc_config.dropout = 0.1 # default 0.5

atepc_config.cross_validate_fold = -1 # disable cross_validate

# atepc_config.model = ATEPCModelList.LCF_ATEPC
atepc_config.model = ATEPCModelList.FAST_LCF_ATEPC

# https://spacy.io/models/pt/
# python3 -m spacy download pt_core_news_md
atepc_config.spacy_model = 'pt_core_news_md'

from transformers import AutoModel
# model = AutoModel.from_pretrained('neuralmind/bert-large-portuguese-cased')
# model = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased')
atepc_config.pretrained_bert = 'neuralmind/bert-base-portuguese-cased'
# atepc_config.pretrained_bert = 'microsoft/deberta-v3-base'

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

dataset_path = 'dataset/100.absapt'

import pandas as pd
data = pd.read_csv("../dataset/test/test_task1.csv", sep=";")
examples = list(data["review"])
# examples = ['But the staff was so nice to us .',
#             'But the staff was so horrible to us .',
#             r'Not only was the food outstanding , but the little ` perks \' were great .',
#             'It took half an hour to get our check , which was perfect since we could sit , have drinks and talk !',
#             'It was pleasantly uncrowded , the service was delightful , the garden adorable , '
#             'the food -LRB- from appetizers to entrees -RRB- was delectable .',
#             'How pretentious and inappropriate for MJ Grill to claim that it provides power lunch and dinners !'
#             ]
inference_source = examples

aspect_extractor = ATEPCTrainer(config=atepc_config,
                                dataset=dataset_path,
                                from_checkpoint='checkpoints2',  # set checkpoint to train on the checkpoint.
                                checkpoint_save_mode=1,
                                auto_device=True
                                ).load_trained_model()

#### Second phase

# #checkpoint_map = available_checkpoints(from_local=True)
# #aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='portuguese', auto_device=True )
# aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='checkpoints/lcf_atepc_custom_dataset_cdw_apcacc_66.0_apcf1_26.51_atef1_0.0/lcf_atepc.config', auto_device=True )

# aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='lcf_atepc_custom_dataset')

atepc_result = aspect_extractor.extract_aspect(inference_source=inference_source,
                                               save_result=True,
                                               print_result=True,  # print the result
                                               pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                               )
