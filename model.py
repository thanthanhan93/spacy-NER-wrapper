import spacy
from spacy.gold import iob_to_biluo
import pandas as pd
from ast import literal_eval
import re
import time
import random
from spacy_custom_vector import add_vectors
from tqdm import tqdm
from spacy.gold import iob_to_biluo, GoldParse, docs_to_json
from spacy.scorer import Scorer
from util import print_beauty_NER, prediction_to_IOB
import json
import os
import subprocess

class ANNOTATED_DATA:
    """
    A class used to load and transform annotated data

    ...

    Attributes
    ----------
    META_DATA : dict
        contains meta information to load and trasnform data
    MODE : str
        the name of usage mode (train/test/predict)
    data : pd.DataFrame
        annotated data

    """
    def __init__(self, meta_data, mode):
        """
        Parameters
        ----------
        meta_data : dict
            contains meta information to load and trasnform data
        mode : str
            the name of usage mode (train/test/predict)
        """
        
        self.META_DATA = meta_data
        self.MODE = mode
    
    def load_data(self, **kwargs):
        """
        Parameters
        ----------
        kwargs : dict
            extra params of pd.read_csv
        """
        
        data_path = self.META_DATA[self.MODE]['path_data']
        col_asList = self.META_DATA[self.MODE]['list_cols']
        
        self.data = pd.read_csv(data_path, **kwargs).fillna('')
        
        # convert string to list
        for col in col_asList:
            self.data[col] = self.data[col].apply(literal_eval)
        return self
    
    
    def search_item(self, string):
        """Filter dataframe with pattern from string
        
        Parameters
        ----------
        string : str
            regex pattern
        """
        return self.data[self.data[self.META_DATA[self.MODE]['feature_col']]\
                            .str.contains(string)]
   

    def preprocess(self):
        """Clean data with defined patterns in meta info
        
        Parameters
        ----------
        """
        
        start_time = time.time()
        
        feature_cols = self.META_DATA[self.MODE]['preprocess_cols']
            
        self.data[feature_cols] = self.data[feature_cols].apply(
            lambda col: col.str.strip(), axis=0)

        for item in self.META_DATA['preprocess_rule']:
            self.data[feature_cols] = \
                self.data[feature_cols].apply(
                lambda col: col.str.replace(*item), axis=0)

        self.data[feature_cols] = \
            self.data[feature_cols].apply(
            lambda col: col.str.replace('\s+', ' ').str.strip(), axis=0)

        print('FINISH PREPROCESS DATA : dimensions %s' %
                    (str(self.data.shape)))
        print("FINISH PREPROCESS DATA : --- %s seconds ---" %
                    (time.time() - start_time))
        return self 
    
    def preprocess_token(self, word):
        """Clean input text with defined patterns in meta info
        
        Parameters
        ----------
        word : str
            text want to be preprocessed
        """
        word = word.strip()

        for item in self.META_DATA['preprocess_rule']:
            word = re.sub(*item, word)

        word = re.sub('\s+', ' ', re.sub('[^\w\s]',' ', word)).strip()
        return word.lower()
    
    def mapping_entities_to_unprocessed_data(self, row):
        text = row.lower_lpn
        tags = row.tag
        tags = sorted(row.tag, key=lambda x: x[1][0])
        entities = []
        for tag in tags:
            tag_tokens = self.preprocess_token(tag[0]).split(' ')
            tag_loc = tag[1]
            start_loc = 0
            end_loc = entities[-1][1] if len(entities)>0 else 0
            flag = 1
            try:
                for token in tag_tokens:
                    patterns = re.compile( r'\b' + token + r'\w*\b')
                    start_loc_new, end_loc = patterns.search(text, end_loc).span()
                    if flag:
                        start_loc=start_loc_new
                        flag =0

                entities.append([start_loc, end_loc, tag_loc[-1]])
                if(end_loc-start_loc < len(' '.join(tag_tokens))):
                    print("MATCHED TOKEN SORTER THAN ORIGINAL: ",
                          text, tag,
                          '(', start_loc, ',', end_loc, ')')
            except:
                print("MIDDLE STRING ING", text, tag)


        matched = 0
        if(len(entities) == len(tags)):
            matched = 1

        start_len = len(set([item[0] for item in entities]))
        end_len = len(set([item[1] for item in entities]))
        ent_len = len(entities) 
        if ((start_len != ent_len) | (end_len != ent_len)):
            print("DUPLICATES PROBLEM: ", text)
            print(tags)
            print(' |'.join([' |'.join([text[entity[0]:entity[1]],
                                        str(entity[0]),
                                        str(entity[1])]) for entity in entities]))
            print('')

        return pd.Series({'text':row.origin_lpn, 'entities':entities, 'match_all': matched})
    
    def convert_to_spacy_format(self):
        """Convert data to right format for spacy model
        
        Parameters
        ----------
        """
        start_time = time.time()
        self.data = self.data.apply(self.mapping_entities_to_unprocessed_data, axis=1)
        print("FINISH MAPPING TO ORIGINAL : --- %s seconds ---" %
                    (time.time() - start_time))
        return self.data.apply(lambda x: [x.text, {"entities": x.entities}], axis=1).tolist()
    
    def get_data(self):
        """Return data object based on input MODE
        
        Parameters
        ----------
        """
        
        if (self.MODE == 'predict'):
            return self.data
        else:
            return self.convert_to_spacy_format()
        
        
class NER_MODEL:
    """
    Spacy model wrapper

    ...

    Attributes
    ----------
    META_DATA : dict
        contains meta information to load, train and test the model
    MODE : str
        the name of usage mode (train/test/predict)
    data : pd.DataFrame or list
        pd.DataFrame with mode "predict", and list of entities with mode "train/test"
    nlp : spacy object
        the model contains Tokenizer + NER component 
    """
    def __init__(self, meta, mode):
        """
        Parameters
        ----------
        meta_data : dict
            contains meta information to load and trasnform data
        mode : str
            the name of usage mode (train/test/predict)
        """
        self.META_DATA = meta
        self.MODE = mode
    
    
    def change_mode(self, mode):
        """Change to new mode 
        
        Parameters
        ----------
        mode : str
            the name of usage mode (train/test/predict)
        """
        self.MODE = mode
     
    
    def load_models(self, model_path=None, init_model='de_core_news_sm'):
        """Load or init the NER model
        
        Parameters
        ----------
        model_path : str
            the path of the model of interest
        init_model : str
            only be used when model_path is empty, name of init model 
        """
        
        if model_path is not None:
            self.nlp = spacy.load(model_path, disable=['tagger', 'parser'])
            return self 
        
        if (self.MODE == 'train'):
            self.nlp = spacy.load(init_model, disable=["ner"])
        else:
            self.nlp = spacy.load(self.META_DATA['path_model'], disable=['tagger', 'parser'])
        return self
        
        
    def load_data(self, **kwargs):
        """
        Parameters
        ----------
        kwargs : dict
            extra params of pd.read_csv
        """
        
        self.data = \
            ANNOTATED_DATA(self.META_DATA, self.MODE)\
                .load_data(**kwargs)\
                .preprocess()
                
        self.data = self.data.get_data()
        return self
    
    
    def train_spacy(self, iterations, log_file='tmp.log', **kwargs):
        """
        Parameters
        ----------
        iteration : int
            number of epoch for model training
        log_file : str
            log file path - store loss from training
        **kwargs : dict
            extra parameter for spacy.update func
        """
        
        start_time = time.time()
        
        f = open(log_file, "w")
        f.write("Now the file has more content!")
        f.close()
        
        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.create_pipe("ner")
            self.nlp.add_pipe(ner, last=True)
        else:
            ner = self.nlp.get_pipe("ner")
       
        # add labels
        for _, annotations in self.data:
             for ent in annotations.get("entities"):
                ner.add_label(ent[2])
        
        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        
        # train
        with self.nlp.disable_pipes(*other_pipes):  # only train NER
            optimizer = self.nlp.begin_training()
            for itn in tqdm(range(iterations)):
                print("Statring iteration " + str(itn))
                random.shuffle(self.data)
                losses = {}
                for text, annotations in self.data:
                    self.nlp.update(
                        [text],  # batch of texts
                        [annotations],  # batch of annotations
                        losses=losses, 
                        sgd =optimizer,  # callable to update weights
                        **kwargs)
                
                f = open(log_file, "a")
                f.write(json.dumps(losses))
                f.write('\n')
                f.close()
                print(losses)
                
        print("FINISH TRAINING : --- %s seconds ---" %
                    (time.time() - start_time))
        return self
    
    def annotate_data(self, sample):
        doc = self.nlp(sample)
        labels = [(i.label_, i.text) for i in doc.ents]
        return labels 
    
    def init_model_pretrain_vect(model_path, vec_path, lang='de'):
        """Init spacy model with pre-trained vector file
        
        Parameters
        ----------
        model_path : str
            the path for saving init model 
        vec_path : str
            the path of pre-trained vector
        """
        
        cmd = 'python -m spacy init-model {} {} --vectors-loc {}'
        print(subprocess.check_output(cmd.format(lang, model_path, vec_path),
                                stderr=subprocess.STDOUT,
                                shell=True).decode('utf-8'))
        return None
        
    
    def predict(self):
        """Only available with mode 'predict'. Do prediction on DataFrame data
        
        Parameters
        ----------
        """
        start_time = time.time()
        def predict_map(row):
            ner_results = self.annotate_data(row[self.META_DATA[self.MODE]['feature_col']])
            is_brand_present = False
            is_favour_present = False
            ner_tokens = []

            for item in ner_results:
                if item[0] == 'Brand':
                    is_brand_present = True
                if item[0] == 'Flavor':
                    is_favour_present = True
                ner_tokens.append(item)

            ner_string = ' '.join([token[1] for token in ner_tokens])
            row['ner_tokens'] = str(ner_tokens)
            row['ner_tokens'] = literal_eval(row['ner_tokens'])
            row['ner_string'] = ner_string
            row['is_brand_present'] = is_brand_present
            row['is_favour_present'] = is_favour_present
            return row

                     
        self.data = self.data.apply(predict_map, axis=1)
        print("FINISH PREDICTION : --- %s seconds ---" %
                    (time.time() - start_time))
        return self
    
    def save_model(self):
        """Save model to path from config
        
        Parameters
        ----------
        """
        self.nlp.to_disk(self.META_DATA['path_model'])
        return self
    
    def evaluate(self, verbose=1):
        """Do evaluation on test data
        
        Parameters
        ----------
        verbose : bool
            print out the wrong case from prediction
        """
        scorer = Scorer()
        wrong_case = 0
        for input_, annot in self.data:

            doc_gold_text = self.nlp.make_doc(input_)
            gold = GoldParse(doc_gold_text, entities=annot['entities'])
            pred_value = self.nlp(input_)
            #return gold

            current_score = scorer.ents_f
            scorer.score(pred_value, gold)
            if (current_score > scorer.ents_f):
                wrong_case +=1 
                if (verbose==1):
                    print_beauty_NER(prediction_to_IOB(pred_value, gold))
                
            current_score = scorer.ents_f

        return scorer.scores#, wrong_case, len(self.data)