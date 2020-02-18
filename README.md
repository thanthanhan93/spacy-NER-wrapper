# spacy-NER-wrapper

this project aims at create wrapper for name entity module in spacy and expose some customization for model improvement at:
- hyper-parameter 
- custom embedding vector
- visualize wrong case in mode `test` with highlight 

In addition, some visualization of my result on my project.

## Usage of this API

### Configfile
we expect the config in the following format:

```json
meta_data = {
    # model
    "path_model": "model-path", 
    "preprocess_rule": [["<\/?[^>]+(>|$)", " "], 
         [u"\xa0|\-|\‐", " "], 
         [u"\'", ""],
         [r"([^\d\W]+)(\d+[^\s]*)", r"\1 \2"]],
    
    "train":
    {
        "path_data": "data/train_data.csv",
        "preprocess_cols": ['lower_lpn', 'origin_lpn'],
        "list_cols" : ['tag'],
        'feature_col' : 'origin_lpn'
    },
    "test":
    {
        "path_data": "data/test_data.csv",
        "preprocess_cols": ['lower_lpn', 'origin_lpn'],
        "list_cols" : ['tag'],
        "feature_col" : "origin_lpn"
    },
    "predict":
    {
        "path_data": "data/subbrand_data_processed.csv",
        "preprocess_cols": ["original_name"],
        "list_cols" : [],
        "feature_col" : "original_name",
    },
    "params":
    {
        "drop":0.4
    }
}
```

`preprocess_cols` are columns which will be preprocessed whereas `feature_col` is a column for input of NER. `tag` is groundtruth columns

### Example code
```python
# train 
models = NER_MODEL(meta_data, 'train')\
    .load_models()\
    .load_data(nrows=10)\
    .train_spacy(1, **meta_data['params'])\
    .save_model()
    
# test
models_test = NER_MODEL(meta_data, 'test')\
    .load_models()\
    .load_data()\
    .evaluate(verbose=0)
    
#predict
models_predict = NER_MODEL(meta_data, 'predict')\
    .load_models()\
    .load_data(nrows=10)\
    .predict()
```
