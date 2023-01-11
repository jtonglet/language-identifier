# -*- coding: utf-8 -*-
"""
Created on Saturday Jan 07 2023 
@author : Jonathan Tonglet
"""

#Import packages
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 


class LanguageIdentifier :
    '''
    A language identifier model, which given a text, predicts its language.
    Attributes:
        embeddings_dims (int): The embeddings dimensions for the LSTM submodels.
        multilingual_alphabets (list): The list of alphabets with multiple associated languages. 
        model_dict (dict): a dictionary with alphabet names as keys and corresponding LSTM submodels as values.
        labels_dict (dict): a dictionary with alphabet names as keys and corresponding languages labels as values.
        tokenizers_dict (dict): a dictionary with alphabet names as keys and corresponding tokenizer as values.
    '''
    def __init__(self,embeddings_dims,multilingual_alphabets=['cjk','arabic','cyrillic','latin']):
        self.embeddings_dims=embeddings_dims
        self.multilingual_alphabets = multilingual_alphabets
        self.model_dict = {}
        self.labels_dict = {}
        self.tokenizers_dict = {}


    def fit(self,train,validation=None):
        '''
        Instantiate and train a bidirectional LSTM submodel for each alphabet with more than 2 languages.
        Params:
            train (pandas.DataFrame) : the train set with columns ['text','labels','alphabet']
            validation (pandas.DataFrame) : the validation set with columns ['text','labels','alphabet']
        '''
        for a in train['alphabet'].unique():
            num_languages_in_alphabet = train[train.alphabet==a].labels.nunique()
            if a in self.multilingual_alphabets:   
                print('Alphabet %s : %s languages'%(a,num_languages_in_alphabet))
                #Define model architecture
                model = keras.Sequential([
                        keras.layers.Embedding(1000, self.embeddings_dims, input_length=200),
                        keras.layers.Bidirectional(keras.layers.LSTM(16, return_sequences=True)),
                        keras.layers.Bidirectional(keras.layers.LSTM(16)),
                        keras.layers.Dense(num_languages_in_alphabet, activation='softmax')
                                        ])
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                #Add model to dictionary
                self.model_dict[a] = model
            else:
                #Monolingual alphabet, no further classification, take the most frequent value of the category as language label
                self.labels_dict[a] = train[train['alphabet']==a].labels.value_counts().index[0]
        for k in self.model_dict.keys():
            #Create subsets of the train and validation set corresponding to the alphabet
            train_alphabet = train[train['alphabet']==k]
            languages_in_alphabet = list(train_alphabet.labels.unique())
            if not validation is None:
                validation_alphabet = validation[validation['alphabet']==k]
                validation_alphabet = validation_alphabet[validation['labels'].isin(languages_in_alphabet)]
            else:
                validation_alphabet = None
            #Store target labels in the label dictionary
            lencoder =LabelEncoder()
            lencoder.fit(train_alphabet['labels'])
            self.labels_dict[k] = list(lencoder.classes_)
            #Train LSTM model
            self.fit_LSTM_model(train_alphabet,validation_alphabet,self.model_dict[k])


           
    def fit_LSTM_model(self,train, validation, model):
        '''
        Train a bidirectional LSTM model for instances belonging to the same alphabet.
        '''
        #Initialize tokenizer
        alphabet = train['alphabet'].to_list()[0]
        if alphabet=='cjk':
            #For CJK, character level information is used and some punctuation (.,?!) is kept.
            tokenizer = keras.preprocessing.text.Tokenizer(num_words=1000, lower=True,filters='"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n',
                                                        split=" ",char_level=True)
        else:
            #A standard word-level tokenizer is used for the other alphabets
            tokenizer = keras.preprocessing.text.Tokenizer(num_words=1000, lower=True, split=" ")
        #Train tokenizer
        tokenizer.fit_on_texts(train['text'])
        #Add tokenizer to dictionary of tokenizers
        self.tokenizers_dict[alphabet] = tokenizer
        #Tokenize datasets
        x_train_seq = tokenizer.texts_to_sequences(train['text'])
        x_train_padded = keras.preprocessing.sequence.pad_sequences(x_train_seq,maxlen=200)
        lencoder =LabelEncoder()
        y_train = lencoder.fit_transform(train['labels'])
        y_train = tf.keras.utils.to_categorical(y_train)
        if not validation is None:
            x_validation_seq = tokenizer.texts_to_sequences(validation['text'])
            x_validation_padded = keras.preprocessing.sequence.pad_sequences(x_validation_seq,maxlen=200)
            y_validation = lencoder.transform(validation['labels'])
            y_validation = tf.keras.utils.to_categorical(y_validation)
            validation_data = (x_validation_padded, y_validation)
        else:
            validation_data = None
            
        #Fit model
        model.fit(x_train_padded, y_train, batch_size=32, epochs=2, validation_data=validation_data)


    def predict(self,data):
        '''
        Predict the language of all instances in a dataset
        '''
        preds = [0 for _ in range(data.shape[0])]
        #Retrieve index of texts and predictions for each alphabet
        for a in data['alphabet'].unique():
            if a in self.multilingual_alphabets: 
                alphabet_idx = data[data['alphabet']==a].index
                alphabet_preds = self.pred_LSTM(data[data['alphabet']==a])
                for idx in alphabet_idx:
                    preds[idx] =  self.labels_dict[a][np.argmax(alphabet_preds.pop(0))]
            else:
            #Alphabets with one language, direct prediction
                for idx in data[data['alphabet']==a].index:
                    preds[idx] = self.labels_dict[a]
        return preds


    def pred_LSTM(self, data):
        '''
        Predict all languages for one alphabet with the corresponding LSTM model.
        '''
        alphabet = data['alphabet'].to_list()[0] #Identify alphabet
        x_seq = self.tokenizers_dict[alphabet].texts_to_sequences(data['text']) #Retrieve tokenizer
        x_seq_padded = keras.preprocessing.sequence.pad_sequences(x_seq,maxlen=200)
        preds = list(self.model_dict[alphabet].predict(x_seq_padded,verbose=0))
        return preds

    
    def save(self,path='model/langidentifier/'):
        if not os.path.isdir(path):
            os.makedirs(path)
        for k, m in self.model_dict.items():
            #Save LSTM models
            filename = 'LSTM_'+k
            m.save(os.path.join(path,filename),save_format='h5')
        with open(os.path.join(path,'labels.pickle'),'wb') as handle:
            pickle.dump(self.labels_dict,handle)
        with open(os.path.join(path,'tokenizers.pickle'),'wb') as handle:
            pickle.dump(self.tokenizers_dict,handle)

    def load(self,path='model/langidentifier/'):
        for filename in  os.listdir(path):
            f = os.path.join(path,filename)
            if 'LSTM' in f:
                self.model_dict[filename.split('_')[-1]] = keras.models.load_model(f)
            if filename=='labels.pickle':
                with open(f,'rb') as handle:
                    self.labels_dict = pickle.load(handle)
            if filename=='tokenizers.pickle':
                with open(f,'rb') as handle:
                    self.tokenizers_dict = pickle.load(handle)