# Language Identifier

Use this model to predict the language of an input text.

The model was trained on the papluca/language-identification dataset available on [HuggingFace](https://huggingface.co/datasets/papluca/language-identification). Pre-trained weights are stored in the *model* folder. It is possible to train the model on another dataset, as long as it contains 2 columns named "text" and "labels".

The  model consists of two hierarchical steps :
1) Identify the alphabet by detecting unicode characters with regular expressions 
2) Identify the language 
    - For alphabets with only one corresponding language : assign language directly based on regular expressions
    - For alphabets with more than one corresponding language (e.g. latin, arabic) : train a bi-directional LSTM network to predict the language

✅ Currently supports the following alphabets : Arabic, CJK, Cyrillic, Devanagari, Modern Greek, Latin, Thai.
✅ Currently supports the following languages : Arabic, Bulgarian, Chinese, Dutch, English, French, German, Greek, Hindi, Italian, Japanese, Polish, Portuguese, Russian, Spanish, Swahili, Thai, Turkish, Urdu, Vietnamese.

The model achieves an accuracy of 97% on the test set of papluca/language-identification.


# How to use

```python

    from lang_identifier import LanguageIdentifier 
    #A sentence written in French
    sentence = "Vous savez, moi je ne crois pas qu'il y ait de bonne ou de mauvaise situation."
    identifier = LanguageIdentifier()
    #Load pre-trained model 
    identifier.load('model/langidentifier') 
    #Accepted input formats are strings (one text), lists, or pandas DataFrames (multiple texts)
    identifier.predict(sentence)
    >>> ['fr']
    
```


# Contribute

Feel free to contribute by adding support for additional languages. 
