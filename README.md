# Language Identifier


Language Identifier model : takes a sentence as input and predict its language.

The model was trained on the papluca/language-identification dataset available on [HuggingFace](https://huggingface.co/datasets/papluca/language-identification). Pre-trained weights are stored in the *model* folder. It is possible to train the model on another dataset, as long as it contains 3 columns named "text","labels", and "alphabet".

The  model consists of two hierarchical steps :
1) Identify with regular expressions the main alphabet used in the text
2) Identify the language 
    - For alphabets with only one corresponding language : assign language directly based on regular expressions
    - For alphabets with more than one corresponding language (e.g. latin, arabic) : train a bi-directional LSTM network to predict the language

Currently supports the following languages : Arabic, Bulgarian, Chinese, Dutch, English, French, German, Greek, Hindi, Italian, Japanese, Polish, Portuguese, Russian, Spanish, Swahili, Thai, Turkish, Urdu, Vietnamese.

The model achieves an accuracy of 97% on the test set of papluca/language-identification.


# How to use

```python

    from lang_identifier import LanguageIdentifier 
    
    sentence = "Vous savez, moi je ne crois pas qu'il y ait de bonne ou de mauvaise situation."
    identifier = LanguageIdentifier()
    identifier.load('model/langidentifier')
    identifier.predict(sentence)
    >>> ['fr']
    
```


# Requirements

- datasets
- matplotlib
- numpy
- pandas
- scikit-learn
- tensorflow 
