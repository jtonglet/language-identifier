# -*- coding: utf-8 -*-
"""
Created on Tuesday Jan 10 2023 
@author : Jonathan Tonglet
"""

import re
#Alphabet recognition regular expressions
#Create additional is_alphabet functions with corresponding unicodes to cover more alphabets 

def is_latin(text):
    #Languages in dataset : Dutch, English, French, German, Italian, Polish, Portuguese, Spanish, Swahili, Turkish, Vietnamese
    return bool(re.search('[\u0000-\u007F]', text))
def is_cyrillic(text):
    #Languages in dataset : Bulgarian, Russian
    return bool(re.search('[\u0400-\u04FF]', text))
def is_devanagari(text):
    #Languages in dataset : Hindi.
    return bool(re.search('[\u0900-\u097F]', text))
def is_arabic(text):
    #Languages in dataset : Arabic, Urdu.
    return bool(re.search('[\u0600-\u06FF]', text))
def is_greek(text):
    #Languages in dataset : Modern Greek.
    return bool(re.search('[\u0370-\u03FF]', text))
def is_cjk(text):
    #Languages in dataset : Chinese, Japanese.
    return bool(re.search('[\u4E00-\u9FFF]', text))
def is_thai(text):
    #Languages in dataset : Thai.
    return bool(re.search('[\u0E00-\u0E7F]', text))


def check_alphabet(text):
    #Assign its corresponding alphabet to a text.
    if is_cjk(text):
        alphabet = 'cjk'
    elif is_arabic(text):
        alphabet = 'arabic'
    elif is_devanagari(text):
        alphabet = 'devanagari'
    elif is_cyrillic(text):
        alphabet = 'cyrillic'
    elif is_greek(text):
        alphabet = 'greek'
    elif is_thai(text):
        alphabet = 'thai'
    elif is_latin(text):
        alphabet = 'latin'
    else:
    #Non-assigned texts in train set are CJK texts with special characters not listed in the regular expressions
        alphabet = 'cjk'
    return alphabet