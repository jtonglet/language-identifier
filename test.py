from lang_identifier import LanguageIdentifier 
    
sentence = ["Vous savez, moi je ne crois pas qu'il y ait de bonne ou de mauvaise situation.",
            "Whether 'tis nobler in the mind to suffer. The slings and arrows of outrageous fortune.",
            "Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura,ché la diritta via era smarrita."
]
identifier = LanguageIdentifier()
identifier.load('model/langidentifier')
print(identifier.predict(sentence))