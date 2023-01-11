import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from lang_identifier import LanguageIdentifier 
from alphabets import check_alphabet


if __name__=='__main__':
    #Load datasets
    dataset = datasets.load_dataset('papluca/language-identification')
    train = pd.DataFrame(dataset['train'])
    validation = pd.DataFrame(dataset['validation'])
    test = pd.DataFrame(dataset['test'])
    #Train the identifier model
    identifier = LanguageIdentifier(embeddings_dims=50,multilingual_alphabets=['cjk','arabic','cyrillic','latin'])  
    identifier.fit(train,validation)

    #Make predictions
    preds = identifier.predict(test)
    #Evaluate results
    print('Accuracy score : %s'%accuracy_score(test['labels'],preds))
    print('F1 score : %s'%f1_score(test['labels'],preds,average='macro'))
    confusion_matrix = confusion_matrix(test['labels'],preds)
    #Plot confusion matrix
    fig, ax = plt.subplots(figsize=(7,7))
    display_labels = ['ar','bg','de','el','en','es','fr','hi','it','jp','nl','pl','pt','ru','sw','th','tr','ur','vi','zh']
    disp = ConfusionMatrixDisplay(confusion_matrix,display_labels=display_labels)
    disp.plot(ax=ax, xticks_rotation='vertical', values_format='d')
    plt.title("Confusion Matrix of the model predictions")
    plt.show()