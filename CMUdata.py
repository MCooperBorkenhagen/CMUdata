

# %%
import pandas as pd
import numpy as np
import os
import json
import re
import random
import nltk


def pad(wordform, maxlen):
    padlen = maxlen - len(wordform)
    return(wordform + ('_'*padlen))


def remove(list, pattern = '[0-9]'): 
    """
    Remove a string from each element of a list, defaults
    to removing numeric strings.
    """
    list = [re.sub(pattern, '', i) for i in list] 
    return(list)


# utility functions:
def phontable(PATH, phonlabel='two-letter'):
    df = pd.read_excel(PATH, engine='openpyxl')
    df = df[df[phonlabel].notna()]
    df = df.rename(columns={phonlabel: 'phoneme'})
    df = df.set_index('phoneme')
    feature_cols = [column for column in df if column.startswith('#')]
    df = df[feature_cols]
    df.columns = df.columns.str[1:]
    return(df)



# %%
class CMUdata():

    def __init__(self, download=False, phonlabel='two-letter'):
        """
        Parameters
        ----------
        download : bool
            If True, download cmudict before calling, else assume it is downloaded

        phonpath : str
            The path to the table specifying phonemes and their features, also passed to an init value

        phonlabel : str
            The type of phoneme coding to use, as specified in cols of the phontable
        """

        if download:
            nltk.download('cmudict')
        self.phonpath = 'https://raw.githubusercontent.com/MCooperBorkenhagen/ARPAbet/master/ARPAbet.xlsx' # set the default path to the phoneme data
        self.phonlabel = phonlabel # set the default phoneme label, can be overridden in any method           
        self.cmudict = nltk.corpus.cmudict.dict() # the raw cmudict object
        self.phontable = phontable(self.phonpath, phonlabel)


    def phon_features(self, phonlabel='two-letter', include_blank=True):
        """generate representations from phontable
        
        Parameters
        ----------
        phonlabel : str
            Label used for phonetic coding (specifies column from which the 
            label is drawn). Options are 'two-letter' (default; cmudict standard), 
            'one-letter', 'IPA', 'klattese', or 'wpa'. Note that 'one-letter',
            'klattese', and 'wpa' are subtle variations of each other.

        include_blank : bool
            Include a representation for '_' or not

        Returns
        -------
        dict
            Keys are phonemes and values are binary representations as specified in self.phontable.

        """
        out = {}
        df = phontable(self.phonpath, phonlabel=phonlabel)
        for index, row in df.iterrows():
            out[index] = row.tolist()
        if not include_blank:
            out.pop('_')
        return(out)


    def cmudict_clean(self, compounds=False, numerals=False, punctuation=False, tolower=True):
        
        cmu = self.cmudict

        if not compounds:
            for k in list(cmu.keys()):
                if '-' in k:
                    cmu.pop(k)

        if not numerals:
            for k, v in list(cmu.items()):
                r = []
                for e in v:
                    r.append(remove(e))
                cmu[k] = r


        if not punctuation:
            regex = re.compile('[^a-zA-Z]')
            for k in list(cmu.keys()): # you have to convert to list to change key of dict in loop
                if not k.isalpha():
                    new = regex.sub('', k)
                    cmu[new] = cmu.pop(k)

        if tolower:
            for k in list(cmu.keys()):
                new = k.lower()
                cmu[new] = cmu.pop(k)
        
        return(cmu)




    # %%
    def cmudict_with_reps(self, clean=True, check_phones=True, index='first', seed=999, compounds=False, numerals=False, punctuation=False, tolower=True):
        random.seed(seed)
        """
        Generate phonological codings for cmu words.

        :param reps: dict, specifying phone-wise codings to use, with labels as keys and codings (ie, feature representations) as values
        :param cmu: dict, version of cmudict that you want to encode as binary patterns using reps
        :param download: bool, download cmudict from nltk or not, passed to load_cmu call
        :param index: str, which cmudict phonological form to select. Takes 'first' or 'random' only
        :param check_phones: bool, before generating reps check that all the phonemes in cmu are present in reps
            (otherwise the phonological) constructions will be incomplete
        """
        random.seed(seed)
        
        if clean:
            cmu = self.cmudict_clean(compounds=compounds, numerals=numerals, punctuation=punctuation, tolower=tolower)
        else:
            cmu = self.cmudict

        reps = self.phon_features(phonlabel='two-letter')

        def represent(codelist, code_index):
            assert type(codelist[0]) == list, 'You have not provided a list of lists. Your return object will not be what you expect.'
            code = codelist[code_index]
            rep = []
            for phone in code:
                rep.append(reps[phone])
            return(rep)

        if check_phones:
            phones = []
            for v in cmu.values():
                for l in v:
                    for e in l:
                        phones.append(e)
            assert set(phones).issubset(reps.keys()), 'phones represented in cmudict are not present in reps'

        out = {}
        if index == 'first':
            i = 0
            for k, v in cmu.items():
                rep = represent(v, i)
                out[k] = rep
        elif index == 'random':
            for k, v in cmu.items():
                i = random.choice(range(len(v)))
                rep = represent(v, i)
                out[k] = rep
        return(out)



    def cmudict_padded(self, pad='right', maxphon=6, print_vocab_size=True, clean=True, check_phones=True, index='first', seed=999, compounds=False, numerals=False, punctuation=False, tolower=True): # cmu, padrep, 
        padded = {}
        cmu = self.cmudict_with_reps(clean=clean, check_phones=check_phones, index=index, seed=seed, compounds=compounds, numerals=numerals, punctuation=punctuation, tolower=tolower)
        tmp = self.phon_features(phonlabel=self.phonlabel)
        padrep = tmp['_']

        if maxphon is None:
            lens = set([len(phonform) for phonform in cmu.values()])
            maxphon = max(lens)
        else:
            cmu = {k: v for k, v in cmu.items() if len(v) <= maxphon}

        for word in list(cmu.keys()):
            padlen = maxphon-len(cmu[word])
            p = cmu[word]
            if pad == 'right':
                for slot in range(padlen):
                    p.append(padrep)
            elif pad == 'left':
                pad_ = []
                for slot in range(padlen):
                    pad_.append(padrep)
                pad_.extend(p)
                p = pad_
            padded[word] = p
        if print_vocab_size:
            print(len(list(padded.keys())), 'phonological wordforms from cmudict provided')
        return(padded)


    def cmudict_array(self, return_labels=False, maxphon=6, print_vocab=True, clean=True, check_phones=True, index='first', seed=999, compounds=False, numerals=False, punctuation=False, tolower=True):
        
        """Return the padded feature representations for cmudict phonological wordforms
        """
        
        cmu = self.cmudict_padded(maxphon=maxphon, print_vocab=print_vocab, clean=clean, check_phones=check_phones, index=index, seed=seed, compounds=compounds, numerals=numerals, punctuation=punctuation, tolower=tolower)
        X = []
        labels = []

        for k, v in cmu.items():
            X.append(v)
            labels.append(k)
        X = np.array(X)
        if return_labels:
            return(X, labels)
        else:
            return(X)


    

# %%
