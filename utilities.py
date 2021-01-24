"""
utility methods
"""
from __future__ import print_function, division
from __future__ import absolute_import, unicode_literals
import numpy as np
import pandas as pd
import os
import re

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
