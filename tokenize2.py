import re

import unicodedata

from emoji import EMOJI_DATA


all__ = [
    'UNICODE_PUNCT_CHARSET', 'UNICODE_SYMBOL_CHARSET',
    'UNICODE_LETTER_CHARSET', 'UNICODE_MARK_CHARSET', 'UNICODE_NUMBER_CHARSET',
    'UNICODE_PUNCT_SYMBOL_CHARSET', 'UNICODE_LETTER_MARK_NUMBER_CHARSET',
    'EMOJI_ALL_CHARSET', 'EMOJI_SINGLECHAR_CHARSET', 'EMOJI_MULTICHAR_CHARSET', 
    'AR_LETTERS_CHARSET', 'AR_DIAC_CHARSET', 'AR_CHARSET',
    'BW_LETTERS_CHARSET', 'BW_DIAC_CHARSET', 'BW_CHARSET',
    'SAFEBW_LETTERS_CHARSET', 'SAFEBW_DIAC_CHARSET', 'SAFEBW_CHARSET',
    'XMLBW_LETTERS_CHARSET', 'XMLBW_DIAC_CHARSET', 'XMLBW_CHARSET',
    'HSB_LETTERS_CHARSET', 'HSB_DIAC_CHARSET', 'HSB_CHARSET',
]


UNICODE_PUNCT_CHARSET = set()
UNICODE_SYMBOL_CHARSET = set()
UNICODE_LETTER_CHARSET = set()
UNICODE_MARK_CHARSET = set()
UNICODE_NUMBER_CHARSET = set()

for x in range(0x110000):
    x_chr = chr(x)
    x_cat = unicodedata.category(x_chr)
    if x_cat[0] == 'L':
        UNICODE_LETTER_CHARSET.add(x_chr)
    elif x_cat[0] == 'M':
        UNICODE_MARK_CHARSET.add(x_chr)
    elif x_cat[0] == 'N':
        UNICODE_NUMBER_CHARSET.add(x_chr)
    elif x_cat[0] == 'P':
        UNICODE_PUNCT_CHARSET.add(x_chr)
    elif x_cat[0] == 'S':
        UNICODE_SYMBOL_CHARSET.add(x_chr)

UNICODE_PUNCT_CHARSET = frozenset(UNICODE_PUNCT_CHARSET)
UNICODE_SYMBOL_CHARSET = frozenset(UNICODE_SYMBOL_CHARSET)
UNICODE_LETTER_CHARSET = frozenset(UNICODE_LETTER_CHARSET)
UNICODE_MARK_CHARSET = frozenset(UNICODE_MARK_CHARSET)
UNICODE_NUMBER_CHARSET = frozenset(UNICODE_NUMBER_CHARSET)
UNICODE_PUNCT_SYMBOL_CHARSET = UNICODE_PUNCT_CHARSET | UNICODE_SYMBOL_CHARSET
UNICODE_LETTER_MARK_NUMBER_CHARSET = (UNICODE_LETTER_CHARSET |
                                      UNICODE_MARK_CHARSET |
                                      UNICODE_NUMBER_CHARSET)


EMOJI_ALL_CHARSET = frozenset(EMOJI_DATA.keys())
EMOJI_SINGLECHAR_CHARSET = frozenset([
    x for x in EMOJI_ALL_CHARSET if len(x) == 1])
EMOJI_MULTICHAR_CHARSET = frozenset([
    x for x in EMOJI_ALL_CHARSET if len(x) > 1])


AR_LETTERS_CHARSET = frozenset(u'\u0621\u0622\u0623\u0624\u0625\u0626\u0627'
                               u'\u0628\u0629\u062a\u062b\u062c\u062d\u062e'
                               u'\u062f\u0630\u0631\u0632\u0633\u0634\u0635'
                               u'\u0636\u0637\u0638\u0639\u063a\u0640\u0641'
                               u'\u0642\u0643\u0644\u0645\u0646\u0647\u0648'
                               u'\u0649\u064a\u0671\u067e\u0686\u06a4\u06af')
AR_DIAC_CHARSET = frozenset(u'\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652'
                            u'\u0670')
AR_CHARSET = AR_LETTERS_CHARSET | AR_DIAC_CHARSET

BW_LETTERS_CHARSET = frozenset(u'$&\'*<>ADEGHJPSTVYZ_bdfghjklmnpqrstvwxyz{|}_')
BW_DIAC_CHARSET = frozenset(u'FKN`aiou~')
BW_CHARSET = BW_LETTERS_CHARSET | BW_DIAC_CHARSET

SAFEBW_LETTERS_CHARSET = frozenset(u'ABCDEGHIJLMOPQSTVWYZ_bcdefghjklmnpqrstvwx'
                                   u'yz')
SAFEBW_DIAC_CHARSET = frozenset(u'FKNaeiou~')
SAFEBW_CHARSET = SAFEBW_LETTERS_CHARSET | SAFEBW_DIAC_CHARSET

XMLBW_LETTERS_CHARSET = frozenset(u'$\'*ABDEGHIJOPSTWYZ_bdfghjklmnpqrstvwxyz{|'
                                  u'}')
XMLBW_DIAC_CHARSET = frozenset(u'FKN`aiou~')
XMLBW_CHARSET = XMLBW_LETTERS_CHARSET | XMLBW_DIAC_CHARSET

HSB_LETTERS_CHARSET = frozenset(u'\'ADHST_bcdfghjklmnpqrstvwxyz'
                                u'\u00c2\u00c4\u00e1\u00f0\u00fd\u0100\u0102'
                                u'\u010e\u0127\u0161\u0175\u0177\u03b3\u03b8'
                                u'\u03c2')
HSB_DIAC_CHARSET = frozenset(u'.aiu~\u00c4\u00e1\u00e3\u0129\u0169')
HSB_CHARSET = HSB_LETTERS_CHARSET | HSB_DIAC_CHARSET


__all__ = ['simple_word_tokenize']


_ALL_PUNCT_SYMBOLS = (UNICODE_PUNCT_SYMBOL_CHARSET | EMOJI_MULTICHAR_CHARSET)
_ALL_PUNCT_SYMBOLS = [re.escape(x) for x in _ALL_PUNCT_SYMBOLS]
_ALL_PUNCT_SYMBOLS = sorted(_ALL_PUNCT_SYMBOLS, key=len, reverse=True)

_ALL_NUMBER = u''.join(UNICODE_NUMBER_CHARSET)
_ALL_LETTER_MARK = u''.join((UNICODE_LETTER_CHARSET | UNICODE_MARK_CHARSET))
_ALL_LETTER_MARK_NUMBER = u''.join(UNICODE_LETTER_MARK_NUMBER_CHARSET)

_TOKENIZE_RE = re.compile(u'|'.join(_ALL_PUNCT_SYMBOLS) + r'|[' +
                          re.escape(_ALL_LETTER_MARK_NUMBER) + r']+')
_TOKENIZE_NUMBER_RE = re.compile(u'|'.join(_ALL_PUNCT_SYMBOLS) + r'|[' +
                                 re.escape(_ALL_NUMBER) + r']+|[' +
                                 re.escape(_ALL_LETTER_MARK) + r']+')


def simple_word_tokenize(sentence, split_digits=False):
    """Tokenizes a sentence by splitting on whitespace and seperating
    punctuation. The resulting tokens are either alpha-numeric words, single
    punctuation/symbol/emoji characters, or multi-character emoji sequences.
    This function is language agnostic and splits all characters marked as
    punctuation or symbols in the Unicode specification.
    For example, tokenizing :code:`'Hello,    world!!!'`
    would yield :code:`['Hello', ',', 'world', '!', '!', '!']`.
    If split_digits is set to True, it also splits on number.
    For example, tokenizing :code:`'Hello,    world123!!!'`
    would yield :code:`['Hello', ',', 'world', '123', '!', '!', '!']`.

    Args:
        sentence (:obj:`str`): Sentence to tokenize.
        split_digits (:obj:`bool`, optional): The flag to split on number.
            Defaults to False.

    Returns:
        :obj:`list` of :obj:`str`: The list of tokens.
    """

    if split_digits:
        return _TOKENIZE_NUMBER_RE.findall(sentence)
    else:
        return _TOKENIZE_RE.findall(sentence)