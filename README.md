# ltext — annotation offset auto-calculation

![annotation](./doc/annotation.png)

>  picture borrowed from https://brat.nlplab.org

## Inspiration

In NLP tasks, preprocessing corpus and annotating texts (add labels on texts) may happen in turn. Every label on the string should stay still and be kept as it is against all kinds of text operations, like removing extra blanks, upper case some certain substring, etc.

This lib encapsulates offset calculation of labels, in a class `LabeledText`. It aims to act like a string (supporting common methods of `str`), while keeps the excat location of every label, and protects them from being modified.

Here's two common senarios that you may need ltext:

+ corpus with annotation need a cleanning before model training
+ annotation can be restored on the original text, even when you preprocess the text before you add the annotation

You play with your corpus at will, with labels well protected.

## Example

```python
>>> lt = LabeledText('world--wide-web')
>>> lt
LabeledText.literal('world--wide-web')
>>>
>>> # cleanning
...
>>> lt = lt.replace('-', ' ')
>>> lt
LabeledText.literal('world  wide web')
>>> lt = lt.re_replace(r' +', ' ')
>>> lt
LabeledText.literal('world wide web')
>>>
>>> # annotate
...
>>> lt = lt.add_label([(0, 1), (6, 7), (11, 12)])
>>> lt
LabeledText.literal('[w]orld [w]ide [w]eb')
>>>
>>> # restore
...
>>> lt = lt.restore(till_end=True)
LabeledText.literal('[w]orld--[w]ide-[w]eb')

```

## Installation

Please clone the repo and pip install in editable mode. The lib has not been published to pypi yet.

```shell
git clone https://github.com/workingenius/ltext.git
cd ltext
# activate your custom python env
pip install -e ./
```

And enjoy.

