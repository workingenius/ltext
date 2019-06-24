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

![labeled text example](./doc/lt_example.png)

```python
>>> lt0 = LabeledText('<p>鸟宿池边树，僧推月下门。</p>')
>>>
>>> # 1. Removing html tags
>>> lt1 = lt0.re_replace(r'<\/?p>', '')
>>> lt1
LabeledText.literal('鸟宿池边树，僧推月下门。')
>>>
>>> # 2. Annotate subjects
>>> #   add type-free labels
>>> #   show labels in square parenthesis paris
>>> lt2 = lt1.add_label([(0, 1), (6, 7)])
>>> lt2
LabeledText.literal('[鸟]宿池边树，[僧]推月下门。')
>>>
>>> # 3. Restore html tags
>>> lt3 = lt2.restore()
>>> lt3
LabeledText.literal('<p>[鸟]宿池边树，[僧]推月下门。</p>')
>>>
>>> # 4. zh punctuation to en
>>> lt4 = lt2.replace('，', ',').replace('。', '.')
>>> lt4
LabeledText.literal('[鸟]宿池边树,[僧]推月下门.')
>>>
>>> # 5. Annotate objects
>>> lt5 = lt4.add_label([(11, 12)])
>>> lt5
LabeledText.literal('[鸟]宿池边树,[僧]推月下[门].')
>>>
>>> # 6. Restore punctuation conversion
>>> lt6 = lt5.restore().restore()
>>> lt6
LabeledText.literal('[鸟]宿池边树，[僧]推月下[门]。')
>>>
>>> # 7. Restore html tags
>>> lt7 = lt6.restore()
>>> lt7
LabeledText.literal('<p>[鸟]宿池边树，[僧]推月下[门]。</p>')
>>>
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

