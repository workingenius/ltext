# -*- coding:utf8 -*-

"""

Text with non-overlapping labels

It aims to help a typical clean, label, and finally restore workflow.
When the text is restored, the labels added are preserved, by auto-calculating offset of the labels.
Here's an example

>>> lt = LabeledText('world--wide-web')
>>> lt
LabeledText.literal('world--wide-web')
>>>
>>> # Start clean
...
>>> lt = lt.replace('-', ' ')
>>> lt
LabeledText.literal('world  wide web')
>>> lt = lt.re_replace(r' +', ' ')
>>> lt
LabeledText('world wide web')
>>>
>>> # Start label
...
>>> lt = lt.add_label([(0, 1), (6, 7), (11, 12)])
>>> lt
LabeledText.literal('[w]orld [w]ide [w]eb')
>>>
>>> # Start restore
...
>>> lt = lt.restore(till_end=True)
LabeledText.literal('[w]orld--[w]ide-[w]eb')

"""

import itertools
import re
import unittest
from pprint import pformat
import logging

try:
    # noinspection PyUnresolvedReferences
    from typing import List, Tuple, Optional
except ImportError:
    pass


logger = logging.getLogger(__name__)


class LTextBaseException(Exception):
    pass


class LTextSpanOverlapping(LTextBaseException):
    pass


class Span(object):
    def __init__(self, start, end):
        if end < start:
            raise ValueError('start should always be lte end')
        self.start = start
        self.end = end

    @property
    def length(self):
        return self.end - self.start

    @property
    def center(self):
        return (self.start + self.end) // 2

    def __gt__(self, other):
        return is_right(self, other)

    def __lt__(self, other):
        return is_left(self, other)

    def __repr__(self):
        return 'Span({}, {})'.format(self.start, self.end)

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __bool__(self):
        return not is_empty(self)

    __nonzero__ = __bool__


# Relation checkers of the two Span
# suppose that span a and span b is on the same string
#
# possible relations are:
#
#   a is at RIGHT of b     |--b--|  |---a---|
#
#   a is at LEFT of b      |--a--|  |--b--|
#
#   a and b OVERLAPs       |---a---|
#                              |--b--|
#
#   a CONTAINs b           |---a---|
#                           |--b--|
#
#   a and b are EQUAL       |---a---|
#                           |---b---|
#
# But empty span is subtle.
# If two empty span at same location, is_right(a, b) is_left(a, b) is_equal(a, b) will all be True
# TODO: empty span


def is_right(spa, spb):
    """span a is at right of b"""
    return spa.start >= spb.end


def is_left(spa, spb):
    """span a is at left of b"""
    return spb.start >= spa.end


def is_overlap(spa, spb):
    """a and b overlaps"""
    return not is_right(spa, spb) and not is_left(spa, spb)


def is_contain(spa, spb):
    """a contains b"""
    return not is_equal(spa, spb) and spa.start <= spb.start and spa.end >= spb.end


def is_equal(spa, spb):
    """a and b are equal"""
    return spa.start == spb.start and spa.end == spb.end


def _log_overlap(span_a, span_b):
    logger.error('{} overlaps with {}'.format(span_a, span_b))


def sort_span(span_lst):
    """sort several non-overlapping span-like objects

    raise exception on overlapping"""

    span_lst = sorted(list(span_lst), key=lambda s: s.center)

    for sp1, sp2 in zip(span_lst[:-1], span_lst[1:]):
        if not is_left(sp1, sp2):
            _log_overlap(sp1, sp2)
            raise LTextSpanOverlapping()

    return span_lst


def handle_overlapping(base_span_lst, new_span_lst, raises=True):
    """
    <base_span_lst> is a ordered span lst, traverse spans in <new_span_lst> to check if it overlaps with any
    span in <base_span_lst>.

    If <raises> is set, raise exception on overlapping
    If it is not set, remove the new span
    """

    if not base_span_lst:
        for sp1 in new_span_lst:
            yield sp1

    for sp1 in new_span_lst:

        # do it by bin-search
        conflict = None

        left = 0
        if is_overlap(sp1, base_span_lst[left]):
            conflict = base_span_lst[left]

        right = len(base_span_lst) - 1
        if conflict is None and is_overlap(sp1, base_span_lst[right]):
            conflict = base_span_lst[right]

        if conflict is None and base_span_lst[left] < sp1 < base_span_lst[right]:
            while left + 1 < right:
                mid = (left + right) // 2

                if is_overlap(sp1, base_span_lst[mid]):
                    conflict = base_span_lst[mid]
                    break
                elif sp1 < base_span_lst[mid]:
                    right = mid
                elif sp1 > base_span_lst[mid]:
                    left = mid

        if not conflict:
            yield sp1

        else:
            _log_overlap(sp1, conflict)

            if raises:
                raise LTextSpanOverlapping()

            else:
                # just remove the span by not yielding it
                pass


def is_empty(sp):
    return sp.start == sp.end


class SpanV(Span):
    """A span with value"""

    def __init__(self, start, end, value):
        super(SpanV, self).__init__(start, end)
        self.value = value

        if self.length != len(self.value):
            raise ValueError('value does not match with span')

    def __repr__(self):
        return 'SpanV({}, {}, value={})'.format(self.start, self.end, repr(self.value))

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end and self.value == other.value


class SpanTrans(object):
    def __init__(self, frm, to):
        self.frm = frm
        self.to = to

        assert self.frm or self.to

    # delegate almost all properties and methods to self.frm
    # to have it act just like self.frm in relation check

    @property
    def start(self):
        return self.frm.start

    @property
    def end(self):
        return self.frm.end

    @property
    def length(self):
        return self.frm.length

    @property
    def center(self):
        return self.frm.center

    def __gt__(self, other):
        return self.frm > other

    def __lt__(self, other):
        return self.frm < other

    def __repr__(self):
        return '{} -> {}'.format(repr(self.frm), repr(self.to))

    def __iter__(self):
        # for syntax sugar
        # sp_frm, sp_to = SpanTrans(span_from, span_to)
        yield self.frm
        yield self.to

    def __bool__(self):
        return not is_empty(self)

    __nonzero__ = __bool__

    def inverse(self):
        return SpanTrans(frm=self.to, to=self.frm)


class Transform(object):
    """An operation that knows how a set of non-overlapping spans are mapping to other set of spans one by one"""

    def __init__(self, spt_lst):
        assert isinstance(spt_lst, list)
        self.spt_lst = spt_lst  # type: List[SpanTrans]

        self.spt_lst = sort_span(self.spt_lst)

    def __iter__(self):
        return iter(self.spt_lst)

    def __len__(self):
        return len(self.spt_lst)

    def __repr__(self):
        return pformat(self.spt_lst)

    def inverse(self):
        return Transform([spt.inverse() for spt in self.spt_lst])

    @classmethod
    def make(cls, span_to_v_pair_lst, text):

        def ensure_spanv(sp):
            if isinstance(sp, tuple):
                s, e = sp
                return SpanV(start=s, end=e, value=text[s:e])

            if not isinstance(sp, SpanV):
                s, e = sp.start, sp.end
                return SpanV(start=s, end=e, value=text[s:e])

        offset = 0
        spt_lst = []

        for sp, seg in span_to_v_pair_lst:
            sp = ensure_spanv(sp)

            spt = SpanTrans(
                frm=sp,
                to=SpanV(
                    start=sp.start + offset,
                    end=sp.start + offset + len(seg),
                    value=seg
                )
            )
            spt_lst.append(spt)

            offset += (len(seg) - sp.length)

        return cls(spt_lst)

    def trans_text(self, text):
        """Apply a transform operation on a text"""
        assert isinstance(self, Transform)
        assert isinstance(text, str)

        char_lst = list(text)

        for sp_frm, sp_to in reversed(list(self)):
            char_lst[sp_frm.start:sp_frm.end] = sp_to.value

        return ''.join(char_lst)

    def trans_labels(self, label_lst, raises_on_overlapping=True):
        """Apply a transform operation on several labels"""
        assert isinstance(self, Transform)

        if not self.spt_lst or not label_lst:
            return label_lst

        # check overlapping
        label_lst = handle_overlapping(self.spt_lst, new_span_lst=label_lst, raises=raises_on_overlapping)

        # Suppose that labels is sorted with center, ascending.
        # If transform span overlaps with a label, raise exception.

        new_label_lst = []

        all_span_lst = sort_span(self.spt_lst + list(label_lst))

        cur_ofs = 0
        cur_spt = None

        for spt_or_lb in all_span_lst:

            if isinstance(spt_or_lb, SpanTrans):
                cur_spt = spt_or_lb
                cur_ofs += (cur_spt.to.length - cur_spt.frm.length)

            elif isinstance(spt_or_lb, Label):
                lb = spt_or_lb

                if cur_spt and cur_spt.length and is_overlap(cur_spt, lb):
                    raise ValueError('transform overlaps with label')

                new_label_lst.append(
                    lb.translate(cur_ofs)
                )

        return new_label_lst


class Label(SpanV):
    def __init__(self, start, end, value):
        super(Label, self).__init__(start, end, value)
        assert bool(self), 'label can not be empty'

    def __repr__(self):
        return 'Label({}, {}, value={})'.format(self.start, self.end, repr(self.value))

    def translate(self, n):
        # 平移
        return Label(start=self.start + n, end=self.end + n, value=self.value)


class TestTransform(unittest.TestCase):
    def test_trans_text(self):
        trans = Transform([
            SpanTrans(frm=Span(0, 1), to=SpanV(0, 0, value='')),  # delete a char
        ])

        self.assertEqual(
            'bc',
            trans.trans_text('abc')
        )

    def test_trans_label(self):
        trans = Transform([
            SpanTrans(frm=Span(1, 2), to=SpanV(0, 0, value='')),  # delete a char
        ])

        lbs = [
            Label(0, 1, value='a'),
            Label(2, 3, value='c')
        ]
        self.assertEqual(
            [Label(0, 1, value='a'), Label(1, 2, value='c')],
            trans.trans_labels(lbs)
        )

    def test_make_trans(self):
        text = 'abcdefg'

        trans = Transform.make([
            (Span(0, 1), 'A'),  # Capitalize
            (Span(2, 3), ''),  # remove 'c'
            (Span(3, 4), 'ddd'),  # triple 'd'
            (Span(5, 7), 'xy'),  # replace 'fg' -> 'xy'
        ], text)

        self.assertEqual(
            'Abdddexy',
            trans.trans_text(text)
        )

        # test inverse
        self.assertEqual(
            'abcdefg',
            trans.inverse().trans_text('Abdddexy')
        )


class LabeledText(object):
    """Text with non-overlapping labels"""

    def __init__(self, text, label_lst=None, src_lt=None, src_trans=None):
        self.text = text  # type: str
        self.label_lst = list(
            map(
                self._ensure_label,
                label_lst or []
            )
        )  # type: List[Label]

        if src_lt and src_trans:
            self.src_lt = src_lt  # type: Optional[LabeledText]
            self.src_trans = src_trans  # type: Optional[Transform]

        else:
            self.src_lt = None
            self.src_trans = None

        self.label_lst = sort_span(self.label_lst)

    def _ensure_label(self, o):
        if isinstance(o, tuple):
            s, e = o
        elif isinstance(o, (Span, SpanTrans)):
            s, e = o.start, o.end
        elif isinstance(o, dict):
            s, e = o['start'], o['end']
        else:
            raise TypeError('unrecognized type of label')

        return Label(start=s, end=e, value=self.text[s: e])

    def add_label(self, label_lst, raises_on_overlapping=True):
        if self.src_trans:
            base_sp_lst = self.src_trans.inverse().spt_lst
            label_lst = handle_overlapping(base_sp_lst,
                                           new_span_lst=map(self._ensure_label, label_lst),
                                           raises=raises_on_overlapping)

        return LabeledText(
            text=self.text,
            label_lst=self.label_lst + list(label_lst),
            src_lt=self.src_lt,
            src_trans=self.src_trans
        )

    def equals(self, other):
        assert isinstance(other, LabeledText)

        return (
                self.text == other.text
                and all(
                    (isinstance(a, Label) and a == b)
                    for a, b in zip(self.label_lst, other.label_lst)
                )
        )

    def replace(self, old, new, count=None, raises_on_overlapping=True):
        if count is not None:
            if not isinstance(count, int):
                raise TypeError('an integer is required')
            count = count if count > 0 else None

        # locate replace spans
        span_lst = []

        cur = 0
        cnt = 0
        while True:
            try:
                pos = self.text.index(old, cur)
            except ValueError:
                break
            else:
                span_lst.append(Span(pos, pos + len(old)))
                cur = pos + len(old)

                if count and cnt >= count:
                    break

        # transform

        trans = Transform.make(span_to_v_pair_lst=[
            (sp, new) for sp in span_lst
        ], text=self.text)

        n_txt = trans.trans_text(self.text)
        n_lbs = trans.trans_labels(self.label_lst, raises_on_overlapping=raises_on_overlapping)

        return LabeledText(
            text=n_txt,
            label_lst=n_lbs,
            src_lt=self,
            src_trans=trans
        )

    def re_replace(self, pattern, repl, count=None, flags=None, raises_on_overlapping=True):
        if flags:
            seg_iter = re.finditer(pattern, self.text, flags)
        else:
            seg_iter = re.finditer(pattern, self.text)

        if callable(repl):
            repl_func = repl

        else:
            def repl_func(x):
                return repl

        _pairs = (
            (
                Span(seg.start(), seg.end()),
                repl_func(seg.group())
            ) for seg in seg_iter
        )

        if count:
            itertools.islice(_pairs, 0, count)

        trans = Transform.make(_pairs, text=self.text)

        n_txt = trans.trans_text(self.text)
        n_lbs = trans.trans_labels(self.label_lst, raises_on_overlapping=raises_on_overlapping)

        return LabeledText(
            text=n_txt,
            label_lst=n_lbs,
            src_lt=self,
            src_trans=trans
        )

    def capitalize(self):
        return replace_cc(str.capitalize)(self)

    def lower(self):
        return replace_cc(str.lower)(self)

    def upper(self):
        return replace_cc(str.upper)(self)

    def restore(self, till_end=False):
        if not (self.src_lt and self.src_trans):
            raise ValueError('no history or history deserted, unable to restore')

        def _restore_once(_lt):
            return LabeledText(
                text=_lt.src_lt.text,
                label_lst=_lt.src_trans.inverse().trans_labels(_lt.label_lst),
                src_lt=_lt.src_lt.src_lt,
                src_trans=_lt.src_lt.src_trans
            )

        lt = self

        if till_end:
            while lt.src_lt and lt.src_trans:
                lt = _restore_once(lt)

        else:
            lt = _restore_once(lt)

        return lt

    def pp(self):
        u"""Pretty Print

        text with overlapping labels are not supported
        you would see odd prints if so
        """
        from termcolor import colored

        s = list(self.text)
        for lb in reversed(self.label_lst):
            s[lb.start:lb.end] = list(colored(lb.value, 'green'))

        s = ''.join(s)
        print(s)

    @classmethod
    def literal(cls, lit, left='[', right=']'):
        u"""make labeled text in literal

        literal is not able to express overlapping labels
        """
        assert isinstance(lit, str)

        if left == right:
            raise ValueError('left pattern should not be same as right pattern')

        txt = lit
        sp_lst = []
        pos = 0

        while True:
            try:
                l_idx = txt.index(left, pos)
            except ValueError:
                break
            txt = txt.replace(left, '', 1)

            try:
                r_idx = txt.index(right, pos)
            except ValueError:
                raise ValueError('closing pattern is missing')
            txt = txt.replace(right, '', 1)

            if l_idx > r_idx:
                raise ValueError('extra border pattern in literal text')

            sp_lst.append((l_idx, r_idx))

            # look for overlapping parenthesis
            a, b = None, None  # has extra left, has extra right
            try:
                a = txt.index(left, pos, r_idx)
                b = txt.index(right, pos, r_idx)
            except ValueError:
                pass
            if not (a is None and b is None):
                raise ValueError('crossed label met')

            pos = r_idx

        return cls(txt, label_lst=sp_lst)

    def to_literal(self):
        char_lst = list(self.text)

        spot_lst = []
        for lb in self.label_lst:
            spot_lst.append((lb.start, 's'))
            spot_lst.append((lb.end, 'e'))

        s_par = '['
        e_par = ']'

        for spot, se in reversed(sorted(spot_lst)):
            if se == 's':
                char_lst[spot: spot] = [s_par]
            elif se == 'e':
                char_lst[spot: spot] = [e_par]

        return ''.join(char_lst)

    def __repr__(self):
        return '{cls}.literal({lt})'.format(cls=self.__class__.__name__, lt=repr(self.to_literal()))


def mk_trans1(txt1, txt2):
    """make Transform by diff two text with same length"""
    assert len(txt1) == len(txt2)

    spt_lst = []
    cur_sp = None

    for i, (chr1, chr2) in enumerate(zip(txt1, txt2)):
        if chr1 != chr2:

            if cur_sp is None:
                cur_sp = [i, i + 1]
            elif cur_sp[1] + 1 == i:
                cur_sp[1] = i

            else:
                raise

        if chr1 == chr2:
            if cur_sp is None:
                pass
            else:
                s, e = cur_sp
                spt_lst.append(
                    SpanTrans(
                        frm=SpanV(s, e, value=txt1[s:e]),
                        to=SpanV(s, e, value=txt2[s:e])
                    )
                )
                cur_sp = None

    return Transform(spt_lst=spt_lst)


def replace_cc(func):

    def replace(lt):
        assert isinstance(lt, LabeledText)

        txt1 = lt.text
        txt2 = func(txt1)
        assert len(txt1) == len(txt2)

        trans = mk_trans1(txt1, txt2)
        lbs2 = trans.trans_labels(lt.label_lst)

        return LabeledText(
            text=txt2,
            label_lst=lbs2,
            src_lt=lt,
            src_trans=trans
        )

    return replace


class TestLabeledText(unittest.TestCase):
    def test_replace_happy(self):
        ls = LabeledText.literal('鸟宿池边树，僧[推]月下门')

        self.assertTrue(
            ls.equals(
                ls.replace(old='踢', new='猛踢')
            )
        )

        self.assertTrue(
            ls.replace('，', ',').equals(
                LabeledText.literal('鸟宿池边树,僧[推]月下门')
            )
        )

        with self.assertRaises(LTextSpanOverlapping):
            ls.replace('推', '敲')

        with self.assertRaises(LTextSpanOverlapping):
            ls.replace('僧推月下门', '僧敲月下门')

    def test_re_replace(self):
        ls = LabeledText.literal('[r]egular    [e]xpression  [o]perations')

        self.assertTrue(
            ls.re_replace(r'\s+', ' ').equals(
                LabeledText.literal('[r]egular [e]xpression [o]perations')
            )
        )

        self.assertTrue(
            ls.re_replace(r'\s+', ' ').add_label([
                (6, 7), (17, 18), (28, 29),
            ]).restore().equals(
                LabeledText.literal('[r]egula[r]    [e]xpressio[n]  [o]peration[s]')
            )
        )

    def test_upper(self):
        lt = LabeledText.literal('a[B]c[D]e[F]g[H]i[J]k')

        self.assertTrue(
            LabeledText.literal('A[B]C[D]E[F]G[H]I[J]K').equals(
                lt.upper()
            )
        )

    def test_handle_overlapping(self):
        lt = LabeledText.literal('world wide web')

        # Add label on replaced span would alert
        lt1 = lt.replace('w', 'W')
        with self.assertRaises(LTextSpanOverlapping):
            lt1.add_label([(0, 2)])

        # replace on label would alert
        lt2 = lt.add_label([(0, 1), (6, 7)])
        with self.assertRaises(LTextSpanOverlapping):
            lt2.replace('w', 'W')

        # replace spans overlap is ok
        lt3 = lt.replace(' ', '</br>').replace('</br>', '.').add_label([(0, 1), (6, 7), (11, 12)])
        self.assertTrue(
            lt3.equals(LabeledText.literal('[w]orld.[w]ide.[w]eb'))
        )

        # suppressing alerts when replace on label:
        self.assertTrue(
            lt
                .replace('w', 'W')
                .add_label([(0, 2), (2, 4)], raises_on_overlapping=False)
                .equals(
                    LabeledText.literal('Wo[rl]d Wide Web')
                )
        )

        # suppressing alerts when add label on replaced span
        self.assertTrue(
            lt
                .add_label([(0, 2), (2, 4)])
                .replace('w', 'W', raises_on_overlapping=False)
                .equals(
                    LabeledText.literal('Wo[rl]d Wide Web')
                )
        )
