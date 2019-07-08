import unittest

from ltext import Transform, Span, SpanV, TransformCC, LabeledText, LTextSpanOverlapping, SpanTrans


class TestTransform(unittest.TestCase):
    def test_trans_text(self):
        trans = Transform([
            SpanTrans(frm=Span(0, 1), to=SpanV(0, 0, value='')),  # delete a char
        ])

        self.assertEqual(
            'bc',
            trans.trans_text('abc')
        )

    def test_trans_spans(self):
        trans = Transform([
            SpanTrans(frm=Span(1, 2), to=SpanV(0, 0, value='')),  # delete a char
        ])

        lbs = [
            Span(0, 1),
            Span(2, 3)
        ]
        self.assertEqual(
            [Span(0, 1), Span(1, 2)],
            trans.trans_spans(lbs)
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


class TestTransformCC(unittest.TestCase):
    def test_trans_text2(self):
        t1 = 'aBcDeFgHiJk'
        t2 = 'ABCDEFGHIJK'

        trans = TransformCC.make_by_diff(t1, t2)

        self.assertEqual(
            trans.trans_text(t1), t2
        )


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

    def test_upper_overlapping(self):
        lt = LabeledText.literal('[aBcD]e[F]g[H]i[J]k')

        self.assertTrue(
            LabeledText.literal('[ABCD]E[F]G[H]I[J]K').equals(
                lt.upper()
            )
        )

        self.assertTrue(
            lt.upper().restore().equals(lt)
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
