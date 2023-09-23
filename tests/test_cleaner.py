from pathlib import Path

import pytest

import lcc.cleaner

# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cleaner(resource_path: Path) -> lcc.cleaner.SentenceCleaner:
    dn_rules = resource_path / "cleaner"
    text_type = None
    lang_code = None
    fn_replacements = "StringReplacements.list"

    sentencecleaner = lcc.cleaner.SentenceCleaner(
        str(dn_rules),
        text_type=text_type,
        lang_code=lang_code,
        fn_replacements=fn_replacements,
    )

    return sentencecleaner


# ---------------------------------------------------------------------------


def test_cleaner_drop_rules(resource_path: Path):
    dn_rules = resource_path / "cleaner"
    text_type = None
    lang_code = None
    fn_replacements = "StringReplacements.list"

    # default
    sentencecleaner = lcc.cleaner.SentenceCleaner(
        str(dn_rules),
        text_type=text_type,
        lang_code=lang_code,
        fn_replacements=fn_replacements,
    )
    assert len(sentencecleaner.filters) == 39

    # drop/disable two rules
    lang_code = "chr"
    sentencecleaner = lcc.cleaner.SentenceCleaner(
        str(dn_rules),
        text_type=text_type,
        lang_code=lang_code,
        fn_replacements=fn_replacements,
    )
    assert len(sentencecleaner.filters) == 37

    # overwrite rule
    lang_code = "fas"
    sentencecleaner = lcc.cleaner.SentenceCleaner(
        str(dn_rules),
        text_type=text_type,
        lang_code=lang_code,
        fn_replacements=fn_replacements,
    )
    assert len(sentencecleaner.filters) == 39

    # add new rules
    lang_code = "ces"
    sentencecleaner = lcc.cleaner.SentenceCleaner(
        str(dn_rules),
        text_type=text_type,
        lang_code=lang_code,
        fn_replacements=fn_replacements,
    )
    assert len(sentencecleaner.filters) == 40


# ---------------------------------------------------------------------------


def test_general_rules(cleaner: lcc.cleaner.SentenceCleaner):
    # inputfile / outputfile
    do_replacements = False

    sentences_input = [
        """Bei diesem Satz sollte keine Regel anschlagen.""",
        """Das ist ein Satz mit	Tab.""",
        """Das ist ein Satz mit  zwei Leerzeichen!""",
        """Das is,t,,, ei,n Sa,tz mit ,vi,,,,elen Kommata.""",
        """Das.. ist.. ei..n Sat..z mit vie..len Punkt..en..""",
        """D a s i s t e i n S a t z m i t v i e l e n L e e r z e i c he n.""",
        """Ein Satz ohne richtiges Satzendesymbol""",
        """D a s i s t e i n S a t z m i t v i e l e n L e e r z e i c he n.""",
        """Das ist gut.""",
        """Ich bin <a href="www.testdomain.de">aaa</> hier zu finden.""",
        """Hier oder <tr> hier.""",
        """Dies ist ein langer, ein sehr langer, ein äußerst langer, unfassbar langer Satz, der ziemlich, ja außerordentlich, geradezu übermässig lang ist und bei dieser Mordslänge einfach nicht enden will, aber das irgendwann doch einmal tun muss, zumindest sollte man das hoffen, denn ewig habe ich auch keine Zeit und eigentlich dachte ich am Anfang dass dies ein Satz wie jeder andere sein würde, bis ich im Laufe der Zeit merkte, dass diese alberne Satzlänge mehr Zeit verschlingt als ich bereit war zu opfern und eigentlich wollte ich ja noch einkaufen und joggen, naja zumindest einkaufen, nur Brot und Bier, ok eigentlich nur Bier, ich gebe es ja zu, dummes Brot kriege ich ja überall aber ordentliche Bier will erstmal gefunden werden, das fallen mir Geschichten ein, oh mann da stehen einem die Haar zu Berge, will man kaum glauben und doch wahr, wo war ich gleich, ach ja, dieser nervige Satz, der kein Ende finden will und mich am Brotkauf hindert, wo meine armen kleinen Kinder zu Hause laut Schreien "Papi, Papi!, wo ist das Bier?" ähhhh "Wo ist das Brot meine ich natürlich, das mit dem Bier kriegen sie in meiner Familie erst mit 12 mit.Dies ist ein langer, ein sehr langer, ein äußerst langer, unfassbar langer Satz, der ziemlich, ja außerordentlich, geradezu übermässig lang ist und bei dieser Mordslänge einfach nicht enden will, aber das irgendwann doch einmal tun muss, zumindest sollte man das hoffen, denn ewig habe ich auch keine Zeit und eigentlich dachte ich am Anfang dass dies ein Satz wie jeder andere sein würde, bis ich im Laufe der Zeit merkte, dass diese alberne Satzlänge mehr Zeit verschlingt als ich bereit war zu opfern und eigentlich wollte ich ja noch einkaufen und joggen, naja zumindest einkaufen, nur Brot und Bier, ok eigentlich nur Bier, ich gebe es ja zu, dummes Brot kriege ich ja überall aber ordentliche Bier will erstmal gefunden werden, das fallen mir Geschichten ein, oh mann da stehen einem die Haar zu Berge, will man kaum glauben und doch wahr, wo war ich gleich, ach ja, dieser nervige Satz, der kein Ende finden will und mich am Brotkauf hindert, wo meine armen kleinen Kinder zu Hause laut Schreien "Papi, Papi!, wo ist das Bier?" ähhhh "Wo ist das Brot meine ich natürlich, das mit dem Bier kriegen sie in meiner Familie erst mit 12 mit.""",
        """Ich brauch mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehr Geld!""",
        """&& & & /&::&AAAAAAAAAAAAAAAA.""",
        """A<<.""",
        """A32532532525 3255235325 32524242323.""",
        """AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA.""",
        """A5,5,5,5,5,5,5,5.""",
        """Das Auto ist rot aber es ist nicht blau und Das Auto ist rot aber es ist nicht blau unDas Auto ist rot aber es ist nicht blau undas ist gut so, denn es sollte ja rot werden. . .""",
        """Das kann ja wohl nicht sein??!!""",
        """Ich Und DU Werden Das Problem Schon Lösen Können.""",
        """Das ist gut , nicht schlecht.""",
        """Das ist in Reutl -ing.""",
        """Andritz Va Tech Hydro (Autriche) et Metso (Finlande) - qui ont conclu la transaction avec GE Hydro.""",
        """Das sind spitze Klammern: <<.""",
        """Das Auto ist  rot.""",
        """|Das ist ein Testsatz.""",
        """§Das ist ein Testsatz.""",
        """[Das ist ein Testsatz.""",
        """#Das ist ein Testsatz.""",
        """*Das ist ein Testsatz.""",
        """:Das ist ein Testsatz.""",
        """Du erreichst mich unter me@fubar.de nbsp; unter http://fubar.de.""",
        """Hier sind Geodaten: Mapy: 50° 4' 50" s. š., 13° 12' 58" v. d..""",
        """@@Quelle@@su/a/n/a/Anatomi.html Gambar anatomis otot manusa na Encyclopédie.""",
        """639166534368 eow pu sa mga bez fwendz ko na cla juliene, leah, kristel, jenyne and of coUrSe, minelle. mga tol, c kriStiEn tOh, tnX aBaNtE!""",
        """Darin findet man einen Übersichtsplan aller Parkmöglichkeiten in der Lübecker Altstadt und ihrer unmittelbaren Umgebung samt Preis­informationen, sowie die aktuelle Liste der zahlreichen teilnehmenden Einzelhändler, Dienstleistern und Gastronomen.""",
        """Das ist ein Satz mit ….""",
        """Das ist ein Satz mit --- vielen Sonderzeichen.""",
        """Das ist ein Satz mit _.""",
        """Das iPhone entpuppte sich mehr und mehr als Desaster.""",
        """Das ist ein Satz mit _ Zeichen.""",
        """Das ist ein Satz --- mit Zeichen.""",
        """Das ist ein Satz mit .. Zeichen.""",
        """Das ist ein Satz mit ¿½ Encodingproblemen.""",
        """Das ist ein Satz mit weiteren Ã© Encodingproblemen.""",
        """Das ist noch ein Satz mit � Encodingproblemen.""",
        """Das ist ein Satz mit merkwürdigen ?zeichenkombinationen.""",
        """Das ist noch ein Satz mit merkwürdigen ;zeichenkombinationen.""",
        """Das ist ein Satz mit merkwürdigen ** Zeichen.""",
        """Das ist noch ein Satz mit merkwürdigen ~~ Zeichen.""",
        """Das ist noch ein weiterer Satz mit merkwürdigen @@ Zeichen.""",
        """Das ist ein Satz mit weiteren kaputten ď»ż Zeichen.""",
        """Das ist ein Satz mit noch mal weiteren kaputten ďż˝ Zeichen.""",
        """Diesen Punkt • wollen wir hier auch nicht sehen.""",
        """Matthias Kramer)""",
        """Alles normale wird lächerlich gemacht. schade!""",
        """Alles nur Ablenkung von den wichtigen Themen, wie die Flüchtlinge die über unser Land herfallen wie eine Heuschreckenplage. es ist ja viel Halbwissen hier unterwegs.""",
        """Alle Sponsoren sagten zu, die Aufwendungen nachzubessern, zudem werden die 17 Gesellschafter weitere Zahlungen leisten. sid Der Schriftsteller Herbert Rosendorfer erhält den mit 10 000 Euro dotierten Münchner Literaturpreis 2005.""",
        """Das ist ein Satz, der mit zwei Punkten endet..""",
        """15. Dieser Satz beginnt mit Ziffern.""",
    ]

    sentences_output = [
        """Bei diesem Satz sollte keine Regel anschlagen.""",
        """Hier oder <tr> hier.""",
        """Andritz Va Tech Hydro (Autriche) et Metso (Finlande) - qui ont conclu la transaction avec GE Hydro.""",
        """Das ist ein Satz mit ….""",
        """Das ist ein Satz mit --- vielen Sonderzeichen.""",
        """Das iPhone entpuppte sich mehr und mehr als Desaster.""",
        """Das ist ein Satz --- mit Zeichen.""",
        """Das ist ein Satz mit ¿½ Encodingproblemen.""",
        """Das ist noch ein Satz mit merkwürdigen ;zeichenkombinationen.""",
        """Das ist ein Satz mit merkwürdigen ** Zeichen.""",
        """Das ist noch ein Satz mit merkwürdigen ~~ Zeichen.""",
        """Das ist ein Satz mit weiteren kaputten ď»ż Zeichen.""",
        """Das ist ein Satz mit noch mal weiteren kaputten ďż˝ Zeichen.""",
        """Diesen Punkt • wollen wir hier auch nicht sehen.""",
    ]
    sentences_output_idx = [0, 10, 23, 37, 38, 40, 42, 44, 48, 49, 50, 52, 53, 54]

    for sidx, sentence in enumerate(sentences_input):
        sentence_out = (
            sentences_output[sentences_output_idx.index(sidx)]
            if sidx in sentences_output_idx
            else None
        )

        result = cleaner.filter_sentence(sentence, do_replacements=do_replacements)
        assert result == sentence_out, f"sentence example: {sidx}"


def test_general_rules_2(cleaner: lcc.cleaner.SentenceCleaner):
    # inputfile_raw / outputfile_raw
    do_replacements = True

    sentences_input = [
        """Welt-Finanzgipfel Regierungschefs suchen Weg aus der Krise""",
        """Welt-Finanzgipfel Regierungschefs suchen Weg aus der Krise.""",
        """Das Auto ist groß.""",
        """Das Auto ist groß...""",
        """(Oh ja!)""",
        """Das ist ein Satz mit	Tab.""",
        """Das ist ein Satz mit  zwei Leerzeichen!""",
        """Das is,t,,, ei,n Sa,tz mit ,vi,,,,elen Kommata.""",
        """Das.. ist.. ei..n Sat..z mit vie..len Punkt..en..""",
        """D a s i s t e i n S a t z m i t v i e l e n L e e r z e i c he n.""",
        """Ein Satz ohne richtiges Satzendesymbol""",
        """D a s i s t e i n S a t z m i t v i e l e n L e e r z e i c he n.""",
        """Das ist gut.""",
        """Ich bin <a href="www.testdomain.de">aaa</> hier zu finden.""",
        """Hier oder <tr> hier.""",
        """Dies ist ein langer, ein sehr langer, ein äußerst langer, unfassbar langer Satz, der ziemlich, ja außerordentlich, geradezu übermässig lang ist und bei dieser Mordslänge einfach nicht enden will, aber das irgendwann doch einmal tun muss, zumindest sollte man das hoffen, denn ewig habe ich auch keine Zeit und eigentlich dachte ich am Anfang dass dies ein Satz wie jeder andere sein würde, bis ich im Laufe der Zeit merkte, dass diese alberne Satzlänge mehr Zeit verschlingt als ich bereit war zu opfern und eigentlich wollte ich ja noch einkaufen und joggen, naja zumindest einkaufen, nur Brot und Bier, ok eigentlich nur Bier, ich gebe es ja zu, dummes Brot kriege ich ja überall aber ordentliche Bier will erstmal gefunden werden, das fallen mir Geschichten ein, oh mann da stehen einem die Haar zu Berge, will man kaum glauben und doch wahr, wo war ich gleich, ach ja, dieser nervige Satz, der kein Ende finden will und mich am Brotkauf hindert, wo meine armen kleinen Kinder zu Hause laut Schreien "Papi, Papi!, wo ist das Bier?" ähhhh "Wo ist das Brot meine ich natürlich, das mit dem Bier kriegen sie in meiner Familie erst mit 12 mit.Dies ist ein langer, ein sehr langer, ein äußerst langer, unfassbar langer Satz, der ziemlich, ja außerordentlich, geradezu übermässig lang ist und bei dieser Mordslänge einfach nicht enden will, aber das irgendwann doch einmal tun muss, zumindest sollte man das hoffen, denn ewig habe ich auch keine Zeit und eigentlich dachte ich am Anfang dass dies ein Satz wie jeder andere sein würde, bis ich im Laufe der Zeit merkte, dass diese alberne Satzlänge mehr Zeit verschlingt als ich bereit war zu opfern und eigentlich wollte ich ja noch einkaufen und joggen, naja zumindest einkaufen, nur Brot und Bier, ok eigentlich nur Bier, ich gebe es ja zu, dummes Brot kriege ich ja überall aber ordentliche Bier will erstmal gefunden werden, das fallen mir Geschichten ein, oh mann da stehen einem die Haar zu Berge, will man kaum glauben und doch wahr, wo war ich gleich, ach ja, dieser nervige Satz, der kein Ende finden will und mich am Brotkauf hindert, wo meine armen kleinen Kinder zu Hause laut Schreien "Papi, Papi!, wo ist das Bier?" ähhhh "Wo ist das Brot meine ich natürlich, das mit dem Bier kriegen sie in meiner Familie erst mit 12 mit.""",
        """Ich brauch mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeldmehrGeld mehr Geld!""",
        """&& & & /&::&AAAAAAAAAAAAAAAA.""",
        """A<<.""",
        """A32532532525 3255235325 32524242323.""",
        """AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA.""",
        """A5,5,5,5,5,5,5,5.""",
        """Das Auto ist rot aber es ist nicht blau und Das Auto ist rot aber es ist nicht blau unDas Auto ist rot aber es ist nicht blau undas ist gut so, denn es sollte ja rot werden. . .""",
        """Das kann ja wohl nicht sein??!!""",
        """Ich Und DU Werden Das Problem Schon Lösen Können.""",
        """Das ist gut , nicht schlecht.""",
        """Das ist in Reutl -ing.""",
        """Andritz Va Tech Hydro (Autriche) et Metso (Finlande) - qui ont conclu la transaction avec GE Hydro.""",
        """Das sind spitze Klammern: <<.""",
        """Das Auto ist  rot und dieser Satz hat einen Kodierungsfehler.""",
        """|Das ist ein Testsatz.""",
        """§Das ist ein Testsatz.""",
        """[Das ist ein Testsatz.""",
        """#Das ist ein Testsatz.""",
        """*Das ist ein Testsatz.""",
        """:Das ist ein Testsatz.""",
        """Das ist ein Testsatz mit HTML entity &alpha;.""",
        """Das hier soll ein tschechischer Satz sein, wegen den folgenden Buchstaben: éěšč.""",
        """Dieser Satz enthält mathematischen Kram: 2*2*2*2=16.""",
        """Dieser Satz dient zum spontanen Testen,A.""",
    ]

    sentences_output = [
        """Welt-Finanzgipfel Regierungschefs suchen Weg aus der Krise.""",
        """Das Auto ist groß.""",
        """Hier oder <tr> hier.""",
        """Andritz Va Tech Hydro (Autriche) et Metso (Finlande) - qui ont conclu la transaction avec GE Hydro.""",
        """Das ist ein Testsatz mit HTML entity α.""",
        """Das hier soll ein tschechischer Satz sein, wegen den folgenden Buchstaben: éěšč.""",
    ]
    sentences_output_idx = [1, 2, 14, 27, 36, 37]

    for sidx, sentence in enumerate(sentences_input):
        sentence_out = (
            sentences_output[sentences_output_idx.index(sidx)]
            if sidx in sentences_output_idx
            else None
        )

        result = cleaner.filter_sentence(sentence, do_replacements=do_replacements)
        assert result == sentence_out, f"sentence example: {sidx}"


def test_replace_entities(cleaner: lcc.cleaner.SentenceCleaner):
    do_replacements = True

    sentences_input = [
        """Das ist ein Testsatz mit HTML entity &alpha;.""",
    ]
    sentences_output = [
        """Das ist ein Testsatz mit HTML entity α.""",
    ]
    sentences_output_idx = [1, 2, 14, 27, 36, 37]

    for sentence, sentence_out in zip(sentences_input, sentences_output):
        result = cleaner.filter_sentence(sentence, do_replacements=do_replacements)
        assert result == sentence_out


# ---------------------------------------------------------------------------
