from clb.dataprep import filter_eval, split_eval, clean_train, overlap_report, plain_question

P = {"p1": "The Germania sailed from Bremerhaven on 15 June 1869 under Captain Koldewey."}


def test_plain():
    assert plain_question("Who commanded the Germania on its voyage in 1869?")
    assert not plain_question("According to the text, who commanded the ship?")
    assert not plain_question("Who?")


def test_filter_eval_answerability():
    c = [dict(pid="p1", q="Who commanded the Germania on its voyage in 1869?", a="Koldewey", verify="Captain Koldewey"),
         dict(pid="p1", q="Who commanded the Germania on its voyage in 1870?", a="Hegemann", verify="Hegemann"),  # not in passage
         dict(pid="p1", q="Where did the Germania sail from in June 1869?", a="Bremerhaven", verify="I don't know")]  # teacher fails
    k = filter_eval(c, P)
    assert [x["a"] for x in k] == ["Koldewey"]


def test_split_disjoint():
    items = [dict(pid=f"p{i % 5}", q=f"q{i}", a="a") for i in range(60)]
    dev, test = split_eval(items, 10, 30)
    assert not {x["q"] for x in dev} & {x["q"] for x in test} and len(dev) == 10 and len(test) == 30


def test_clean_train_zero_overlap():
    ev = [dict(pid="p1", q="Who commanded the Germania on its voyage in 1869 from Bremerhaven?", a="Koldewey")]
    qa = [dict(q="Who commanded the Germania on its voyage in 1869 from Bremerhaven?", a="Koldewey"),
          dict(q="Which captain led the German ship?", a="Koldewey")]
    notes = [dict(text="Koldewey commanded the Germania on its voyage in 1869 from Bremerhaven harbour."), dict(text="The Hansa was crushed by ice in October 1869.")]
    q2, n2 = clean_train(qa, notes, ev)
    assert len(q2) == 1 and len(n2) == 1 and overlap_report(ev, q2, n2) == 0


def test_specific_rejects_vague():
    assert not plain_question("Where were the ships being prepared at the time?")
    assert plain_question("Where were the Alert and Discovery being prepared in 1875?")


def test_parse_pretty_printed_json():
    from clb.llm import parse_json_lines
    t = '{\n"q": "a?",\n"a": "b"\n}\nnoise {"q": "c?", "a": "d"},\n{broken'
    assert [d["a"] for d in parse_json_lines(t)] == ["b", "d"]


def test_needs_rewrite_only_vague_but_answerable_and_filter_drops_leaked_answer():
    from clb.dataprep import needs_rewrite
    ps = {0: "In 1875 the Alert and the Discovery lay in Portsmouth Dockyard."}
    vague = dict(pid=0, q="Where were the ships located?", a="Portsmouth Dockyard", verify="In Portsmouth Dockyard.")
    good = dict(pid=0, q="Where did the Alert and Discovery lie in 1875?", a="Portsmouth Dockyard", verify="Portsmouth Dockyard")
    wrong = dict(vague, verify="In London.")
    assert needs_rewrite(vague, ps) and not needs_rewrite(good, ps) and not needs_rewrite(wrong, ps)
    leak = dict(pid=0, q="Where did the Alert and Discovery lie in Portsmouth Dockyard in 1875?", a="Portsmouth Dockyard", verify="Portsmouth Dockyard")
    assert [k["q"] for k in filter_eval([leak, good], ps)] == [good["q"]]


def test_quoted_ship_names_count_as_anchors():
    assert plain_question('How thick was the teak sheathing on the "Alert" in 1875?')


def test_one_name_is_enough_but_vague_questions_still_rejected():
    assert plain_question('How thick was the teak sheathing at the bow of the "Alert"?')
    assert not plain_question("Where were the ships located at that time?")
