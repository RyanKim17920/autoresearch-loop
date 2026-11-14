"""v3-c8a x c8b merge wiring: the kind bandit sees bold units; split parts and opposite arms are credited to their origin."""
from types import SimpleNamespace as NS

from loop.core import alloc as AL


def _led(nodes):
    return NS(nodes={i: NS(id=i, parent=p, notes=n, base="b") for i, p, n in nodes}, epoch="b")


def test_bold_is_an_llm_kind_the_bandit_counts():
    assert "bold" in AL.llm_kinds() and AL.group_of("bold") == "llm"
    led = _led([("r", None, ""), ("n1", "r", "brief=bold request=x")])
    assert AL.unit_of(led, "n1") == ("n1", "bold")


def test_split_part_is_its_origins_unit():
    led = _led([("r", None, ""), ("n1", "r", "brief=bold mixed=a,b"),
                ("n2", "r", "brief=bold split_of=n1 group=a"), ("n3", "n2", "brief=bold split_of=n1 group=b")])
    assert AL.unit_of(led, "n2") == AL.unit_of(led, "n1") == ("n1", "bold")
    assert AL.unit_of(led, "n3")[0] == "n1"


def test_opposite_arm_is_its_origin_arms_unit():
    led = _led([("r", None, ""), ("a", "r", "brief=program op=P arm=x stage=A program_arm=1"),
                ("o", "r", "brief=program op=P arm=x_opp stage=A program_arm=1 opposite_of=x")])
    assert AL.unit_of(led, "o") == AL.unit_of(led, "a") == ("a", "program")
