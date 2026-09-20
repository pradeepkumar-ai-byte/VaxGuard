import pytest
from vaxguard.attacks.library import AttackLibrary
from vaxguard.models.attack import AttackCategory, AttackVector


@pytest.fixture
def library():
    return AttackLibrary()


def test_library_loading(library):
    attacks = library.get_all()
    assert len(attacks) > 0
    for attack in attacks:
        assert isinstance(attack, AttackVector)
        assert attack.id
        assert attack.name
        assert attack.payload
        assert 1 <= attack.severity <= 10


def test_library_filter_by_category(library):
    jb_attacks = library.get_by_category(AttackCategory.JAILBREAK)
    assert len(jb_attacks) > 0
    for a in jb_attacks:
        assert a.category == AttackCategory.JAILBREAK


def test_library_filter_by_prompt_injection(library):
    pi_attacks = library.get_by_category(AttackCategory.PROMPT_INJECTION)
    assert len(pi_attacks) > 0
    for a in pi_attacks:
        assert a.category == AttackCategory.PROMPT_INJECTION


def test_library_get_by_id_found(library):
    first = library.get_all()[0]
    found = library.get_by_id(first.id)
    assert found is not None
    assert found.id == first.id
    assert found.name == first.name


def test_library_get_by_id_not_found(library):
    missing = library.get_by_id("non_existent_attack_id_9999")
    assert missing is None


def test_library_all_categories_present(library):
    all_attacks = library.get_all()
    categories_found = {a.category for a in all_attacks}
    assert AttackCategory.PROMPT_INJECTION in categories_found
    assert AttackCategory.JAILBREAK in categories_found
