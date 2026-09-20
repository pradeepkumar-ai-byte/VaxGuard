from vaxguard.attacks.library import AttackLibrary
from vaxguard.models.attack import AttackCategory

def test_library_loading():
    # Ensure it loads without validation errors
    library = AttackLibrary()
    attacks = library.get_all()
    
    assert len(attacks) > 0
    
    # Test filtering by enum
    jb_attacks = library.get_by_category(AttackCategory.JAILBREAK)
    assert len(jb_attacks) > 0
    assert jb_attacks[0].category == AttackCategory.JAILBREAK
