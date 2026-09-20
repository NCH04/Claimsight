"""Tests de la phrase de synthèse."""

from claimsight.pipeline.summary import make_summary

FRONT = [{"view": "front-left", "confidence": 0.9}]


def part(name: str, damage: str = "dent", severity: str = "moderate") -> dict:
    return {"name": name, "damage": damage, "severity": severity}


def test_sans_prediction_exploitable():
    assert make_summary([], "unknown", "unknown") == "Insufficient evidence to assess damage."


def test_vehicule_intact():
    assert make_summary(FRONT, "none", "none") == "No visible damage detected."


def test_une_piece_est_nommee_plutot_que_la_vue():
    s = make_summary(FRONT, "dent", "moderate", [part("front bumper")])
    assert s == "Front bumper shows moderate dent damage."


def test_deux_pieces_sont_coordonnees():
    s = make_summary(FRONT, "dent", "moderate", [part("front bumper"), part("hood")])
    assert s == "Front bumper and hood show moderate dent damage."


def test_au_dela_de_deux_pieces_on_compte_le_reste():
    parts = [part(n, severity="severe") for n in
             ("front bumper", "hood", "windshield", "headlights")]
    s = make_summary(FRONT, "dent", "severe", parts)
    assert s == "Front bumper, hood and 2 other parts show severe dent damage."


def test_une_seule_piece_restante_reste_au_singulier():
    parts = [part(n) for n in ("front bumper", "hood", "windshield")]
    assert "1 other part show" in make_summary(FRONT, "dent", "minor", parts)


def test_repli_sur_la_vue_sans_detecteur_de_pieces():
    s = make_summary(FRONT, "scratch", "minor", [])
    assert s == "Front-Left view shows minor scratch damage."


def test_gravite_inconnue_n_est_pas_citee():
    """Un modèle de gravité absent ne doit pas produire «unknown dent damage»."""
    s = make_summary(FRONT, "dent", "unknown", [part("hood", severity="unknown")])
    assert s == "Hood shows dent damage."
    assert make_summary(FRONT, "scratch", "unknown", []) == "Front-Left view shows scratch damage."


def test_le_constat_decrit_la_piece_citee_pas_l_image_entiere():
    """Nommer une pièce avec le dégât d'une autre est pire que ne rien nommer.

    Le verdict image-entière et le verdict sur crop peuvent diverger: c'est le
    second qui qualifie la pièce, puisque c'est de lui qu'on tient son nom.
    """
    parts = [part("windshield", damage="deformation_impact", severity="severe")]
    s = make_summary(FRONT, "broken_glass", "minor", parts)
    assert s == "Windshield shows severe deformation_impact damage."


def test_la_pire_piece_donne_le_ton():
    """`aggregate_parts` trie du plus grave au moins grave: la 1re fait foi."""
    parts = [part("hood", damage="dent", severity="severe"),
             part("front bumper", damage="scratch", severity="minor")]
    s = make_summary(FRONT, "dent", "severe", parts)
    assert s == "Hood and front bumper show severe dent damage."
