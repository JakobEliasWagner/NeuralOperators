from nos.data.helmholtz.domain_properties import AdiabaticAbsorberDescription


def test_adiabatic_absorber_description():
    absorber = AdiabaticAbsorberDescription(23.0, 11.0, 42)

    assert absorber.lambda_depth == 23.0
    assert absorber.round_trip == 11.0
    assert absorber.degree == 42
