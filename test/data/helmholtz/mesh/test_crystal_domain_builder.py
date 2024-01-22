import gmsh

from nos.data.helmholtz.mesh import CrystalDomainBuilder, CShapedCrystalDomainBuilder, CylindricalCrystalDomainBuilder


def test_cylindrical_crystal_domain_builder(cylindrical_descriptions):
    description = cylindrical_descriptions[0]
    gmsh.initialize()
    gmsh.model.add("test cylindrical domain")
    b = CylindricalCrystalDomainBuilder(description)
    assert len(b.build()) == 1
    gmsh.finalize()


def test_c_shaped_crystal_domain_builder(c_shaped_descriptions):
    description = c_shaped_descriptions[0]
    gmsh.initialize()
    gmsh.model.add("test c-shaped domain")
    b = CShapedCrystalDomainBuilder(description)
    assert len(b.build()) == 1
    gmsh.finalize()


def test_none_crystal_domain_builder(none_descriptions):
    description = none_descriptions[0]
    gmsh.initialize()
    gmsh.model.add("test none domain")
    b = CrystalDomainBuilder(description)
    assert len(b.build()) == 1
    gmsh.finalize()
