import gmsh

from nos.data.helmholtz.mesh import CrystalBuilder, CShapedCrystalBuilder, CylindricalCrystalBuilder


def test_base_crystal_builder_empty(descriptions):
    for description in descriptions:
        cb = CrystalBuilder(description)
        assert len(cb.build()) == 0


def test_cylindrical_builder(cylindrical_descriptions):
    description = cylindrical_descriptions[0]
    cb = CylindricalCrystalBuilder(description)
    gmsh.initialize()
    gmsh.model.add("test_cylindrical_builder")
    n_cylinders = description.crystal_description.n_x * description.crystal_description.n_y
    assert len(cb.build()) == n_cylinders
    gmsh.finalize()


def test_c_shaped_builder(c_shaped_descriptions):
    description = c_shaped_descriptions[0]
    cb = CShapedCrystalBuilder(description)
    gmsh.initialize()
    gmsh.model.add("test_c_shaped_builder")
    n_cylinders = description.crystal_description.n_x * description.crystal_description.n_y
    assert len(cb.build()) == n_cylinders
    gmsh.finalize()
