import nos.data.helmholtz.domain_properties as d


def test_create_cylinder_d_instance():
    c = d.CylinderDescription("Cylinder", 0.22, 19, 2, 42.42)
    assert c.type_name == "Cylinder"
    assert c.grid_size == 0.22
    assert c.n_x == 19
    assert c.n_y == 2
    assert c.radius == 42.42


def test_create_c_shape_d_instance():
    c = d.CShapeDescription("C-shaped", 123.23, 12, 23, 1.22, 0.9, 0.5)
    assert c.type_name == "C-shaped"
    assert c.grid_size == 123.23
    assert c.n_x == 12
    assert c.n_y == 23
    assert c.outer_radius == 1.22
    assert c.inner_radius == 0.9
    assert c.gap_width == 0.5


def test_create_none_d_instance():
    c = d.NoneDescription("None", 0.22, 12, 2)
    assert c.type_name == "None"
    assert c.grid_size == 0.22
    assert c.n_x == 12
    assert c.n_y == 2
