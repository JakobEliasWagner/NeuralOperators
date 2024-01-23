import nos.data.helmholtz.domain_properties as d


def test_create_cylinder_d_instance():
    c = d.CylinderDescription(0.22, 19, 42.42)
    assert c.grid_size == 0.22
    assert c.n == 19
    assert c.radius == 42.42


def test_create_c_shape_d_instance():
    c = d.CShapeDescription(123.23, 12, 1.22, 0.9, 0.5)
    assert c.grid_size == 123.23
    assert c.n == 12
    assert c.radius == 1.22
    assert c.inner_radius == 0.9
    assert c.gap_width == 0.5


def test_create_none_d_instance():
    c = d.NoneDescription(0.22, 12)
    assert c.grid_size == 0.22
    assert c.n == 12
