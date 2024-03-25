from dataclasses import (
    dataclass,
)

from nos.utils import (
    dataclass_to_dict,
)


@dataclass
class Address:
    street: str
    city: str


@dataclass
class Person:
    name: str
    age: int
    address: Address


@dataclass
class Team:
    name: str
    members: list[Person]


@dataclass
class Project:
    title: str
    details: dict


def test_simple_dataclass():
    address = Address(street="Sonnenstrasse", city="Munich")
    expected = {"street": "Sonnenstrasse", "city": "Munich"}
    assert dataclass_to_dict(address) == expected


def test_nested_dataclass():
    person = Person(name="Foo", age=30, address=Address(street="Sonnenstrasse", city="Munich"))
    expected = {"name": "Foo", "age": 30, "address": {"street": "Sonnenstrasse", "city": "Munich"}}
    assert dataclass_to_dict(person) == expected


def test_dataclass_with_list():
    team = Team(
        name="Dream Team",
        members=[
            Person(name="Foo", age=30, address=Address(street="Sonnenstrasse", city="Munich")),
            Person(name="Bar", age=28, address=Address(street="Frauenhoferstrasse", city="Munich")),
        ],
    )
    expected = {
        "name": "Dream Team",
        "members": [
            {"name": "Foo", "age": 30, "address": {"street": "Sonnenstrasse", "city": "Munich"}},
            {"name": "Bar", "age": 28, "address": {"street": "Frauenhoferstrasse", "city": "Munich"}},
        ],
    }
    assert dataclass_to_dict(team) == expected


def test_dataclass_with_dict():
    project = Project(title="AI Research", details={"budget": 1, "duration": "2 hours"})
    expected = {"title": "AI Research", "details": {"budget": 1, "duration": "2 hours"}}
    assert dataclass_to_dict(project) == expected


def test_non_dataclass_object():
    non_dataclass_object = "I am not a dataclass"
    assert dataclass_to_dict(non_dataclass_object) == non_dataclass_object
