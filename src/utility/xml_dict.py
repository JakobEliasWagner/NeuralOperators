import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from typing import List, Union


class XmlDict(dict):
    def __init__(self, node: ET.Element):
        super().__init__()

        # update values for dict from current node
        self.update(node.attrib)
        # check for text
        if node.text:
            txt = node.text.strip()
            if txt:
                # empty strings do not need to be saved
                self["text"] = txt
        if node.tag:
            self["tag"] = node.tag
            # helps identify unnamed elements
            if "Name" not in self:
                self["Name"] = node.tag

        children = [XmlDict(child) for child in node]

        # update dict if possible
        children = self.insert_unique(children)

        # assign, according to names
        names = defaultdict(list)
        unnamed = []
        for child in children:
            try:
                names[child["Name"]].append(child)
            except KeyError:
                # the child has no name
                unnamed.append(child)

        # named dicts
        for child_name, child_list in names.items():
            if len(child_list) == 1:
                self[child_name] = child_list[0]
                continue
            # more than one child with the same name
            self[child_name] = child_list

        # only unnamed children remain
        children = unnamed

        # not overlapping anymore
        children = self.insert_unique(children)

        # still overlapping
        if children:
            self["unnamed"] = children

    def insert_unique(self, others: List["XmlDict"]) -> List:
        """Inserts items from lower level dict from list into this one when keys are unique for both.

        Args:
            others: list of other dictionaries.

        Returns: items from other that could not be inserted uniquely.
        """
        counter = self.count_keys(others)
        remaining = []
        for o in others:
            o_counter = Counter(list(o.keys()))

            diff = Counter({key: counter.get(key, 0) - value for key, value in o_counter.items()})
            if all([val == 0 for val in diff.values()]):
                # only has unique attributes including node
                self.update(o)
                continue

            # attributes could not be inserted unique
            remaining.append(o)

        return remaining

    def count_keys(self, others: Union[List, ET.Element]) -> Counter:
        """Counts the number of keys in both this dict and others.

        Args:
            others: List or ET.Element.

        Returns: Counter with keys from self and all elements on list
        """
        keys = list(self.keys())
        for o in others:
            keys.extend(list(o.keys()))
        return Counter(keys)
