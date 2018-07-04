class DecisionFork:
    """A fork of a decision tree holds an attribute to test, and a dict
    of branches, one for each of the attribute's values."""

    def __init__(self, attr, attrname=None, default_child=None, branches=None):
        """Initialize by saying what attribute this node tests."""
        self.attr = attr
        self.attrname = attrname or attr
        self.default_child = default_child
        self.branches = branches or {}

    def __call__(self, example):
        """Given an example, classify it using the attribute and the branches."""
        attrvalue = example[self.attr]
        if attrvalue in self.branches:
            return self.branches[attrvalue](example)
        else:
            # return default class when attribute is unknown
            return self.default_child(example)

    def add(self, val, subtree):
        """Add a branch.  If self.attr = val, go to the given subtree."""
        self.branches[val] = subtree

    def display(self, indent=0):
        name = self.attrname
        print('Test', name)
        for (val, subtree) in self.branches.items():
            print(' ' * 4 * indent, name, '=', val, '==>', end=' ')
            subtree.display(indent + 1)
        print()   # newline

    def __repr__(self):
        return ('DecisionFork({0!r}, {1!r}, {2!r})'
                .format(self.attr, self.attrname, self.branches))