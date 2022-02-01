#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utilities.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 04.08.2021


from textwrap import indent




def autorepr(c):
    '''Automatically create a `__repr__` method for pretty-printing class
    attributes; they are discovered at runtime following some rules:

    1. Attribute names do not start with underscores and are not callable.
    2. If the class attribute `_repr_hidden` (set[str]) is defined, those
       attribute names are skipped.
    3. If the class attribute `_repr_short` (set[str]) is defined, those
       attributes' representations are shortened to 80 characters.
    4. If the attribute representation is multiline (i.e. has newlines) then it
       is printed on a separate line and indented with 2 spaces.

    Example:

    >>> @auto_repr
    >>> class SomeClass:
    >>>     def __init__(self):
    >>>         self.x = "spam"
    >>>         self.y = "eggs"
    >>>
    >>> print(SomeClass())
    SomeClass
    ---------
    x = spam
    y = eggs

    '''

    def __repr__(self):
        # Skip those attributes from the representation
        if hasattr(self, "_repr_hidden"):
            _repr_hidden = set(self._repr_hidden)
        else:
            _repr_hidden = set()

        # Shorten those attributes' representation
        if hasattr(self, "_repr_short"):
            _repr_short = set(self._repr_short)
        else:
            _repr_short = set()

        # Return pretty string representation of an arbitrary object
        docs = []
        for att in dir(self):
            if not att.startswith("_") and att not in _repr_hidden:
                memb = getattr(self, att)
                if not callable(memb):
                    # If memb_str is multiline, indent it
                    memb_str = str(memb)
                    if att not in _repr_short and "\n" in memb_str:
                        memb_str = "\n" + indent(memb_str, "  ")

                    docs.append(f"{att} = {memb_str}")
                    if att in _repr_short and len(docs[-1]) > 80:
                        docs[-1] = docs[-1].replace("\n", "\\n")[:67] + "..."

        name = self.__class__.__name__
        underline = "-" * len(name)
        return f"{name}\n{underline}\n" + "\n".join(docs)

    c.__repr__ = __repr__
    return c
