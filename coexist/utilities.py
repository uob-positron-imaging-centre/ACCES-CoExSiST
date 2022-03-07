#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utilities.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 04.08.2021


import  sys
import  signal
from    textwrap    import  indent




class UniversalSet:
    def __contains__(self, x):
        return True


universal_set = UniversalSet()




def autorepr(_c = None, *, short = {}, hide = {}):
    '''Automatically create a ``__repr__`` method for pretty-printing class
    attributes; they are discovered at runtime following some rules.

    1. Attribute names do not start with underscores and are not callable.
    2. The attributes given in ``short`` (set[str] | bool) are printed up to
       80 characters. If ``short == True``, then all attributes are shortened.
    3. The attributes given in ``hide`` (set[str]) are skipped.
    4. If the attribute representation is multiline (i.e. has newlines) then it
       is printed on a separate line and indented with 2 spaces.

    Examples
    --------
    >>> @autorepr
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

    >>> @autorepr(hide = {"x"})
    >>> class SomeClass:
    >>>     def __init__(self):
    >>>         self.x = "spam"
    >>>         self.y = "eggs"
    >>>
    >>> print(SomeClass())
    SomeClass
    ---------
    y = eggs
    '''

    def __repr__(self):
        # Skip those attributes from the representation
        if hasattr(self, "_repr_hide"):
            _repr_hide = self._repr_hide
        else:
            _repr_hide = set()

        # Shorten those attributes' representation
        if hasattr(self, "_repr_short"):
            _repr_short = self._repr_short
        else:
            _repr_short = set()

        # Return pretty string representation of an arbitrary object
        docs = []
        for att in dir(self):
            if not att.startswith("_") and att not in _repr_hide:
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

    def setrepr(c):
        # Attach the attribute names to shorten / hide as class attributes
        if isinstance(short, bool) and short is True:
            c._repr_short = universal_set
        elif len(short):
            c._repr_short = set(short)

        if len(hide):
            c._repr_hide = set(hide)

        c.__repr__ = __repr__
        return c

    if _c is None:
        return setrepr
    return setrepr(_c)




def interrupt_handler(signum, stackframe):
    li = "\n" + "*" * 80 + "\n"
    print(
        f"{li}Caught signal {signum} - will kill subprocesses and abort!{li}",
        flush = True, file = sys.stderr,
    )
    raise KeyboardInterrupt




class SignalHandlerKI:
    '''Handle typical OS termination signals by raising a ``KeyboardInterrupt``
    exception.

    If a signal is not found on a given platform (e.g. SIGBREAK only exists on
    Windows) it is simply skipped.
    '''

    def __init__(
        self,
        signals = [
            "SIGINT",
            "SIGTERM",
            "SIGBREAK",
            "SIGABRT",
        ],
    ):
        self.signals = signals
        self.previous_handlers = {}


    def set(self):
        '''Set the signals' handlers. Save the previous handlers.
        '''

        # The signal number may not exist (AttributeError) or be wrong
        # (ValueError) on some platforms, so catch possible exceptions and
        # simply ignore these signals
        for sig in self.signals:
            try:
                s = getattr(signal, sig)                       # AttributeError
                self.previous_handlers[sig] = signal.getsignal(s)
                signal.signal(s, interrupt_handler)            # Key|ValueError
            except (AttributeError, KeyError, ValueError):
                pass


    def unset(self):
        '''Unset the signals' handlers. Return to previous handlers.
        '''

        # The signal number may not exist (AttributeError) or be wrong
        # (ValueError) on some platforms, so catch possible exceptions and
        # simply ignore these signals
        for sig in self.signals:
            try:
                s = getattr(signal, sig)                       # AttributeError
                signal.signal(s, self.previous_handlers[sig])  # Key|ValueError
            except (AttributeError, KeyError, ValueError):
                pass
