from __future__ import division, with_statement

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from floopy.fortran.diagnostic import TranslationError
import numpy as np


class Scope(object):
    def __init__(self, subprogram_name, arg_names=set()):
        self.subprogram_name = subprogram_name

        # map name to data
        self.data_statements = {}

        # map first letter to type
        self.implicit_types = {}

        # map name to dim tuple
        self.dim_map = {}

        # map name to dim tuple
        self.type_map = {}

        # map name to data
        self.data = {}

        self.arg_names = arg_names

        self.used_names = set()

    def known_names(self):
        return (self.used_names
                | set(self.dim_map.iterkeys())
                | set(self.type_map.iterkeys()))

    def is_known(self, name):
        return (name in self.used_names
                or name in self.dim_map
                or name in self.type_map)

    def use_name(self, name):
        self.used_names.add(name)

    def get_type(self, name):
        try:
            return self.type_map[name]
        except KeyError:

            if self.implicit_types is None:
                raise TranslationError(
                        "no type for '%s' found in implict none routine"
                        % name)

            return self.implicit_types.get(name[0], np.dtype(np.int32))

    def get_shape(self, name):
        return self.dim_map.get(name, ())

    def translate_var_name(self, name):
        shape = self.dim_map.get(name)
        if name in self.data and shape is not None:
            return "%s_%s" % (self.subprogram_name, name)
        else:
            return name


class FTreeWalkerBase(object):
    def __init__(self):
        self.scope_stack = []

        self.expr_parser = FortranExpressionParser(self)

    def rec(self, expr, *args, **kwargs):
        mro = list(type(expr).__mro__)
        dispatch_class = kwargs.pop("dispatch_class", type(self))

        while mro:
            method_name = "map_"+mro.pop(0).__name__

            try:
                method = getattr(dispatch_class, method_name)
            except AttributeError:
                pass
            else:
                return method(self, expr, *args, **kwargs)

        raise NotImplementedError(
                "%s does not know how to map type '%s'"
                % (type(self).__name__,
                    type(expr)))

    ENTITY_RE = re.compile(
            r"^(?P<name>[_0-9a-zA-Z]+)"
            "(\((?P<shape>[-+*0-9:a-zA-Z,]+)\))?$")

    def parse_dimension_specs(self, dim_decls):
        def parse_bounds(bounds_str):
            start_end = bounds_str.split(":")

            assert 1 <= len(start_end) <= 2

            return (self.parse_expr(s) for s in start_end)

        for decl in dim_decls:
            entity_match = self.ENTITY_RE.match(decl)
            assert entity_match

            groups = entity_match.groupdict()
            name = groups["name"]
            assert name

            if groups["shape"]:
                shape = [parse_bounds(s) for s in groups["shape"].split(",")]
            else:
                shape = None

            yield name, shape

    def __call__(self, expr, *args, **kwargs):
        return self.rec(expr, *args, **kwargs)

    # {{{ expressions

    def parse_expr(self, expr_str):
        return self.expr_parser(expr_str)

    # }}}


