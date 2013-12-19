from __future__ import division, with_statement

__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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

import loopy as lp
import numpy as np
from warnings import warn
from floopy.fortran.tree import FTreeWalkerBase
from floopy.fortran.diagnostic import (
        TranslationError, TranslatorWarning)
import islpy as isl
from islpy import dim_type


# {{{ scope

class Scope(object):
    def __init__(self, subprogram_name, arg_names=set()):
        self.subprogram_name = subprogram_name

        # map name to data
        self.data_statements = {}

        # map first letter to type
        self.implicit_types = {}

        # map name to dim tuple
        self.dim_map = {}

        # map name to type
        self.type_map = {}

        # map name to data
        self.data = {}

        self.arg_names = arg_names

        self.index_sets = []

        # This dict has a key for every iname that is
        # currently active. These keys map to the loopy-side
        # name of the iname, which may differ because
        # duplicate inames need to be renamed for loopy.
        self.active_iname_aliases = {}

        self.instructions = []
        self.temporary_variables = []

        self.used_names = set()

        self.previous_instruction_id = None

    def known_names(self):
        return (self.used_names
                | set(self.dim_map.iterkeys())
                | set(self.type_map.iterkeys()))

    def is_known(self, name):
        return (name in self.used_names
                or name in self.dim_map
                or name in self.type_map
                or name in self.arg_names)

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

    def get_iname_alias_subst_mapper(self):
        from pymbolic.mapper.substitutor import make_subst_func
        from loopy.symbolic import SubstitutionMapper

        from pymbolic import var
        iname_aliases_with_vars = dict(
                (iname, var(alias))
                for iname, alias in self.active_iname_aliases.iteritems())

        return SubstitutionMapper(
                make_subst_func(iname_aliases_with_vars))

# }}}


def remove_common_indentation(lines):
    while lines[0].strip() == "":
        lines.pop(0)
    while lines[-1].strip() == "":
        lines.pop(-1)

    if lines:
        base_indent = 0
        while lines[0][base_indent] in " \t":
            base_indent += 1

        for line in lines[1:]:
            if line[:base_indent].strip():
                raise ValueError("inconsistent indentation")

    return "\n".join(line[base_indent:] for line in lines)


# {{{ translator

class F2LoopyTranslator(FTreeWalkerBase):
    def __init__(self):
        FTreeWalkerBase.__init__(self)

        self.scope_stack = []
        self.isl_context = isl.Context()

        self.insn_id_counter = 0

        self.kernels = []

        # Flag to record whether 'loopy begin transform' comment
        # has been seen.
        self.in_transform_code = False

        self.transform_code_lines = []


    # {{{ map_XXX functions

    def map_BeginSource(self, node):
        scope = Scope(None)
        self.scope_stack.append(scope)

        for c in node.content:
            self.rec(c)

    def map_Subroutine(self, node):
        assert not node.prefix
        assert not hasattr(node, "suffix")

        scope = Scope(node.name, list(node.args))
        self.scope_stack.append(scope)

        for c in node.content:
            self.rec(c)

        self.scope_stack.pop()

        self.kernels.append(scope)

    def map_EndSubroutine(self, node):
        return []

    def map_Implicit(self, node):
        scope = self.scope_stack[-1]

        if not node.items:
            assert not scope.implicit_types
            scope.implicit_types = None

        for stmt, specs in node.items:
            tp = self.dtype_from_stmt(stmt)
            for start, end in specs:
                for char_code in range(ord(start), ord(end)+1):
                    scope.implicit_types[chr(char_code)] = tp

        return []

    # {{{ types, declarations

    def map_Equivalence(self, node):
        raise NotImplementedError("equivalence")

    TYPE_MAP = {
            ("real", "4"): np.float32,
            ("real", "8"): np.float64,
            ("real", "16"): np.float128,

            ("complex", "8"): np.complex64,
            ("complex", "16"): np.complex128,
            ("complex", "32"): np.complex256,

            ("integer", ""): np.int32,
            ("integer", "4"): np.int32,
            ("complex", "8"): np.int64,
            }

    def dtype_from_stmt(self, stmt):
        length, kind = stmt.selector
        assert not kind
        return np.dtype(self.TYPE_MAP[(type(stmt).__name__.lower(), length)])

    def map_type_decl(self, node):
        scope = self.scope_stack[-1]

        tp = self.dtype_from_stmt(node)

        for name, shape in self.parse_dimension_specs(node.entity_decls):
            if shape is not None:
                assert name not in scope.dim_map
                scope.dim_map[name] = shape
                scope.use_name(name)

            assert name not in scope.type_map
            scope.type_map[name] = tp

        return []

    map_Logical = map_type_decl
    map_Integer = map_type_decl
    map_Real = map_type_decl
    map_Complex = map_type_decl

    def map_Dimension(self, node):
        scope = self.scope_stack[-1]

        for name, shape in self.parse_dimension_specs(node.items):
            if shape is not None:
                assert name not in scope.dim_map
                scope.dim_map[name] = shape
                scope.use_name(name)

        return []

    def map_External(self, node):
        raise NotImplementedError("external")

    # }}}

    def map_Data(self, node):
        scope = self.scope_stack[-1]

        for name, data in node.stmts:
            name, = name
            assert name not in scope.data
            scope.data[name] = [self.parse_expr(i) for i in data]

        return []

    def map_Parameter(self, node):
        raise NotImplementedError("parameter")

    # {{{ I/O

    def map_Open(self, node):
        raise NotImplementedError

    def map_Format(self, node):
        warn("'format' unsupported", TranslatorWarning)

    def map_Write(self, node):
        warn("'write' unsupported", TranslatorWarning)

    def map_Print(self, node):
        warn("'print' unsupported", TranslatorWarning)

    def map_Read1(self, node):
        warn("'read' unsupported", TranslatorWarning)

    # }}}

    def map_Assignment(self, node):
        scope = self.scope_stack[-1]

        iname_alias_subst_map = scope.get_iname_alias_subst_mapper()
        lhs = iname_alias_subst_map(self.parse_expr(node.variable))
        from pymbolic.primitives import Subscript
        if isinstance(lhs, Subscript):
            lhs_name = lhs.aggregate.name
        else:
            lhs_name = lhs.name

        scope.use_name(lhs_name)

        from loopy.kernel.data import ExpressionInstruction

        rhs = iname_alias_subst_map(self.parse_expr(node.expr))

        new_id = "insn%d" % self.insn_id_counter
        self.insn_id_counter += 1

        if scope.previous_instruction_id:
            insn_deps = frozenset([scope.previous_instruction_id])
        else:
            insn_deps = frozenset()

        insn = ExpressionInstruction(
                lhs, rhs,
                forced_iname_deps=frozenset(
                    scope.active_iname_aliases.itervalues()),
                insn_deps=insn_deps,
                id=new_id)

        scope.previous_instruction_id = new_id
        scope.instructions.append(insn)

    def map_Allocate(self, node):
        raise NotImplementedError("allocate")

    def map_Deallocate(self, node):
        raise NotImplementedError("deallocate")

    def map_Save(self, node):
        raise NotImplementedError("save")

    def map_Line(self, node):
        #from warnings import warn
        #warn("Encountered a 'line': %s" % node)
        raise NotImplementedError

    def map_Program(self, node):
        raise NotImplementedError

    def map_Entry(self, node):
        raise NotImplementedError

    # {{{ control flow

    def map_Goto(self, node):
        raise NotImplementedError("goto")

    def map_Call(self, node):
        raise NotImplementedError("call")

    def map_Return(self, node):
        raise NotImplementedError("return")

    def map_ArithmeticIf(self, node):
        raise NotImplementedError("arithmetic-if")

    def map_If(self, node):
        raise NotImplementedError("if")
        # node.expr
        # node.content[0]

    def map_IfThen(self, node):
        raise NotImplementedError("if-then")

    def map_EndIfThen(self, node):
        return []

    def map_Do(self, node):
        scope = self.scope_stack[-1]

        if node.loopcontrol:
            loop_var, loop_bounds = node.loopcontrol.split("=")
            loop_var = loop_var.strip()
            scope.use_name(loop_var)
            loop_bounds = [self.parse_expr(s) for s in loop_bounds.split(",")]

            if len(loop_bounds) == 2:
                start, stop = loop_bounds
                step = 1
            elif len(loop_bounds) == 3:
                start, stop, step = loop_bounds
            else:
                raise RuntimeError("loop bounds not understood: %s"
                        % node.loopcontrol)

            if step != 1:
                raise NotImplementedError(
                        "do loops with non-unit stride")

            if not isinstance(step, int):
                print type(step)
                raise TranslationError(
                        "non-constant steps not supported: %s" % step)

            from loopy.symbolic import get_dependencies
            loop_bound_deps = (
                    get_dependencies(start)
                    | get_dependencies(stop)
                    | get_dependencies(step))

            # {{{ find a usable loopy-side loop name

            loopy_loop_var = loop_var
            loop_var_suffix = None
            while True:
                already_used = False
                for iset in scope.index_sets:
                    if loopy_loop_var in iset.get_var_dict(dim_type.set):
                        already_used = True
                        break

                if not already_used:
                    break

                if loop_var_suffix is None:
                    loop_var_suffix = 0

                loop_var_suffix += 1
                loopy_loop_var = loop_var + "_%d" % loop_var_suffix

            # }}}

            space = isl.Space.create_from_names(self.isl_context,
                    set=[loopy_loop_var], params=list(loop_bound_deps))

            from loopy.isl_helpers import iname_rel_aff
            from loopy.symbolic import aff_from_expr
            index_set = (
                    isl.BasicSet.universe(space)
                    .add_constraint(
                        isl.Constraint.inequality_from_aff(
                            iname_rel_aff(space,
                                loopy_loop_var, ">=",
                                aff_from_expr(space, start))))
                    .add_constraint(
                        isl.Constraint.inequality_from_aff(
                            iname_rel_aff(space,
                                loopy_loop_var, "<=",
                                aff_from_expr(space, stop)))))

            scope.active_iname_aliases[loop_var] = loopy_loop_var
            scope.index_sets.append(index_set)

            for c in node.content:
                self.rec(c)

            del scope.active_iname_aliases[loop_var]

        else:
            raise NotImplementedError("unbounded do loop")

    def map_EndDo(self, node):
        pass

    def map_Continue(self, node):
        raise NotImplementedError("continue")

    def map_Stop(self, node):
        raise NotImplementedError("stop")

    def map_Comment(self, node):
        stripped_comment_line = node.content.strip()

        if stripped_comment_line == "$loopy begin transform":
            if self.in_transform_code:
                raise TranslationError("can't enter transform code twice")
            self.in_transform_code = True

        elif stripped_comment_line == "$loopy end transform":
            if not self.in_transform_code:
                raise TranslationError("can't leave transform code twice")
            self.in_transform_code = False

        elif self.in_transform_code:
            self.transform_code_lines.append(node.content)

    # }}}

    # }}}

    def make_kernels(self, target):
        kernel_names = [
                sub.subprogram_name
                for sub in self.kernels]

        proc_dict = {}
        proc_dict["lp"] = lp
        proc_dict["np"] = np

        #import pudb
        #pu.db
        for sub in self.kernels:
            knl = lp.make_kernel(target,
                    sub.index_sets,
                    sub.instructions,
                    sub.arg_names,
                    name=sub.subprogram_name)
            proc_dict[sub.subprogram_name] = knl

        transform_code = remove_common_indentation(
                self.transform_code_lines)

        exec(compile(transform_code,
            "<loopy transforms>", "exec"), proc_dict)

        return [proc_dict[knl_name]
                for knl_name in kernel_names]

# }}}

# vim: foldmethod=marker
