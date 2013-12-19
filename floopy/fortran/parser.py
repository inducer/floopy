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

import cgen
import numpy as np
import re
from pymbolic.mapper import CombineMapper
from pymbolic.mapper.c_code import CCodeMapper as CCodeMapperBase


# {{{ AST components

def dtype_to_ctype(dtype):
    if dtype is None:
        raise ValueError("dtype may not be None")

    dtype = np.dtype(dtype)
    if dtype == np.int64:
        return "long"
    elif dtype == np.uint64:
        return "unsigned long"
    elif dtype == np.int32:
        return "int"
    elif dtype == np.uint32:
        return "unsigned int"
    elif dtype == np.int16:
        return "short int"
    elif dtype == np.uint16:
        return "short unsigned int"
    elif dtype == np.int8:
        return "signed char"
    elif dtype == np.uint8:
        return "unsigned char"
    elif dtype == np.float32:
        return "float"
    elif dtype == np.float64:
        return "double"
    elif dtype == np.complex64:
        return "cfloat_t"
    elif dtype == np.complex128:
        return "cdouble_t"
    else:
        raise ValueError("unable to map dtype '%s'" % dtype)


class POD(cgen.POD):
    def get_decl_pair(self):
        return [dtype_to_ctype(self.dtype)], self.name

# }}}


# {{{ expression generator

class TypeInferenceMapper(CombineMapper):
    def __init__(self, scope):
        self.scope = scope

    def combine(self, dtypes):
        return sum(dtype.type(1) for dtype in dtypes).dtype

    def map_literal(self, expr):
        return expr.dtype

    def map_constant(self, expr):
        return np.asarray(expr).dtype

    def map_variable(self, expr):
        return self.scope.get_type(expr.name)

    def map_call(self, expr):
        name = expr.function.name
        if name == "fromreal":
            arg, = expr.parameters
            base_dtype = self.rec(arg)
            tgt_real_dtype = (np.float32(0)+base_dtype.type(0)).dtype
            assert tgt_real_dtype.kind == "f"
            if tgt_real_dtype == np.float32:
                return np.dtype(np.complex64)
            elif tgt_real_dtype == np.float64:
                return np.dtype(np.complex128)
            else:
                raise RuntimeError("unexpected complex type")

        else:
            return CombineMapper.map_call(self, expr)


class ComplexCCodeMapper(CCodeMapperBase):
    def __init__(self, infer_type):
        CCodeMapperBase.__init__(self)
        self.infer_type = infer_type

    def complex_type_name(self, dtype):
        if dtype == np.complex64:
            return "cfloat"
        if dtype == np.complex128:
            return "cdouble"
        else:
            raise RuntimeError

    def map_sum(self, expr, enclosing_prec):
        tgt_dtype = self.infer_type(expr)
        is_complex = tgt_dtype.kind == 'c'

        if not is_complex:
            return CCodeMapperBase.map_sum(self, expr, enclosing_prec)
        else:
            tgt_name = self.complex_type_name(tgt_dtype)

            reals = [child for child in expr.children
                    if 'c' != self.infer_type(child).kind]
            complexes = [child for child in expr.children
                    if 'c' == self.infer_type(child).kind]

            from pymbolic.mapper.stringifier import PREC_SUM
            real_sum = self.join_rec(" + ", reals, PREC_SUM)
            complex_sum = self.join_rec(" + ", complexes, PREC_SUM)

            if real_sum:
                result = "%s_fromreal(%s) + %s" % (tgt_name, real_sum, complex_sum)
            else:
                result = complex_sum

            return self.parenthesize_if_needed(result, enclosing_prec, PREC_SUM)

    def map_product(self, expr, enclosing_prec):
        tgt_dtype = self.infer_type(expr)
        is_complex = 'c' == tgt_dtype.kind

        if not is_complex:
            return CCodeMapperBase.map_product(self, expr, enclosing_prec)
        else:
            tgt_name = self.complex_type_name(tgt_dtype)

            reals = [child for child in expr.children
                    if 'c' != self.infer_type(child).kind]
            complexes = [child for child in expr.children
                    if 'c' == self.infer_type(child).kind]

            from pymbolic.mapper.stringifier import PREC_PRODUCT, PREC_NONE
            real_prd = self.join_rec("*", reals, PREC_PRODUCT)

            if len(complexes) == 1:
                myprec = PREC_PRODUCT
            else:
                myprec = PREC_NONE

            complex_prd = self.rec(complexes[0], myprec)
            for child in complexes[1:]:
                complex_prd = "%s_mul(%s, %s)" % (
                        tgt_name, complex_prd,
                        self.rec(child, PREC_NONE))

            if real_prd:
                # elementwise semantics are correct
                result = "%s * %s" % (real_prd, complex_prd)
            else:
                result = complex_prd

            return self.parenthesize_if_needed(result, enclosing_prec, PREC_PRODUCT)

    def map_quotient(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE
        n_complex = 'c' == self.infer_type(expr.numerator).kind
        d_complex = 'c' == self.infer_type(expr.denominator).kind

        tgt_dtype = self.infer_type(expr)

        if not (n_complex or d_complex):
            return CCodeMapperBase.map_quotient(self, expr, enclosing_prec)
        elif n_complex and not d_complex:
            # elementwise semnatics are correct
            return CCodeMapperBase.map_quotient(self, expr, enclosing_prec)
        elif not n_complex and d_complex:
            return "%s_rdivide(%s, %s)" % (
                    self.complex_type_name(tgt_dtype),
                    self.rec(expr.numerator, PREC_NONE),
                    self.rec(expr.denominator, PREC_NONE))
        else:
            return "%s_divide(%s, %s)" % (
                    self.complex_type_name(tgt_dtype),
                    self.rec(expr.numerator, PREC_NONE),
                    self.rec(expr.denominator, PREC_NONE))

    def map_remainder(self, expr, enclosing_prec):
        tgt_dtype = self.infer_type(expr)
        if 'c' == tgt_dtype.kind:
            raise RuntimeError("complex remainder not defined")

        return CCodeMapperBase.map_remainder(self, expr, enclosing_prec)

    def map_power(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE

        tgt_dtype = self.infer_type(expr)
        if 'c' == tgt_dtype.kind:
            if expr.exponent in [2, 3, 4]:
                value = expr.base
                for i in range(expr.exponent-1):
                    value = value * expr.base
                return self.rec(value, enclosing_prec)
            else:
                b_complex = 'c' == self.infer_type(expr.base).kind
                e_complex = 'c' == self.infer_type(expr.exponent).kind

                if b_complex and not e_complex:
                    return "%s_powr(%s, %s)" % (
                            self.complex_type_name(tgt_dtype),
                            self.rec(expr.base, PREC_NONE),
                            self.rec(expr.exponent, PREC_NONE))
                else:
                    return "%s_pow(%s, %s)" % (
                            self.complex_type_name(tgt_dtype),
                            self.rec(expr.base, PREC_NONE),
                            self.rec(expr.exponent, PREC_NONE))

        return CCodeMapperBase.map_power(self, expr, enclosing_prec)


class CCodeMapper(ComplexCCodeMapper):
    # Whatever is needed to mop up after Fortran goes here.
    # Stuff that deals with generating real-valued code
    # from complex code goes above.

    def __init__(self, translator, scope):
        ComplexCCodeMapper.__init__(self, scope.get_type_inference_mapper())
        self.translator = translator
        self.scope = scope

    def map_subscript(self, expr, enclosing_prec):
        idx_dtype = self.infer_type(expr.index)
        if not 'i' == idx_dtype.kind or 'u' == idx_dtype.kind:
            ind_prefix = "(int) "
        else:
            ind_prefix = ""

        idx = expr.index
        if isinstance(idx, tuple) and len(idx) == 1:
            idx, = idx

        from pymbolic.mapper.stringifier import PREC_NONE, PREC_CALL
        return self.parenthesize_if_needed(
                self.format("%s[%s%s]",
                    self.scope.translate_var_name(expr.aggregate.name),
                    ind_prefix,
                    self.rec(idx, PREC_NONE)),
                enclosing_prec, PREC_CALL)

    def map_call(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE

        tgt_dtype = self.infer_type(expr)

        name = expr.function.name
        if 'f' == tgt_dtype.kind and name == "abs":
            name = "fabs"

        if 'c' == tgt_dtype.kind:
            if name in ["conjg", "dconjg"]:
                name = "conj"

            if name[:2] == "cd" and name[2:] in ["log", "exp", "sqrt"]:
                name = name[2:]

            if name == "aimag":
                name = "imag"

            if name == "dble":
                name = "real"

            name = "%s_%s" % (
                    self.complex_type_name(tgt_dtype),
                    name)

        return self.format("%s(%s)",
                name,
                self.join_rec(", ", expr.parameters, PREC_NONE))

    def map_variable(self, expr, enclosing_prec):
        # guaranteed to not be a subscript or a call

        name = expr.name
        shape = self.scope.get_shape(name)
        name = self.scope.translate_var_name(name)
        if expr.name in self.scope.arg_names:
            arg_idx = self.scope.arg_names.index(name)
            if self.translator.arg_needs_pointer(
                    self.scope.subprogram_name, arg_idx):
                return "*"+name
            else:
                return name
        elif shape not in [(), None]:
            return "*"+name
        else:
            return name

    def map_literal(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE
        if expr.dtype.kind == "c":
            r, i = expr.value
            return "{ %s, %s }" % (self.rec(r, PREC_NONE), self.rec(i, PREC_NONE))
        else:
            return expr.value

    def map_wildcard(self, expr, enclosing_prec):
        return ":"


# }}}

class ArgumentAnalayzer(FTreeWalkerBase):
    def __init__(self):
        FTreeWalkerBase.__init__(self)

        # map (func, arg_nr) to
        # 'w' for 'needs pointer'
        # [] for no obstacle to de-pointerification known
        # [(func_name, arg_nr), ...] # depends on how this arg is used

        self.arg_usage_info = {}

    def arg_needs_pointer(self, func, arg_nr):
        data = self.arg_usage_info.get((func, arg_nr), [])

        if isinstance(data, list):
            return any(
                    self.arg_needs_pointer(sub_func, sub_arg_nr)
                    for sub_func, sub_arg_nr in data)

        return True

    # {{{ map_XXX functions

    def map_BeginSource(self, node):
        scope = Scope(None)
        self.scope_stack.append(scope)

        for c in node.content:
            self.rec(c)

    def map_Subroutine(self, node):
        scope = Scope(node.name, list(node.args))
        self.scope_stack.append(scope)

        for c in node.content:
            self.rec(c)

        self.scope_stack.pop()

    def map_EndSubroutine(self, node):
        pass

    def map_Implicit(self, node):
        pass

    # {{{ types, declarations

    def map_Equivalence(self, node):
        raise NotImplementedError("equivalence")

    def map_Dimension(self, node):
        scope = self.scope_stack[-1]

        for name, shape in self.parse_dimension_specs(node.items):
            if name in scope.arg_names:
                arg_idx = scope.arg_names.index(name)
                self.arg_usage_info[scope.subprogram_name, arg_idx] = "w"

    def map_External(self, node):
        pass

    def map_type_decl(self, node):
        scope = self.scope_stack[-1]

        for name, shape in self.parse_dimension_specs(node.entity_decls):
            if shape is not None and name in scope.arg_names:
                arg_idx = scope.arg_names.index(name)
                self.arg_usage_info[scope.subprogram_name, arg_idx] = "w"

    map_Logical = map_type_decl
    map_Integer = map_type_decl
    map_Real = map_type_decl
    map_Complex = map_type_decl

    # }}}

    def map_Data(self, node):
        pass

    def map_Parameter(self, node):
        raise NotImplementedError("parameter")

    # {{{ I/O

    def map_Open(self, node):
        pass

    def map_Format(self, node):
        pass

    def map_Write(self, node):
        pass

    def map_Print(self, node):
        pass

    def map_Read1(self, node):
        pass

    # }}}

    def map_Assignment(self, node):
        scope = self.scope_stack[-1]

        lhs = self.parse_expr(node.variable)

        from pymbolic.primitives import Subscript, Call
        if isinstance(lhs, Subscript):
            lhs_name = lhs.aggregate.name
        elif isinstance(lhs, Call):
            # in absence of dim info, subscripts get parsed as calls
            lhs_name = lhs.function.name
        else:
            lhs_name = lhs.name

        if lhs_name in scope.arg_names:
            arg_idx = scope.arg_names.index(lhs_name)
            self.arg_usage_info[scope.subprogram_name, arg_idx] = "w"

    def map_Allocate(self, node):
        raise NotImplementedError("allocate")

    def map_Deallocate(self, node):
        raise NotImplementedError("deallocate")

    def map_Save(self, node):
        raise NotImplementedError("save")

    def map_Line(self, node):
        raise NotImplementedError

    def map_Program(self, node):
        raise NotImplementedError

    def map_Entry(self, node):
        raise NotImplementedError

    # {{{ control flow

    def map_Goto(self, node):
        pass

    def map_Call(self, node):
        scope = self.scope_stack[-1]

        from pymbolic.primitives import Subscript, Variable
        for i, arg_str in enumerate(node.items):
            arg = self.parse_expr(arg_str)
            if isinstance(arg, (Variable, Subscript)):
                if isinstance(arg, Subscript):
                    arg_name = arg.aggregate.name
                else:
                    arg_name = arg.name

                if arg_name in scope.arg_names:
                    arg_idx = scope.arg_names.index(arg_name)
                    arg_usage = self.arg_usage_info.setdefault(
                            (scope.subprogram_name, arg_idx),
                            [])
                    if isinstance(arg_usage, list):
                        arg_usage.append((node.designator, i))

    def map_Return(self, node):
        pass

    def map_ArithmeticIf(self, node):
        pass

    def map_If(self, node):
        for c in node.content:
            self.rec(c)

    def map_IfThen(self, node):
        for c in node.content:
            self.rec(c)

    def map_ElseIf(self, node):
        pass

    def map_Else(self, node):
        pass

    def map_EndIfThen(self, node):
        pass

    def map_Do(self, node):
        for c in node.content:
            self.rec(c)

    def map_EndDo(self, node):
        pass

    def map_Continue(self, node):
        pass

    def map_Stop(self, node):
        pass

    def map_Comment(self, node):
        pass

    # }}}

    # }}}


class F2LoopyTranslator(FTreeWalkerBase):
    pass


def f2loopy(source, free_form=False, strict=True):
    from fparser import api
    tree = api.parse(source, isfree=free_form, isstrict=strict,
            analyze=False, ignore_comments=False)

    arg_info = ArgumentAnalayzer()
    arg_info(tree)

    f2loopy = F2LoopyTranslator()
    f2loopy(tree)

    1/0

# vim: foldmethod=marker
