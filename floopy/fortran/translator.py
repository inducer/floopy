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


from floopy.fortran.parser import FTreeWalkerBase


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

        self.index_set = None
        self.instructions = []
        self.temporary_variables = []

        self.in_transform_code = False
        self.transform_code_lines = []

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

# }}}


# {{{ translator

class F2CLTranslator(FTreeWalkerBase):
    def __init__(self):
        self.scope_stack = []

    def map_statement_list(self, content):
        body = []

        for c in content:
            mapped = self.rec(c)
            if mapped is None:
                warn("mapping '%s' returned None" % type(c))
            elif isinstance(mapped, list):
                body.extend(mapped)
            else:
                body.append(mapped)

        return body

    # {{{ map_XXX functions

    def map_BeginSource(self, node):
        scope = Scope(None)
        self.scope_stack.append(scope)

        return self.map_statement_list(node.content)

    def map_Subroutine(self, node):
        assert not node.prefix
        assert not hasattr(node, "suffix")

        scope = Scope(node.name, list(node.args))
        self.scope_stack.append(scope)

        body = self.map_statement_list(node.content)

        pre_func_decl, in_func_decl = self.get_declarations()
        body = in_func_decl + [cgen.Line()] + body

        if isinstance(body[-1], cgen.Statement) and body[-1].text == "return":
            body.pop()

        def get_arg_decl(arg_idx, arg_name):
            decl = self.get_declarator(arg_name)

            if self.arg_needs_pointer(node.name, arg_idx):
                hint = self.addr_space_hints.get((node.name, arg_name))
                if hint:
                    decl = hint(cgen.Pointer(decl))
                else:
                    if self.use_restrict_pointers:
                        decl = cgen.RestrictPointer(decl)
                    else:
                        decl = cgen.Pointer(decl)

            return decl


        result =  cgen.FunctionBody(
                cgen.FunctionDeclaration(
                    cgen.Value("void", node.name),
                    [get_arg_decl(i, arg) for i, arg in enumerate(node.args)]
                    ),
                cgen.Block(body))

        self.scope_stack.pop()
        if pre_func_decl:
            return pre_func_decl + [cgen.Line(), result]
        else:
            return result

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
        lhs = self.parse_expr(node.variable)
        from pymbolic.primitives import Subscript
        if isinstance(lhs, Subscript):
            lhs_name = lhs.aggregate.name
        else:
            lhs_name = lhs.name

        scope = self.scope_stack[-1]
        scope.use_name(lhs_name)
        infer_type = scope.get_type_inference_mapper()

        rhs = self.parse_expr(node.expr)
        lhs_dtype = infer_type(lhs)
        rhs_dtype = infer_type(rhs)

        # check for silent truncation of complex
        if lhs_dtype.kind != 'c' and rhs_dtype.kind == 'c':
            from pymbolic import var
            rhs = var("real")(rhs)
        # check for silent widening of real
        if lhs_dtype.kind == 'c' and rhs_dtype.kind != 'c':
            from pymbolic import var
            rhs = var("fromreal")(rhs)

        return cgen.Assign(self.gen_expr(lhs), self.gen_expr(rhs))

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
        return cgen.Statement("goto label_%s" % node.label)

    def map_Call(self, node):
        def transform_arg(i, arg_str):
            expr = self.parse_expr(arg_str)
            result = self.gen_expr(expr)
            if self.arg_needs_pointer(node.designator, i):
                result = "&"+result

            cast = self.force_casts.get(
                    (node.designator, i))
            if cast is not None:
                result = "(%s) (%s)" % (cast, result)

            return result

        return cgen.Statement("%s(%s)" % (
            node.designator,
            ", ".join(transform_arg(i, arg_str) 
                for i, arg_str in enumerate(node.items))))

    def map_Return(self, node):
        return cgen.Statement("return")

    def map_ArithmeticIf(self, node):
        raise NotImplementedError

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

        body = self.map_statement_list(node.content)

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

            if not isinstance(step, int):
                print type(step)
                raise TranslationError("non-constant steps not yet supported: %s" % step)

            if step < 0:
                comp_op = ">="
            else:
                comp_op = "<="

            return cgen.For(
                    "%s = %s" % (loop_var, self.gen_expr(start)),
                    "%s %s %s" % (loop_var, comp_op, self.gen_expr(stop)),
                    "%s += %s" % (loop_var, self.gen_expr(step)),
                    cgen.block_if_necessary(body))

        else:
            raise NotImplementedError("unbounded do loop")

    def map_EndDo(self, node):
        return []

    def map_Continue(self, node):
        return cgen.Statement("label_%s:" % node.label)

    def map_Stop(self, node):
        raise NotImplementedError("stop")

    def map_Comment(self, node):
        stripped_comment_line = node.content.strip()

        scope = self.scope_stack[-1]
        if stripped_comment_line == "$loopy begin transform":
            if scope.in_transform_code:
                raise TranslationError("can't enter transform code twice")

        elif stripped_comment_line == "$loopy end transform":
            if not scope.in_transform_code:
                raise TranslationError("can't leave transform code twice")

        elif scope.in_transform_code:
            scope.transform_code_lines.append(node.content)

    # }}}

    # }}}

    def make_kernel(self, target):

# }}}

# vim: foldmethod=marker
