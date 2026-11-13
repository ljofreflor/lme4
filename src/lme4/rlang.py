"""Minimal representation of R language objects (calls, symbols, constants).

R formulas such as ``Reaction ~ Days + (Days | Subject)`` are parsed into a
tree of :class:`Call`, :class:`Sym` and constant nodes that mirrors R's own
language objects, so that the formula-manipulation code of lme4 and
reformulas (``findbars``, ``nobars``, ``subbars``, ``expandDoubleVerts``, ...)
can be transcribed almost literally.

The parser follows R's operator precedence (see ``?Syntax``) and
:func:`deparse` reproduces R's ``deparse()`` spacing conventions, which matter
because lme4 builds names (grouping factors, model-matrix columns, ...) from
deparsed expressions.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

__all__ = ["Sym", "Call", "Const", "parse", "deparse", "is_call", "head",
           "Formula", "as_formula", "NULL"]


class _Null:
    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __repr__(self):
        return "NULL"


NULL = _Null()


@dataclass(frozen=True)
class Sym:
    name: str

    def __repr__(self):
        return f"Sym({self.name!r})"


@dataclass(frozen=True)
class Const:
    value: Any          # float, int, str, bool, None (NA)

    def __repr__(self):
        return f"Const({self.value!r})"


@dataclass(frozen=True)
class Call:
    fn: Any                       # usually a Sym
    args: Tuple[Any, ...] = ()
    names: Tuple[Optional[str], ...] = ()   # argument names (None if unnamed)

    def __post_init__(self):
        if len(self.names) != len(self.args):
            object.__setattr__(self, "names", tuple([None] * len(self.args)))

    # R-style indexing: term[[1]] is the function, term[[2]] first arg ...
    def __getitem__(self, i):
        if i == 0:
            return self.fn
        return self.args[i - 1]

    def __len__(self):
        return 1 + len(self.args)

    def replace(self, i, value):
        """return a copy with element i (R-style, 0 = function) replaced"""
        if i == 0:
            return Call(value, self.args, self.names)
        args = list(self.args)
        args[i - 1] = value
        return Call(self.fn, tuple(args), self.names)

    def arg(self, name, pos=None, default=None):
        for a, n in zip(self.args, self.names):
            if n == name:
                return a
        if pos is not None:
            unnamed = [a for a, n in zip(self.args, self.names) if n is None]
            if pos < len(unnamed):
                return unnamed[pos]
        return default


def mkcall(fn: str, *args, names=None):
    return Call(Sym(fn), tuple(args), tuple(names) if names else tuple([None] * len(args)))


def head(x):
    """the operator of a call (R's ``x[[1]]``), or x itself for symbols"""
    if isinstance(x, Call):
        return x.fn
    return x


def is_call(x, name=None):
    if not isinstance(x, Call):
        return False
    if name is None:
        return True
    if isinstance(name, (list, tuple, set)):
        return isinstance(x.fn, Sym) and x.fn.name in name
    return isinstance(x.fn, Sym) and x.fn.name == name


def safe_length(x):
    if isinstance(x, Call):
        return len(x)
    return 1


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"""
    (?P<ws>\s+)
  | (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?L?)
  | (?P<str>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
  | (?P<bt>`[^`]*`)
  | (?P<op>%[^%]*%|\|\||&&|<=|>=|==|!=|<-|::|[-+*/^~:|&!<>(),=\[\]$@{}])
  | (?P<name>(?:[A-Za-z]|\.(?![0-9]))[A-Za-z0-9._]*|\.)
""", re.VERBOSE)


def _tokenize(s: str):
    pos = 0
    out = []
    while pos < len(s):
        m = _TOKEN_RE.match(s, pos)
        if not m:
            raise SyntaxError(f"unexpected character {s[pos]!r} in {s!r}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "ws":
            continue
        out.append((kind, m.group(kind)))
    out.append(("end", None))
    return out


# precedence (higher binds tighter) following ?Syntax
_BINARY = {
    "~": (1, "left"),
    "||": (2, "left"), "|": (2, "left"),
    "&&": (3, "left"), "&": (3, "left"),
    # "!" unary at 4
    "==": (5, "left"), "!=": (5, "left"), "<": (5, "left"), ">": (5, "left"),
    "<=": (5, "left"), ">=": (5, "left"),
    "+": (6, "left"), "-": (6, "left"),
    "*": (7, "left"), "/": (7, "left"),
    # %any% at 8
    ":": (9, "left"),
    # unary +- at 10
    "^": (11, "right"),
    "$": (13, "left"), "@": (13, "left"),
}
_PREC_UNARY_NOT = 4
_PREC_SPECIAL = 8
_PREC_UNARY_PM = 10


class _Parser:
    def __init__(self, s):
        self.toks = _tokenize(s)
        self.i = 0
        self.src = s

    def peek(self):
        return self.toks[self.i]

    def next(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, val):
        t = self.next()
        if t[1] != val:
            raise SyntaxError(f"expected {val!r} but got {t[1]!r} in {self.src!r}")

    def parse(self):
        e = self.expr(0)
        if self.peek()[0] != "end":
            raise SyntaxError(f"unexpected {self.peek()[1]!r} in {self.src!r}")
        return e

    def binop_info(self, tok):
        kind, val = tok
        if kind != "op":
            return None
        if val in _BINARY:
            return _BINARY[val]
        if val.startswith("%") and val.endswith("%") and len(val) >= 2:
            return (_PREC_SPECIAL, "left")
        return None

    def expr(self, minprec):
        left = self.unary()
        while True:
            tok = self.peek()
            info = self.binop_info(tok)
            if info is None:
                break
            prec, assoc = info
            if prec < minprec:
                break
            self.next()
            nxt = prec + 1 if assoc == "left" else prec
            right = self.expr(nxt)
            left = Call(Sym(tok[1]), (left, right))
        return left

    def unary(self):
        kind, val = self.peek()
        if kind == "op" and val in ("-", "+"):
            self.next()
            operand = self.expr(_PREC_UNARY_PM)
            return Call(Sym(val), (operand,))
        if kind == "op" and val == "!":
            self.next()
            operand = self.expr(_PREC_UNARY_NOT + 1)
            return Call(Sym("!"), (operand,))
        if kind == "op" and val == "~":
            self.next()
            operand = self.expr(2)
            return Call(Sym("~"), (operand,))
        return self.postfix(self.primary())

    def postfix(self, e):
        while True:
            kind, val = self.peek()
            if kind == "op" and val == "(":
                self.next()
                args, names = self.arglist(")")
                e = Call(e, tuple(args), tuple(names))
            elif kind == "op" and val == "[":
                self.next()
                if self.peek()[1] == "[":
                    self.next()
                    args, names = self.arglist("]")
                    self.expect("]")
                    e = Call(Sym("[["), (e,) + tuple(args), (None,) + tuple(names))
                else:
                    args, names = self.arglist("]")
                    e = Call(Sym("["), (e,) + tuple(args), (None,) + tuple(names))
            elif kind == "op" and val == "::":
                self.next()
                rhs = self.primary()
                e = Call(Sym("::"), (e, rhs))
            else:
                return e

    def arglist(self, close):
        args, names = [], []
        if self.peek()[1] == close:
            self.next()
            return args, names
        while True:
            name = None
            k1, v1 = self.toks[self.i]
            k2, v2 = self.toks[self.i + 1]
            if k1 in ("name", "bt", "str") and k2 == "op" and v2 == "=":
                name = v1.strip("`\"'")
                self.i += 2
            args.append(self.expr(0))
            names.append(name)
            t = self.next()
            if t[1] == close:
                break
            if t[1] != ",":
                raise SyntaxError(f"expected ',' or {close!r} in {self.src!r}")
        return args, names

    def primary(self):
        kind, val = self.next()
        if kind == "num":
            if val.endswith("L"):
                return Const(int(val[:-1]))
            v = float(val)
            return Const(v)
        if kind == "str":
            return Const(bytes(val[1:-1], "utf-8").decode("unicode_escape"))
        if kind == "bt":
            return Sym(val[1:-1])
        if kind == "name":
            if val == "TRUE" or val == "T":
                return Const(True) if val == "TRUE" else Sym("T")
            if val == "FALSE":
                return Const(False)
            if val == "NULL":
                return NULL
            if val == "NA":
                return Const(None)
            if val == "Inf":
                return Const(math.inf)
            if val == "NaN":
                return Const(math.nan)
            return Sym(val)
        if kind == "op" and val == "(":
            e = self.expr(0)
            self.expect(")")
            return Call(Sym("("), (e,))
        if kind == "op" and val == "{":
            e = self.expr(0)
            self.expect("}")
            return Call(Sym("{"), (e,))
        raise SyntaxError(f"unexpected token {val!r} in {self.src!r}")


def parse(s: str):
    """Parse a string into an R language object."""
    return _Parser(s).parse()


# ---------------------------------------------------------------------------
# Deparsing (R conventions)
# ---------------------------------------------------------------------------

_SPACED = {"~", "|", "||", "&", "&&", "==", "!=", "<", ">", "<=", ">=", "+", "-",
           "*", "/", "<-", "="}
_UNSPACED = {":", "^", "$", "@", "::", "/"}


def _fmt_num(v):
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, int):
        return f"{v}L"
    if math.isinf(v):
        return "Inf" if v > 0 else "-Inf"
    if math.isnan(v):
        return "NaN"
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    s = repr(float("%.15g" % v))
    if "e" in s:
        mant, ex = s.split("e")
        ex = int(ex)
        s = f"{mant}e{'-' if ex < 0 else '+'}{abs(ex):02d}"
    return s


def _needs_backticks(name):
    return not re.fullmatch(r"(?:[A-Za-z]|\.(?![0-9]))[A-Za-z0-9._]*|\.", name) or name in (
        "if", "else", "repeat", "while", "function", "for", "next", "break",
        "TRUE", "FALSE", "NULL", "Inf", "NaN", "NA", "in")


def _prec_of(e):
    if isinstance(e, Call) and isinstance(e.fn, Sym):
        op = e.fn.name
        if len(e.args) == 2:
            if op in _BINARY:
                return _BINARY[op][0]
            if op.startswith("%") and op.endswith("%"):
                return _PREC_SPECIAL
        if len(e.args) == 1 and op in ("-", "+"):
            return _PREC_UNARY_PM
        if len(e.args) == 1 and op == "!":
            return _PREC_UNARY_NOT
        if len(e.args) == 1 and op == "~":
            return 1
    return 100


def deparse(e, backtick=True) -> str:
    """Deparse an R language object following R's ``deparse()`` conventions."""
    if e is NULL:
        return "NULL"
    if isinstance(e, Sym):
        return f"`{e.name}`" if backtick and _needs_backticks(e.name) else e.name
    if isinstance(e, Const):
        v = e.value
        if v is None:
            return "NA"
        if isinstance(v, str):
            return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
        return _fmt_num(v)
    if isinstance(e, Formula):
        return deparse(e.expr)
    if not isinstance(e, Call):
        return str(e)
    fn = e.fn
    if isinstance(fn, Sym):
        op = fn.name
        if op == "(" and len(e.args) == 1:
            return "(" + deparse(e.args[0]) + ")"
        if op == "{" and len(e.args) == 1:
            return "{" + deparse(e.args[0]) + "}"
        if len(e.args) == 2 and (op in _BINARY or (op.startswith("%") and op.endswith("%"))):
            prec, assoc = _BINARY.get(op, (_PREC_SPECIAL, "left"))
            lhs, rhs = e.args
            ls = deparse(lhs)
            rs = deparse(rhs)
            lp, rp = _prec_of(lhs), _prec_of(rhs)
            if lp < prec or (assoc == "right" and lp == prec):
                ls = "(" + ls + ")"
            if rp < prec or (assoc == "left" and rp == prec and op not in ("~",)):
                rs = "(" + rs + ")"
            if op in _UNSPACED:
                return f"{ls}{op}{rs}"
            return f"{ls} {op} {rs}"
        if len(e.args) == 1 and op in ("-", "+", "!"):
            s = deparse(e.args[0])
            if _prec_of(e.args[0]) < _prec_of(e):
                s = "(" + s + ")"
            return op + s
        if len(e.args) == 1 and op == "~":
            return "~" + deparse(e.args[0])
        if op == "[" or op == "[[":
            obj = deparse(e.args[0])
            inner = ", ".join(_dparg(a, n) for a, n in zip(e.args[1:], e.names[1:]))
            return f"{obj}[{inner}]" if op == "[" else f"{obj}[[{inner}]]"
        fs = deparse(fn)
    else:
        fs = deparse(fn)
    inner = ", ".join(_dparg(a, n) for a, n in zip(e.args, e.names))
    return f"{fs}({inner})"


def _dparg(a, n):
    if n is None:
        return deparse(a)
    nn = f"`{n}`" if _needs_backticks(n) else n
    return f"{nn} = {deparse(a)}"


# ---------------------------------------------------------------------------
# Formula objects
# ---------------------------------------------------------------------------

class Formula:
    """An R formula: a ``~`` call plus an environment for variable lookup."""

    def __init__(self, expr, env=None):
        if isinstance(expr, str):
            expr = parse(expr)
        if not is_call(expr, "~"):
            expr = Call(Sym("~"), (expr,))
        self.expr = expr
        self.env = env if env is not None else {}

    # Pickling.  R serializes a formula's environment; in Python the
    # environment is typically the caller's globals(), with modules and other
    # unpicklable objects.  Keep only the entries the formula refers to (by
    # name) that can be pickled; everything else is looked up in the data.
    def __getstate__(self):
        return {"expr": self.expr, "env": prune_env(self.env, self.expr)}

    def __setstate__(self, st):
        self.expr = st["expr"]
        self.env = st["env"]

    # formula[[2]] etc.
    def __len__(self):
        return len(self.expr)

    def __getitem__(self, i):
        return self.expr[i]

    @property
    def lhs(self):
        return self.expr.args[0] if len(self.expr.args) == 2 else None

    @property
    def rhs(self):
        return self.expr.args[-1]

    def with_rhs(self, rhs):
        if len(self.expr.args) == 2:
            return Formula(Call(Sym("~"), (self.expr.args[0], rhs)), self.env)
        return Formula(Call(Sym("~"), (rhs,)), self.env)

    def with_lhs(self, lhs):
        if lhs is None:
            return Formula(Call(Sym("~"), (self.rhs,)), self.env)
        return Formula(Call(Sym("~"), (lhs, self.rhs)), self.env)

    def __str__(self):
        return deparse(self.expr)

    def __repr__(self):
        return deparse(self.expr)

    def __eq__(self, other):
        return isinstance(other, Formula) and self.expr == other.expr

    def __hash__(self):
        return hash(self.expr)


def as_formula(f, env=None):
    if isinstance(f, Formula):
        if env is not None and not f.env:
            return Formula(f.expr, env)
        return f
    if isinstance(f, str):
        return Formula(parse(f), env)
    if isinstance(f, Call):
        return Formula(f, env)
    raise TypeError(f"cannot interpret {f!r} as a formula")


def prune_env(env, expr):
    """The picklable part of an evaluation environment: the entries that
    ``expr`` refers to by name and that can be pickled.  (R serializes whole
    environments; here an environment is usually the caller's globals(),
    with modules and other unpicklable objects.)"""
    import pickle
    out = {}
    try:
        names = set(all_names(expr))
        items = [(k, env[k]) for k in names if k in env]
    except Exception:
        return out
    for k, v in items:
        try:
            pickle.dumps(v)
        except Exception:
            continue
        out[k] = v
    return out


def all_names(e):
    """R's all.names(): all symbol names (including function names)"""
    out = []

    def rec(x):
        if isinstance(x, Sym):
            out.append(x.name)
        elif isinstance(x, Call):
            rec(x.fn)
            for a in x.args:
                rec(a)
        elif isinstance(x, Formula):
            rec(x.expr)
    rec(e)
    return out


def all_vars(e):
    """R's all.vars(): symbol names that are not function names"""
    out = []

    def rec(x):
        if isinstance(x, Sym):
            if x.name not in out:
                out.append(x.name)
        elif isinstance(x, Call):
            if not isinstance(x.fn, Sym):
                rec(x.fn)
            for a in x.args:
                rec(a)
        elif isinstance(x, Formula):
            rec(x.expr)
    rec(e)
    return out
