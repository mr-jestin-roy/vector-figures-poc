"""Safe recursive-descent parser + evaluator for the CONTRACT.md
"equations mini-language".

We deliberately do NOT use Python's `eval()` anywhere in this module --
these strings originate from an LLM (the sibling problem_parser agent's
output) and must be treated as untrusted input. Instead we:

  1. Tokenize with a strict regex whitelist (identifiers, integers, and a
     small fixed set of operator/punctuation tokens only -- anything else
     is a syntax error).
  2. Parse into a tiny AST (Num / Ident / Call / BinOp / UnaryOp) with a
     hand-rolled recursive-descent parser.
  3. Evaluate the AST against a locked-down "environment" (a dict built
     by geometry_compiler.solver from the ProblemIR's declared entities
     and unknown_scalars) using explicit sympy operations -- dot, cross,
     norm, vector +/-, scalar */ vector, ** for scalar powers, and /
     for scalar division.

No `Ident` name is ever resolved except by exact lookup in the supplied
environment (or the two reserved constants `zero` / `0`), so an
expression string cannot reach anything outside what the caller
explicitly exposed -- there is no way to reference builtins, import
anything, or execute arbitrary code.

Grammar actually implemented (see the module docstring in
geometry_compiler/__init__.py and the README for how this compares to
CONTRACT.md's stated grammar -- there are two deliberate supersets):

    equation   := expr "==" expr
    expr       := add
    add        := mul (("+" | "-") mul)*
    mul        := unary (("*" | "/") unary)*
    unary      := "-" unary | power
    power      := atom ("**" unary)?          # right-associative
    atom       := NUMBER
                | IDENT
                | IDENT "(" [expr ("," expr)*] ")"
                | "(" expr ")"

Two places where we intentionally go beyond the literal text of
CONTRACT.md's grammar section, because the fixtures actually use them:
  * bare integer literals other than "0" (e.g. the "4" in
    "dot(c, n) == 4" in fixtures/problem_ir/M26S1J21Q55.json), even
    though CONTRACT.md says "0 and zero... the only literals allowed".
  * a "/" division operator (e.g. "dot(F, v) / norm(v)" in
    fixtures/problem_ir/M26S1J21Q64.json's query_expr), which is not
    listed among "+ - * **" in CONTRACT.md's grammar bullet.
Both are flagged again in geometry_compiler/README.md.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union

from .exceptions import MiniLanguageSyntaxError

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
      (?P<NUMBER>\d+)
    | (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
    | (?P<POW>\*\*)
    | (?P<EQ>==)
    | (?P<PLUS>\+)
    | (?P<MINUS>-)
    | (?P<STAR>\*)
    | (?P<SLASH>/)
    | (?P<LPAREN>\()
    | (?P<RPAREN>\))
    | (?P<COMMA>,)
    | (?P<WS>\s+)
    """,
    re.VERBOSE,
)

_ALLOWED_CHARS = set("()+-*/,=0123456789 \t")


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    pos: int


def tokenize(source: str) -> list[Token]:
    # Reject anything containing characters that aren't letters, digits,
    # underscore, whitespace, or one of the operator/punctuation symbols
    # our grammar defines. This is a belt-and-suspenders check on top of
    # the regex whitelist below (e.g. catches stray unicode look-alikes).
    for ch in source:
        if ch.isalnum() or ch == "_" or ch in _ALLOWED_CHARS:
            continue
        raise MiniLanguageSyntaxError(
            f"illegal character {ch!r} in mini-language expression: {source!r}"
        )

    tokens: list[Token] = []
    pos = 0
    n = len(source)
    while pos < n:
        m = _TOKEN_RE.match(source, pos)
        if not m:
            raise MiniLanguageSyntaxError(
                f"could not tokenize {source!r} at position {pos} "
                f"(near {source[pos:pos + 10]!r})"
            )
        kind = m.lastgroup
        text = m.group()
        if kind != "WS":
            tokens.append(Token(kind, text, pos))
        pos = m.end()
    tokens.append(Token("EOF", "", n))
    return tokens


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Num:
    value: int


@dataclass(frozen=True)
class Ident:
    name: str


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple["Node", ...]


@dataclass(frozen=True)
class BinOp:
    op: str  # "+", "-", "*", "/", "**"
    left: "Node"
    right: "Node"


@dataclass(frozen=True)
class UnaryOp:
    op: str  # "-"
    operand: "Node"


Node = Union[Num, Ident, Call, BinOp, UnaryOp]


# ---------------------------------------------------------------------------
# Recursive-descent parser
# ---------------------------------------------------------------------------


class _Parser:
    def __init__(self, tokens: list[Token], source: str):
        self.tokens = tokens
        self.i = 0
        self.source = source

    def _peek(self) -> Token:
        return self.tokens[self.i]

    def _advance(self) -> Token:
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def _expect(self, kind: str) -> Token:
        tok = self._peek()
        if tok.kind != kind:
            raise MiniLanguageSyntaxError(
                f"expected {kind} but found {tok.kind!r} ({tok.text!r}) "
                f"at position {tok.pos} in {self.source!r}"
            )
        return self._advance()

    # expr := add
    def parse_expr(self) -> Node:
        return self._parse_add()

    def _parse_add(self) -> Node:
        node = self._parse_mul()
        while self._peek().kind in ("PLUS", "MINUS"):
            op_tok = self._advance()
            rhs = self._parse_mul()
            node = BinOp("+" if op_tok.kind == "PLUS" else "-", node, rhs)
        return node

    def _parse_mul(self) -> Node:
        node = self._parse_unary()
        while self._peek().kind in ("STAR", "SLASH"):
            op_tok = self._advance()
            rhs = self._parse_unary()
            node = BinOp("*" if op_tok.kind == "STAR" else "/", node, rhs)
        return node

    def _parse_unary(self) -> Node:
        if self._peek().kind == "MINUS":
            self._advance()
            return UnaryOp("-", self._parse_unary())
        return self._parse_power()

    def _parse_power(self) -> Node:
        node = self._parse_atom()
        if self._peek().kind == "POW":
            self._advance()
            rhs = self._parse_unary()  # right-associative
            node = BinOp("**", node, rhs)
        return node

    def _parse_atom(self) -> Node:
        tok = self._peek()
        if tok.kind == "NUMBER":
            self._advance()
            return Num(int(tok.text))
        if tok.kind == "LPAREN":
            self._advance()
            node = self.parse_expr()
            self._expect("RPAREN")
            return node
        if tok.kind == "IDENT":
            self._advance()
            if self._peek().kind == "LPAREN":
                self._advance()
                args = []
                if self._peek().kind != "RPAREN":
                    args.append(self.parse_expr())
                    while self._peek().kind == "COMMA":
                        self._advance()
                        args.append(self.parse_expr())
                self._expect("RPAREN")
                return Call(tok.text, tuple(args))
            return Ident(tok.text)
        raise MiniLanguageSyntaxError(
            f"unexpected token {tok.kind!r} ({tok.text!r}) at position "
            f"{tok.pos} in {self.source!r}"
        )


def parse_expr(source: str) -> Node:
    """Parse a single mini-language expression (no top-level `==`)."""
    tokens = tokenize(source)
    parser = _Parser(tokens, source)
    node = parser.parse_expr()
    if parser._peek().kind != "EOF":
        tok = parser._peek()
        raise MiniLanguageSyntaxError(
            f"trailing input starting with {tok.text!r} at position "
            f"{tok.pos} in {source!r}"
        )
    return node


def parse_equation(source: str) -> tuple[Node, Node]:
    """Split `LHS == RHS` and parse both sides.

    Exactly one top-level `==` is expected (mini-language equations are
    not chained, e.g. no `a == b == c`).
    """
    tokens = tokenize(source)
    eq_positions = [i for i, t in enumerate(tokens) if t.kind == "EQ"]
    if len(eq_positions) != 1:
        raise MiniLanguageSyntaxError(
            f"expected exactly one '==' in equation, found "
            f"{len(eq_positions)} in {source!r}"
        )
    idx = eq_positions[0]
    lhs_tokens = tokens[:idx] + [Token("EOF", "", tokens[idx].pos)]
    rhs_tokens = tokens[idx + 1 :]

    lhs_parser = _Parser(lhs_tokens, source)
    lhs = lhs_parser.parse_expr()
    if lhs_parser._peek().kind != "EOF":
        raise MiniLanguageSyntaxError(f"malformed LHS in equation {source!r}")

    rhs_parser = _Parser(rhs_tokens, source)
    rhs = rhs_parser.parse_expr()
    if rhs_parser._peek().kind != "EOF":
        raise MiniLanguageSyntaxError(f"malformed RHS in equation {source!r}")

    return lhs, rhs
