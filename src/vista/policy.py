"""Monotone receiver policy language (Sec 3.3) and policy frontier compilation (N1, Sec 20).

Grammar (case-insensitive keywords):
    expr   := term ('OR' term)*
    term   := factor ('AND' factor)*
    factor := '(' expr ')' | NAME ('>=' | '=') LEVEL
'=' is accepted as '>=' (monotone language; e.g. device_continuity = established is the top level).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np

from .claims import Schema


@dataclass(frozen=True)
class Atom:
    i: int
    theta: int


@dataclass(frozen=True)
class And:
    children: tuple


@dataclass(frozen=True)
class Or:
    children: tuple


Node = Atom | And | Or


# ----------------------------------------------------------------------------- evaluation
def evaluate(node: Node, prof) -> bool:
    if isinstance(node, Atom):
        return prof[node.i] >= node.theta
    if isinstance(node, And):
        return all(evaluate(c, prof) for c in node.children)
    return any(evaluate(c, prof) for c in node.children)


def evaluate_many(node: Node, states: np.ndarray) -> np.ndarray:
    """Vectorised evaluation over a (N x n) array of profiles."""
    if isinstance(node, Atom):
        return states[:, node.i] >= node.theta
    parts = [evaluate_many(c, states) for c in node.children]
    out = parts[0].copy()
    for p in parts[1:]:
        out = (out & p) if isinstance(node, And) else (out | p)
    return out


def atoms(node: Node) -> list[Atom]:
    if isinstance(node, Atom):
        return [node]
    out = []
    for c in node.children:
        out.extend(atoms(c))
    return out


def is_read_once(node: Node) -> bool:
    """Each attribute occurs at most once (makes additive MDPS a tree DP)."""
    at = [a.i for a in atoms(node)]
    return len(at) == len(set(at))


def depth(node: Node) -> int:
    if isinstance(node, Atom):
        return 0
    return 1 + max(depth(c) for c in node.children)


def to_string(node: Node, schema: Schema | None = None) -> str:
    if isinstance(node, Atom):
        if schema is None:
            return f"a{node.i}>={node.theta}"
        return f"{schema.attributes[node.i].name} >= {schema.attributes[node.i].levels[node.theta]}"
    sep = " AND " if isinstance(node, And) else " OR "
    return "(" + sep.join(to_string(c, schema) for c in node.children) + ")"


def policy_hash(node: Node, schema: Schema) -> str:
    return hashlib.sha256((schema.version + "|" + to_string(node, schema)).encode()).hexdigest()[:32]


# ----------------------------------------------------------------------------- parser
_TOK = re.compile(r"\s*(\(|\)|>=|=|[A-Za-z0-9_.<>]+)")


def parse(text: str, schema: Schema) -> Node:
    toks = []
    pos = 0
    text = text.strip()
    while pos < len(text):
        m = _TOK.match(text, pos)
        if not m:
            raise SyntaxError(f"cannot tokenise policy at {text[pos:]!r}")
        toks.append(m.group(1))
        pos = m.end()
    k = [0]

    def peek():
        return toks[k[0]] if k[0] < len(toks) else None

    def take(expected=None):
        t = peek()
        if t is None or (expected and t.upper() != expected):
            raise SyntaxError(f"expected {expected} got {t}")
        k[0] += 1
        return t

    def expr():
        parts = [term()]
        while peek() and peek().upper() == "OR":
            take("OR")
            parts.append(term())
        return parts[0] if len(parts) == 1 else Or(tuple(parts))

    def term():
        parts = [factor()]
        while peek() and peek().upper() == "AND":
            take("AND")
            parts.append(factor())
        return parts[0] if len(parts) == 1 else And(tuple(parts))

    def factor():
        if peek() == "(":
            take("(")
            e = expr()
            take(")")
            return e
        name = take()
        op = take()
        if op not in (">=", "="):
            raise SyntaxError(f"monotone policies only allow >= or =, got {op}")
        lab = take()
        th = schema.level(name, lab)
        if th == 0:
            raise SyntaxError(f"threshold 0 on {name} is trivially true; remove it")
        return Atom(schema.index(name), th)

    node = expr()
    if peek() is not None:
        raise SyntaxError(f"trailing tokens {toks[k[0]:]}")
    return node


# ----------------------------------------------------------------------------- frontier (N1)
def minimize(profiles: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Minimal elements (antichain) of a set of profiles under the product order (chunk-vectorised)."""
    if len(profiles) == 0:
        return profiles
    P = np.unique(np.asarray(profiles, dtype=np.int16), axis=0)
    P = P[np.argsort(P.sum(1), kind="stable")]
    keep = np.ones(len(P), dtype=bool)
    for a in range(0, len(P), chunk):
        blk = P[a:a + chunk]
        cand = P[: a + len(blk)]  # only rows with smaller-or-equal sum can dominate
        dom = (cand[None, :, :] <= blk[:, None, :]).all(2)  # dom[i, j]: cand j <= blk i
        dom[np.arange(len(blk)), a + np.arange(len(blk))] = False  # unique rows: ignore self
        keep[a:a + len(blk)] = ~dom.any(1)
    return P[keep]


def frontier(node: Node, n: int, max_size: int = 200_000) -> np.ndarray:
    """Algorithm FRONTIER (Sec 20.3): minimal satisfying profiles M(Phi) = prime implicants."""
    if isinstance(node, Atom):
        v = np.zeros((1, n), dtype=np.int16)
        v[0, node.i] = node.theta
        return v
    fr = [frontier(c, n, max_size) for c in node.children]
    if isinstance(node, Or):
        return minimize(np.vstack(fr))
    acc = fr[0]
    for f in fr[1:]:
        joined = np.maximum(acc[:, None, :], f[None, :, :]).reshape(-1, n)
        if joined.shape[0] > max_size:
            raise OverflowError(f"frontier exceeds {max_size} elements")
        acc = minimize(joined)
    return acc


def frontier_bruteforce(node: Node, states: np.ndarray) -> np.ndarray:
    sat = states[evaluate_many(node, states)]
    return minimize(sat)


def to_cnf(node: Node, max_clauses: int = 5000) -> list[list[Atom]]:
    """Distribute to CNF (list of clauses, each a list of atoms). Exponential in the worst case."""
    if isinstance(node, Atom):
        return [[node]]
    if isinstance(node, And):
        out = []
        for c in node.children:
            out.extend(to_cnf(c, max_clauses))
        return out
    acc = [[]]
    for c in node.children:
        cc = to_cnf(c, max_clauses)
        acc = [a + b for a in acc for b in cc]
        if len(acc) > max_clauses:
            raise OverflowError("CNF too large")
    return acc


# ----------------------------------------------------------------------------- generators
def _rand_atom(schema: Schema, rng, attr_pool=None) -> Atom:
    i = int(rng.choice(attr_pool)) if attr_pool is not None else int(rng.integers(schema.n))
    d = schema.attributes[i].d
    return Atom(i, int(rng.integers(1, d + 1)))


def random_dnf(schema: Schema, m_terms: int, k_atoms: int, rng) -> Node:
    terms = []
    for _ in range(m_terms):
        attrs = rng.choice(schema.n, size=min(k_atoms, schema.n), replace=False)
        ats = [Atom(int(i), int(rng.integers(1, schema.attributes[int(i)].d + 1))) for i in attrs]
        terms.append(ats[0] if len(ats) == 1 else And(tuple(ats)))
    return terms[0] if len(terms) == 1 else Or(tuple(terms))


def random_cnf(schema: Schema, m_clauses: int, width: int, rng) -> Node:
    clauses = []
    for _ in range(m_clauses):
        attrs = rng.choice(schema.n, size=min(width, schema.n), replace=False)
        ats = [Atom(int(i), int(rng.integers(1, schema.attributes[int(i)].d + 1))) for i in attrs]
        clauses.append(ats[0] if len(ats) == 1 else Or(tuple(ats)))
    return clauses[0] if len(clauses) == 1 else And(tuple(clauses))


def random_tree(schema: Schema, depth_: int, branching: int, rng, read_once: bool = False,
                top: str = "or") -> Node:
    """Alternating AND/OR tree. With read_once=True each attribute is used at most once."""
    pool = list(rng.permutation(schema.n)) if read_once else None

    def build(d, kind):
        if d == 0:
            if read_once:
                if not pool:
                    raise ValueError("not enough attributes for a read-once tree")
                i = int(pool.pop())
                return Atom(i, int(rng.integers(1, schema.attributes[i].d + 1)))
            return _rand_atom(schema, rng)
        b = int(rng.integers(2, branching + 1))
        kids = []
        for _ in range(b):
            if read_once and not pool:
                break
            kids.append(build(d - 1, "and" if kind == "or" else "or"))
        if len(kids) == 1:
            return kids[0]
        return Or(tuple(kids)) if kind == "or" else And(tuple(kids))

    return build(depth_, top)
