#!/usr/bin/env python3

from abc import ABC
from itertools import chain
from typing import NewType, Literal, TypeGuard, List, Dict, Tuple
from pykinetic.classes import (ChemicalSystem, Energy, Compound, Reaction,
                               TransitionState, DiffusionTS)
from . import networks as nw


UnitType = Literal["kcal/mol", "hartree", "eV", "cm-1", "kJ/mol", "J/mol", "K",
                  "J", "Hz"]

def nws_to_compound(
    inter: nw.Intermediate
    , unit: UnitType
) -> Compound:
    return Compound(inter.code, Energy(inter.energy, unit))

def nws_to_ts(
    ts: nw.TransitionState
    , unit: UnitType
    , compounds: Dict[str, Compound] | None = None
    , diffusion: Tuple[float, float] | None = None
) -> TransitionState:
    match compounds:
        case None:
            to_c = lambda xs: tuple(nws_to_compound(i,unit=unit) for i in xs)
        case dict():
            to_c = lambda xs: tuple(compounds[i.code] for i in xs)
    inters = tuple(to_c(c) for c in ts.components)
    react = [Reaction(reactants=inters[0], products=inters[1])]
    if ts.backwards:
        react.append(Reaction(reactants=inters[1], products=inters[0]))


    ts_type = TransitionState
    energy = Energy(ts.energy, unit)
    if diffusion:
        barrier, threshold = diffusion
        get_diff = lambda x: energy - sum([i.energy for i in inters[x]]) < threshold
        if (get_diff(0) or get_diff(1)):
            ts_type = DiffusionTS
            energy = Energy(barrier, unit)

    out_ts = ts_type(
        energy
        , reactions=react
        , label=ts.code
    )

    react[0].TS = out_ts
    if ts.backwards:
        react[1].TS = out_ts

    out_ts.reactions.extend(react)
    return out_ts

def nws_to_cs(
    net: nw.OrganicNetwork
    , unit: UnitType = 'kcal/mol'
    , temp: float = 298.15
    , diffusion: Tuple[float, float] | None = None
) -> ChemicalSystem:
    cs = ChemicalSystem(T=temp, unit=unit)
    compounds = {s: nws_to_compound(i, unit) for s, i in net.intermediates.items()}
    t_states = [nws_to_ts(ts, unit, compounds, diffusion) for ts in net.t_states]
    reactions = chain.from_iterable([r.reactions for r in t_states])
    cs.cextend(list(compounds.values()))
    cs.rextend(list(reactions))
    return cs

def test (x: TypeGuard[int]) -> List:
    return nws_to_cs()

def test2 () -> List:
    test("s")
    return test("s")
