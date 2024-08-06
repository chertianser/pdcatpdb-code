#!/usr/bin/env python3

from abc import ABC
from collections.abc import Sequence
from itertools import chain
from typing import NewType, Literal
from pykinetic.classes import (ChemicalSystem, Energy, Compound, Reaction,
                               TransitionState)
from . import networks as nw


UnitType = Literal["kcal/mol", "hartree", "eV", "cm-1", "kJ/mol", "J/mol", "K",
                  "J", "Hz"]

def nws_to_compound(
        inter: nw.Intermediate
        , unit: UnitType
) -> Compound:
    return Compound(inter.code, Energy(inter.energy, unit))

def nws_to_reaction(
        ts: nw.TransitionState
        , unit: UnitType
) -> TransitionState:
    to_c = lambda xs: tuple(nws_to_compound(i,unit=unit) for i in xs)
    compounds = tuple(to_c(c) for c in ts.components)
    ts = TransitionState(Energy(ts.energy, unit),label=ts.code)
    return  Reaction(reactants=compounds[0], products=compounds[1], TS=ts)

def nws_to_cs(
        net: nw.OrganicNetwork
        , unit: UnitType = 'kcal/mol'
        , temp: float = 298.2
) -> ChemicalSystem:
    cs = ChemicalSystem(T=temp, unit=unit)
    reactions = [nws_to_reaction(ts, unit) for ts in net.t_states]
    compounds = list(set(chain.from_iterable((list(r.compounds.keys()) for r in reactions))))
    cs.cextend(compounds)
    cs.rextend(reactions)
    cs.cupdate(update_keys=True)
    cs.rupdate()
    return cs
