from .moniter_reaction import Reaction, BaseProcedures
import sys

num_reactions = sys.argv[0]

rxn = Reaction()
rxn.generate_experiment_details(
    num_reactions=num_reactions,
    temp=60
    )
rxn.run_reaction()