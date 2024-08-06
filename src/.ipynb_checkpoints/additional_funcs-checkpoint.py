from itertools import combinations
import pickle
import networkx as nx
import networkx.algorithms.isomorphism as iso
from . import networks as nws
from itertools import product
from matplotlib.colors import to_hex
from pyRDTP.operations.analysis import bond_analysis, insaturation_check
from pyRDTP.operations import graph

INTERPOL = {'O-H' : {'alpha': 0.39, 'beta': 0.89},
            'C-H' : {'alpha': 0.63, 'beta': 0.81},
            'H-C' : {'alpha': 0.63, 'beta': 0.81},
            'C-C' : {'alpha': 1.00, 'beta': 0.64},
            'C-O' : {'alpha': 1.00, 'beta': 1.24},
            'C-OH': {'alpha': 1.00, 'beta': 1.48}}

BOX_TMP = """<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="20">
  <TR>
    <TD COLSPAN="3" BGCOLOR="{}">{}</TD>
  </TR>
  <TR>
    <TD BGCOLOR="{}">{}</TD>
    <TD BGCOLOR="{}">{}</TD>
    <TD BGCOLOR="{}">{}</TD>
  </TR>
  <TR>
    <TD BGCOLOR="{}">{}</TD>
    <TD BGCOLOR="{}">{}</TD>
    <TD BGCOLOR="{}">{}</TD>
  </TR>
</TABLE>>
"""
BOX_TMP_3 = """<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="20">
  <TR>
    <TD COLSPAN="3" BGCOLOR="{}">{}</TD>
  </TR>
  <TR>
    <TD BGCOLOR="{}">{}</TD>
    <TD BGCOLOR="{}">{}</TD>
    <TD BGCOLOR="{}">{}</TD>
  </TR>
  <TR>
    <TD BGCOLOR="{}">{}</TD>
    <TD BGCOLOR="{}">{}</TD>
    <TD BGCOLOR="{}">{}</TD>
  </TR>
  <TR>
    <TD BGCOLOR="{}">{}</TD>
    <TD BGCOLOR="{}">{}</TD>
    <TD BGCOLOR="{}">{}</TD>
  </TR>
</TABLE>>
"""

BOX_TMP_0 = """<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="20">
  <TR>
    <TD BGCOLOR="{0}">{1}</TD>
  </TR>
</TABLE>>
"""

BOX_TMP_flat = """<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="20">
  <TR>
    <TD COLSPAN="3" BGCOLOR="{0}">{1}</TD>
  </TR>
  <TR>
    <TD BGCOLOR="{6}">{7}</TD>
    <TD BGCOLOR="{12}">{13}</TD>
    <TD BGCOLOR="{18}">{19}</TD>
  </TR>
</TABLE>>
"""


ELEM_WEIGTHS = {'H': 1., 'C': 12, 'O': 16}


def calculate_weigth(elems):
    return sum([ELEM_WEIGTHS[key] * value for key, value in elems.items()])


def break_bonds(molecule):
    """Generate all possible C-O, C-C and C-OH bonds for the given molecule.

    Args:
        molecule (obj:`pyRDTP.molecule.Molecule`): Molecule that will be used
            to detect the breakeages.

    Returns:
        dict with the keys being the types of bond breakings containing the 
        obj:`nx.DiGraph` with the generated molecules after the breakage.
    """
    bonds = {'C-O': [],
             'C-C': [],
             'C-OH': []}
    bond_pack = bond_analysis(molecule)
    new_graph = graph.generate(molecule)
    for bond_type in (('O', 'C'), ('C', 'C')):
        for pair in bond_pack.bond_search(bond_type):
            tmp_graph = new_graph.copy()
            tmp_graph.remove_edge(*pair.atoms)
            sub_mols = list(nx.connected_component_subgraphs(tmp_graph))
            if ('O', 'C') == bond_type:
                oh_bond = False
                for atom in pair.atoms:
                    if atom.element == 'O':
                        if 'H' in [con.element for con in atom.connections]:
                            oh_bond = True
                    else:
                        continue
                if oh_bond:
                    bonds['C-OH'].append(sub_mols)
                else:
                    bonds['C-O'].append(sub_mols)

            else:
                bonds['C-C'].append(sub_mols)
    return bonds


def break_and_connect(network, surface='000000'):
    """For an entire network, perform a breakage search (see `break_bonds`) for
    all the intermediates, search if the generated submolecules belong to the
    network and if affirmative, create an obj:`networks.TransitionState` object
    that connect all the species.

    Args:
        network (obj:`networks.OrganicNetwork`): Network in which the function
            will be performed.
        surface (obj:`networks.Intermediate`, optional): Surface intermediate.
            defaults to '000000'.

    Returns:
        list of obj:`networks.TransitionState` with all the generated
        transition states.
    """
    cate = iso.categorical_node_match(['elem', 'elem'], ['H', 'O'])
    ts_lst = []
    for intermediate in network.intermediates.values():
        sub_graphs = break_bonds(intermediate.molecule)
        for r_type, graphs in sub_graphs.items():
            for graph_numb in graphs:
                new_ts = nws.TransitionState(r_type=r_type)
                in_comp = [[surface, intermediate], []]
                for ind_graph in graph_numb:
                    for loop_inter in network.intermediates.values():
                        if len(loop_inter.graph) != len(ind_graph):
                            continue
                        if graph.elem_inf(loop_inter.graph) != graph.elem_inf(ind_graph):
                            continue
                        if nx.is_isomorphic(loop_inter.graph, ind_graph,
                                            node_match=cate):
                            in_comp[1].append(loop_inter)
                            if len(in_comp[1]) == 2:
                                break
                chk_lst = [0, 0]
                for index, item in enumerate(in_comp):
                    for mol in item:
                        if mol.is_surface:
                            continue
                        chk_lst[index] += len(mol.molecule.atoms)

                if chk_lst[0] != chk_lst[1] and chk_lst[0] != chk_lst[1]/2:
                    continue

                new_ts.components = in_comp
                for item in ts_lst:
                    if item.components == new_ts.components:
                        break
                else:
                    ts_lst.append(new_ts)
    return ts_lst


def change_r_type(network):
    """Given a network, search if the C-H breakages are correct, and if not
    correct them.

    Args:
        network (obj:`networks.OrganicNetwork`): Network in which the function
            will be performed.
    """
    for trans in network.t_states:
        if trans.r_type not in ['H-C', 'C-H']:
            continue
        flatten = [list(comp) for comp in trans.components]
        flatten_tmp = []
        for comp in flatten:
            flatten_tmp += comp
        flatten = flatten_tmp
        flatten.sort(key=lambda x: len(x.molecule.atoms), reverse=True)
        flatten = flatten[1:3]  # With this step we will skip
        bonds = [bond_analysis(comp.molecule) for comp in flatten]
        bond_len = [item.bond_search(('C', 'O')) for item in bonds]
        if bond_len[0] == bond_len[1]:
            trans.r_type = 'C-H'
        else:
            trans.r_type = 'O-H'


def calculate_ts_energy(t_state):
    """Calculate the ts energy of a transition state using the interpolation
     formula

    Args:
        t_state (obj:`networks.TransitionState`): TS that will be evaluated.

    Returns:
        max value between the computed energy and the initial and final state
        energies.
    """
    components = [list(comp) for comp in t_state.bb_order()]
    alpha = INTERPOL[t_state.r_type]['alpha']
    beta = INTERPOL[t_state.r_type]['beta']
    e_is = [comp.energy for comp in components[0]]
    e_fs = [comp.energy for comp in components[1]]
    tmp_vals = []
    for item in [e_is, e_fs]:
        if len(item) == 1:
            item *= 2
        item = sum(item)
        tmp_vals.append(item)
    e_is, e_fs = tmp_vals
    ts_ener = alpha * e_fs + (1. - alpha) * e_is + beta
    return max(e_is, e_fs, ts_ener)


def generate_electron(t_state, electron='e-', proton='H+', def_h='000000', ener_gap=0.):
    new_ts = nws.TransitionState(r_type='C-H', is_electro=True)
    new_components = []
    for comp in t_state.components:
        tmp_comp = []
        for ind in comp:
            if ind.is_surface:
                tmp_comp.append(electron)
            elif ind.code == def_h:
                tmp_comp.append(proton)
            else:
                tmp_comp.append(ind)
        new_components.append(frozenset(tmp_comp))
    new_ts.components = new_components
    if t_state.energy is not None:
        new_ts.energy += ener_gap
    return new_ts


def search_electro(network, electron='e-', proton='H+', def_h='000000', ener_gap=0.):
    electro_states = []
    for t_state in network.t_states:
        if t_state.r_type not in ['C-H', 'H-C']:
            continue
        tmp_el = generate_electron(t_state, electron=electron,
                                   proton=proton, def_h=def_h, ener_gap=ener_gap)
        electro_states.append(tmp_el)
    return electro_states


def generate_colors(inter, colormap, norm):
    """Given an intermediate with associated transition states, a colormap and
    a norm, return the colors of the different transition states depending of
    their energy.

    Args:
        inter (obj:`networks.Intermediate`): Intermediate that will be
            evaluated.
        colormap (obj:`matplotlib.cm`): Colormap to extract the colors.
        norm (obj:`matplotlib.colors.Normalize`): Norm to convert the energy
            value into a number between 0 and 1.

    Returns:
        2 lists of str both with a len of 7. The first containing the hex
        values of the colors and the second one containing the codes of the
        another part of the reaction.

    Notes:
        Both lists contain the colors and the intermediates taking into account
        the bond breaking type in this order:

        [Intermediate, C-OH, C-O, C-C, C-OH, C-O, C-C]
    """
    keys = ['C-OH', 'C-O', 'C-C']
    white = '#ffffff'
    full_colors = [white] * 9
    full_codes = [''] * 9
    for index, brk in enumerate(keys):
        try:
            colors = []
            codes = []
            if index == 2:
                state = 0
            else:
                state = 1
            for t_state in inter.t_states[state][brk]:
                act_norm = norm(t_state.energy)
                color = to_hex(colormap(act_norm))
                colors.append(color)
                for comp in t_state.components:
                    comp_lst = [mol.code for mol in list(comp) if not
                                mol.is_surface and not len(mol.molecule) == 1]
                    if inter in comp_lst:
                        continue
                    if len(comp_lst) == 2:
                        comp_lst = '{}<br/>{}'.format(*comp_lst)
                    else:
                        try:
                            comp_lst = comp_lst[0]
                        except IndexError:
                            pass
                    if act_norm > 0.5:
                        temp = '<FONT COLOR="#ffffff" POINT-SIZE="10">{}</FONT>'
                    else:
                        temp = '<FONT POINT-SIZE="10">{}</FONT>'
                    codes.append(temp.format(comp_lst))
            if len(colors) == 1:
                colors.append(white)
                colors.append(white)
                codes.append('')
                codes.append('')
            elif len(colors) == 2:
                colors.append(white)
                codes.append('')
            full_colors[index] = colors[0]
            full_colors[index + 3] = colors[1]
            full_colors[index + 6] = colors[2]
            full_codes[index] = codes[0]
            full_codes[index + 3] = codes[1]
            full_codes[index + 6] = codes[2]
        except KeyError:
            continue
    color = norm(inter.energy)
    color = to_hex(colormap(color))
    full_colors.insert(0, color)
    return full_colors, full_codes


def generate_label(formula, colors, codes, html_template=None):
    """Generate a html table with the colors and the codes generted with the
    generate_colorn function.

    Args:
        formula (str): Formula of the intermediate. Will be used as the title
            of the table.
        colors (list of str): List that contains the colors of the transition
            states associated to an intermediate.
        codes (list of str): List that contains the codes of the intermediates
            associated with the other part of the reaction.

    Returns:
        str with the generated html table compatible with dot language to use
        it as a label of a node.
    """
    term = colors[0]
    rest = colors[1:]
    mix = [item for sublist in zip(rest, codes) for item in sublist]
    if html_template:
        label = html_template.format(term, formula, *mix)
    else:
        if len(term) > 6:
            label = BOX_TMP_3.format(term, formula, *mix)
        else:
            label = BOX_TMP.format(term, formula, *mix)
    return label

def code_mol_graph(mol_graph, elems=['O', 'C']):
    """Given a molecule graph generated with the lib:`pyRDTP.operation.graph`
    node, return an str with the formula of the molecule.

    Args:
        mol_graph (obj:nx.Graph): Graph of the molecule.
        elems (list of objects, optional): List containing the elements that
            will be taken into account to walkt through the molecule. Defaults
            to ['O', 'C'].

    Retrns:
        str with the formula of the molecule with the format:
        CH-CO-CH3
    """
    new_graph = nx.DiGraph()
    for node in list(mol_graph.nodes()):
        if node.element in elems:
            new_graph.add_node(node)
    for edge in list(mol_graph.edges()):
        if edge[0].element in elems and edge[1].element in elems:
            new_graph.add_edge(*edge)
            new_graph.add_edge(*edge[::-1])

    max_val = 0
    for pair in combinations(list(new_graph.nodes()), 2):
        try:
            path = nx.shortest_path(new_graph, source=pair[0], target=pair[1])
        except nx.NetworkXNoPath:
            return ''
        path_len = len(path)
        if path_len > max_val:
            max_val = path_len
            longest_path = path

    if max_val == 0:
        longest_path = list(new_graph.nodes())

    path = ''
    for item in longest_path:
        if item.element == 'H':
            continue
        path += item.element
        count_H = 0
        H_lst = []
        oh_numb = []
        for hydro in item.connections:
            if hydro.element == 'H' and hydro not in H_lst:
                H_lst.append(hydro)
                count_H += 1
            if hydro.element == 'O':
                oh_numb.append(len([atom for atom in hydro.connections if
                                   atom.element == 'H']))
        if count_H > 1:
            path += '{}{}'.format('H', count_H)
        elif count_H == 1:
            path += '{}'.format('H')
        for numb in oh_numb:
            if numb == 0:
                path += '(O)'
            elif numb == 1:
                path += '(OH)'
            else:
                path += '(O{})'.format(numb)
        path += '-'
    path = path[:-1]
    return path


def radical_calc(inter):
    """Check if an intermediate is a radical.

    Args:
        inter (obj:`networks.Intermediate`): Intermediate to be tested.
    
    Returns:
        bool with the results of the test.
    """
    new_mol = inter.molecule.copy()
    new_mol.connection_clear()
    new_mol.connectivity_search_voronoi()
    return insaturation_check(new_mol)


def underline_label(label):
    """Add the needed marks to a str to underline it in dot.

    Args:
       label (str): String that will be underlined.

    Returns:
       str with the underline marks.
    """
    temp = '<u>{}</u>'
    new_label = temp.format(label)
    return new_label


def change_color_label(label, color):
    """Add the needed marks to an str to change the font color on dat.

    Args:
        label (str): String to change the font color.
        color (str): Dot compatible color.

    Returns:
        str with the font color marks.
    """
    temp = '<FONT COLOR="{}">{}</FONT>'
    new_label = temp.format(color, label)
    return new_label


def adjust_co2(elements):
    """Given a dict with elements, calculate the reference energy for
    the compound.

    Args:
        elements (dict): C, O, H as key and the number of atoms for every
            element as value.
    """
    GASES_ENER = {'CH4': -24.05681734,
                  'H2O': -14.21877278,
                  'H'  : -3.383197435,
                  'CO2' : -22.96215586}
    pivot_dict = elements.copy()
    for elem in ['O', 'C', 'H']:
        if elem not in pivot_dict:
            pivot_dict[elem] = 0

    energy = GASES_ENER['CO2'] * pivot_dict['C']
    energy += GASES_ENER['H2O'] * (pivot_dict['O'] - 2 * pivot_dict['C'])
    energy += GASES_ENER['H'] * (4 * pivot_dict['C'] + pivot_dict['H']
                                 - 2 * pivot_dict['O'])
    return energy


def read_object(filename):
    """Read a pickle object from the specified file

    Args:
        filename (str): Location of the file.

    Returns:
        Object readed from the pickle file.
    """
    with open(filename, 'rb') as obj_file:
        new_obj = pickle.load(obj_file)
    return new_obj


def write_object(obj, filename):
    """Write the given object to the specified file.

    Args:
        obj (obj): Object that will be written.
        filename (str): Name of the file where the
            object will be stored.
    """
    with open(filename, 'wb') as obj_file:
        new_file = pickle.dump(obj, obj_file)


def clear_graph(graph):
    """Generate a copy of the graph only using the edges and clearing the
    attributes of both nodes and edges.

    Args:
        graph (obj:`nx.DiGraph`): Base graph to clear.

    Returns:
        obj:`nx.Graph` that is a copy of the original without attributes.
    """
    new_graph = nx.Graph()
    for edge in graph.edges():
        new_graph.add_edge(*edge)
    return new_graph


def inverse_edges(graph):
    """Generate a copy of the graph and add additional edges that connect
    the nodes in the inverse direction while keeping the originals.

    Args:
        graph (obj:`nx.Graph`): Base graph to add the inverse edges.

    Returns:
        obj:`nx.Graph` that is a copy of the original but with the addition of
        the inverse edges
    """
    new_graph = graph.copy()
    for edge in graph.edges():
        new_graph.add_edge(edge[1], edge[0])
    return new_graph


def print_intermediates(network, filename='inter.dat'):
    """Print a text file containing the information of the intermediates of a
    network.

    Args:
        network (obj:`OrganicNetwork`): Network containing the intermediates.
        filename (str, optional): Location where the file will be writed.
            Defaults to 'inter.dat'
    """
    header = 'Label          iO    Formula\n'
    inter_str = '{:6} {:.16f} {:.20}\n'
    with open(filename, 'w') as outfile:
        outfile.write(header)
        for label, inter in network.intermediates.items():
            formula = code_mol_graph(inter.graph)
            ener = inter.energy
            out_str = inter_str.format(label, ener, formula)
            outfile.write(out_str)


def print_intermediates_kinetics(network, filename='inter.dat'):
    """Print a text file containing the information of the intermediates of a
    network.

    Args:
        network (obj:`OrganicNetwork`): Network containing the intermediates.
        filename (str, optional): Location where the file will be writed.
            Defaults to 'inter.dat'
    """
    header = 'Label       iO       e   frq\n'
    inter_str = 'i{:6} {: .8f} {:2d} {:.20}\n'
    tmp_frq = '[0,0,0]'
    with open(filename, 'w') as outfile:
        outfile.write(header)
        for label, inter in network.intermediates.items():
            ener = inter.energy
            electrons = inter.electrons
            out_str = inter_str.format(label, ener, electrons, tmp_frq)
            outfile.write(out_str)


def print_t_states_kinetics(network, filename='ts.dat'):
    """Print a text file containing the information of the transition states
    of a network.

    Args:
        network (obj:`OrganicNetwork`): Network containing the transition
            states.
        filename (str, optional): Location where the file will be writed.
            Defaults to 'ts.dat'
    """
    header = '          Label                is1     is2     fs1     fs2        iO      e alpha beta   frq\n'
    tmp_frq = '[0,0,0]'
    inter_str = '{:28} i{:6} i{:6} i{:6} i{:6} {: .8f} {: 2d} {:3.2f} {:2.2f} {:.20}\n'
    with open(filename, 'w') as outfile:
        outfile.write(header)
        for t_state in network.t_states:
            label = t_state.code
            initial = [inter.code for inter in list(t_state.components[0])]
            final = [inter.code for inter in list(t_state.components[1])]

            electrons_fin = sum([inter.electrons for inter in
                                 list(t_state.components[1])])
            electrons_in = sum([inter.electrons for inter in
                                list(t_state.components[0])])
            if len(initial) == 1:
                initial *= 2
                electrons_in *= 2
            if len(final) == 1:
                final *= 2
                electrons_fin *= 2
            order = []
            for item in [initial, final]:
                mols = [network.intermediates[item[0]],
                        network.intermediates[item[1]]]
                if mols[0].is_surface:
                    order.append(item[::-1])
                elif not mols[1].is_surface and (mols[0].molecule.atom_numb <
                                                 mols[1].molecule.atom_numb):
                    order.append(item[::-1])
                else:
                    order.append(item)

            initial, final = order
            electrons = electrons_fin - electrons_in
            ener = t_state.energy
            alpha = INTERPOL[t_state.r_type]['alpha']
            beta = INTERPOL[t_state.r_type]['beta']
            label = ''
            for item in initial + final:
                label += 'i' + item
            outfile.write(inter_str.format(label, *initial, *final,
                                           ener, electrons, alpha,
                                           beta, tmp_frq))


def print_t_states(network, filename='ts.dat'):
    """Print a text file containing the information of the transition states
    of a network.

    Args:
        network (obj:`OrganicNetwork`): Network containing the transition
            states.
        filename (str, optional): Location where the file will be writed.
            Defaults to 'ts.dat'
    """
    header = '          Label               is1    is2    fs1    fs2           iO\n'
    inter_str = '{:20} {:6} {:6} {:6} {:6} {:.16f}\n'
    with open(filename, 'w') as outfile:
        outfile.write(header)
        for t_state in network.t_states:
            label = t_state.code
            initial = [inter.code for inter in list(t_state.components[0])]
            final = [inter.code for inter in list(t_state.components[1])]
            if len(initial) == 1:
                initial *= 2
            if len(final) == 1:
                final *= 2
            ener = t_state.energy
            outfile.write(inter_str.format(label, *initial, *final, ener))

def print_gasses_kinetics(gas_dict, filename='gas.dat'):
    """Print a text file containing the information of the intermediates of a
    network.

    Args:
        network (obj:`OrganicNetwork`): Network containing the intermediates.
        filename (str, optional): Location where the file will be writed.
            Defaults to 'inter.dat'
    """
    header = 'Label    Formula              ads         gas      iO         e   mw   frq\n'
    inter_str = '{:6} {:20} i{:7} {: .8f} {: .8f} {:4.2f} {:2d} {:.20}\n'
    tmp_frq = '[0,0,0]'
    with open(filename, 'w') as outfile:
        outfile.write(header)
        for label, gas in gas_dict.items():
            ener = gas['energy']
            electrons = gas['electrons']
            try:
                formula = code_mol_graph(graph.generate(gas['mol'], voronoi=True), ['C'])
            except nx.NetworkXNoPath:
                formula = gas['mol'].formula
            if not formula:
                formula = gas['mol'].formula
            weight = calculate_weigth(gas['mol'].elements_number)
            out_str = inter_str.format(label, formula, label, ener,
                                       ener, weight, electrons, tmp_frq)
            outfile.write(out_str)


def read_energies(filename):
    """Reads an energy filename and convert it to a dictionary.

    Args:
        filename (str): Location of the file containing the energies.

    Returns:
        Two dictionaris in which keys are the code values of the
        intermediates and the associated values are the energy of the
        intermediates.
        The first dictionary contains the values that do not contain failed,
        warning or WARNING at the end of their code. Moreover, if multiple
        codes with the same 6 characters start are found, only the one with
        the lesser energy is selected.
    """
    ener_dict = {}
    discard = {}
    repeated = {}

    with open(filename, 'r') as infile:
        lines = infile.readlines()

    for sg_line in lines:
        try:
            code, energy = sg_line.split()
        except ValueError:
            discard[code[:6]] = 0
        energy = float(energy)  # Energy needs to be a float

        if code[0] == 'i':
            code = code[1:]

        if len(code) > 6:
            continue
            # init_code = code[:6]
            # if code.endswith(('WARNING', 'warning', 'failed')):
            #     discard[init_code] = energy
            #     continue
            # if init_code not in repeated:
            #     repeated[init_code] = []
            # repeated[init_code].append(energy)
            # continue
        ener_dict[code] = energy
#    for code, energies in repeated.items():
#        ener_dict[code] = min(energies)

    return ener_dict, discard


def adjust_electrons_old(elements):
    """Given a dict with elements, calculate the reference energy for
    the compound.

    Args:
        elements (dict): C, O, H as key and the number of atoms for every
            element as value.
    """
    pivot_dict = elements.copy()
    for elem in ['O', 'C', 'H']:
        if elem not in pivot_dict:
            pivot_dict[elem] = 0

    electrons = (4 * pivot_dict['C'] + pivot_dict['H']
                 - 2 * pivot_dict['O'])
    return electrons


def adjust_electrons(molecule):
    elements = molecule.elements_number
    pivot_dict = elements.copy()
    if 'H' not in pivot_dict:
        pivot_dict['H'] = 0

    electrons = pivot_dict['H'] + search_alcoxy(molecule)
    return electrons


def select_larger_inter(inter_lst):
    """Given a list of Intermediates, select the intermediate with the
    highest number of atoms.

    Args:
        inter_lst(list of obj:`networks.Intermediate`): List that will be
            evaluated.
    Returns:
        obj:`networks.Intermediate` with the bigger molecule.
    """
    atoms = [len(inter.mol.atoms) for inter in inter_lst]
    inter_max = [0, 0]
    for size, inter in zip(atoms, inter_lst):
        if size > inter_max[0]:
            inter_max[0] = size
            inter_max[1] = inter
    return inter_max[1]


def search_electro_ts(network, electron='e-', proton='H', water='H2O', ener_up=0.05):
    """Search for all the possible electronic transition states of a network.

    Args:
        network (obj:`networks.OrganicNetwork`): Network in wich the electronic
            states will be generated.
        electron (any object): Object that represents an electron.
        proton (any object): Object that represents a proton.
        water (any object): Object that represents a water molecule.
    """
    cate = iso.categorical_node_match(['elem', 'elem'], ['H', 'O'])
    electro_ts = []
    for inter in network.intermediates.values():
        if 'O' not in inter.molecule.elements_number:
            continue
        tmp_oxy = inter.molecule['O']
        for index, _ in enumerate(tmp_oxy):
            tmp_mol = inter.molecule.copy()
            oxygen = tmp_mol['O'][index]

            hydrogen = [conect for conect in oxygen.connections if
                        conect.element == 'H']
            if not hydrogen:
                continue

            tmp_mol.atom_remove(hydrogen[0])
            tmp_mol.connection_clear()
            tmp_mol.connectivity_search_voronoi()
            tmp_graph = graph.generate(tmp_mol)


            candidates = network.search_graph(tmp_graph, cate=cate)
            for new_inter in candidates:
                oh_comp = [[new_inter, proton], [inter, electron]]
                oh_ts = nws.TransitionState(components=oh_comp,
                                            r_type='O-H', is_electro=True)
                oh_ts.energy = max((inter.energy, new_inter.energy)) + ener_up
                electro_ts.append(oh_ts)

            tmp_mol.atom_remove(oxygen)
            tmp_mol.connection_clear()
            try:
                tmp_mol.connectivity_search_voronoi()
            except ValueError:
                continue
            tmp_graph = graph.generate(tmp_mol)

            
            candidates = network.search_graph(tmp_graph, cate=cate)
            for new_inter in candidates:
                wa_comp = [[inter, proton], [new_inter, water]]            
                wa_ts = nws.TransitionState(components=wa_comp,
                        r_type='O-H', is_electro=True)
                wa_ts.energy = max((inter.energy, new_inter.energy)) + ener_up
                electro_ts.append(wa_ts)

    return electro_ts


def print_electro_kinetics(ts_lst, filename='elec.dat'):
    """Print a text file containing the information of the transition states
    of a network.

    Args:
        network (obj:`OrganicNetwork`): Network containing the transition
            states.
        filename (str, optional): Location where the file will be writed.
            Defaults to 'ts.dat'
    """
    header = '              Label                is1     is2     fs1     fs2        iO      e    frq\n'
    tmp_frq = '[0,0,0]'
    inter_str = '{:32} {:7} {:7} {:7} {:7} {: .8f} {: 2d} {:.20}\n'
    with open(filename, 'w') as outfile:
        outfile.write(header)
        for t_state in ts_lst:
            ordered = t_state.components
            initial = [inter.code for inter in list(ordered[0])]
            final = [inter.code for inter in list(ordered[1])]

            electrons_fin = sum([inter.electrons for inter in
                                 list(ordered[1])])
            electrons_in = sum([inter.electrons for inter in
                                list(ordered[0])])

            for item in [initial, final]:
                for index, _ in enumerate(item):
                    if not item[index]:
                        item[index] = 'None'
                    elif item[index][0] != 'g':
                        item[index] = 'i' + item[index]

            if len(initial) == 1:
                initial *= 2
            if len(final) == 1:
                final *= 2
            order = []
            for item in [initial, final]:
                if item[0] == 'None' or item[0].startswith('g'):
                    order.append(item[::-1])
                else:
                    order.append(item)
            initial, final = order
            label = ''
            for item in initial + final:
                if item == 'None':
                    label += 'g000000'
                else:
                    label += item
            electrons = electrons_fin - electrons_in
            ener = t_state.energy
            outfile.write(inter_str.format(label, *initial, *final,
                                           ener, electrons, tmp_frq))

def search_alcoxy(molecule):
    mol = molecule.copy()
    mol.connectivity_search_voronoi()
    oxy = mol['O']
    alco_numb = 0
    for item in oxy:
        if len(item.connections) > 1 or len(item.connections) == 0:
            continue
        elif item.connections[0].element != 'C':
            continue
        elif len(item.connections[0].connections) != 4:
            continue
        else:
            alco_numb += 1
    return alco_numb


