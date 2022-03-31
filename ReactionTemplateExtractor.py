import re
from copy import deepcopy
from numpy.random import shuffle
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit
rdkit.rdBase.DisableLog('rdApp.*')

VERBOSE = False
USE_STEREOCHEMISTRY = True
MAXIMUM_NUMBER_UNMAPPED_PRODUCT_ATOMS = 5
INCLUDE_ALL_UNMAPPED_REACTANT_ATOMS   = True

def Mols_From_SMILES_List(All_SMILES):
    Mols = []
    for smiles in All_SMILES:
        if not smiles:
            continue
        Mols.append(Chem.MolFromSmiles(smiles))
    return Mols

def Replace_Deuterated(SMILES):
    return re.sub('\[2H\]', r'[H]', SMILES)

def Clear_MapNumber(Mol):
    [a.ClearProp('molAtomMapNumber') for a in Mol.GetAtoms() if a.HasProp('molAtomMapNumber')]
    return Mol

def Get_Tagged_Atoms_From_Mols(Mols):
    Atoms     = []
    Atom_Tags = []
    for mol in Mols:
        New_Atoms, New_Atom_Tags = Get_Tagged_Atoms_From_Mol(mol)
        Atoms     += New_Atoms
        Atom_Tags += New_Atom_Tags
    return Atoms, Atom_Tags

def Get_Tagged_Atoms_From_Mol(Mol):
    Atoms     = []
    Atom_Tags = []
    for atom in Mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            Atoms.append(atom)
            Atom_Tags.append(str(atom.GetProp('molAtomMapNumber')))
    return Atoms, Atom_Tags

def Atoms_Are_Different(Atom1, Atom2):
    if Atom1.GetAtomicNum() != Atom2.GetAtomicNum():
        return True
    if Atom1.GetTotalNumHs() != Atom2.GetTotalNumHs():
        return True
    if Atom1.GetFormalCharge() != Atom2.GetFormalCharge():
        return True
    if Atom1.GetDegree() != Atom2.GetDegree():
        return True
    if Atom1.GetNumRadicalElectrons() != Atom2.GetNumRadicalElectrons():
        return True
    if Atom1.GetIsAromatic() != Atom2.GetIsAromatic():
        return True
    
    Bonds1 = sorted([Bond_to_Label(bond) for bond in Atom1.GetBonds()])
    Bonds2 = sorted([Bond_to_Label(bond) for bond in Atom2.GetBonds()])
    if Bonds1 != Bonds2:
        return True
    
    return False

def Find_Map_Number(Mol, MapNumber):
    return [(a.GetIdx(), a) for a in Mol.GetAtoms() if a.HasProp('molAtomMapNumber') and a.GetProp('molAtomMapNumber') == str(MapNumber)][0]

def Get_Tetrahedral_Atoms(Reactants, Products):
    Tetrahedral_Atoms = []
    for reactant in Reactants:
        for ar in reactant.GetAtoms():
            if not ar.HasProp('molAtomMapNumber'):
                continue
            Atom_Tag = ar.GetProp('molAtomMapNumber')
            ir       = ar.GetIdx()
            for product in Products:
                try:
                    (ip, ap) = Find_Map_Number(product, Atom_Tag)
                    if ar.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED or ap.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
                        Tetrahedral_Atoms.append((Atom_Tag, ar, ap))
                except IndexError:
                    pass

    return Tetrahedral_Atoms

def Set_Isotope_to_equal_MapNumber(Mol):
    for a in Mol.GetAtoms():
        if a.HasProp('molAtomMapNumber'):
            a.SetIsotope(int(a.GetProp('molAtomMapNumber')))

def Get_Fragment_Around_Tetrahedral_Center(Mol, Idx):
    Idx_to_Include = [Idx]
    for neighbor in Mol.GetAtomWithIdx(Idx).GetNeighbors():
        Idx_to_Include.append(neighbor.GetIdx())
    Symbols = ['[{}{}]'.format(a.GetIsotope(), a.GetSymbol()) if a.GetIsotope() != 0 else '[#{}]'.format(a.GetAtomicNum()) for a in Mol.GetAtoms()]
    return Chem.MolFragmentToSmiles(Mol, Idx_to_Include, isomericSmiles=True, atomSymbols=Symbols, allBondsExplicit=True, allHsExplicit=True)

def Check_Tetrahedral_Centers_Equivalent(Atom1, Atom2):
    Atom1_Fragments    = Get_Fragment_Around_Tetrahedral_Center(Atom1.GetOwningMol(), Atom1.GetIdx())
    Atom1_Neighborhood = Chem.MolFromSmiles(Atom1_Fragments, sanitize=False)
    for matched_ids in Atom2.GetOwningMol().GetSubstructMatches(Atom1_Neighborhood, useChirality=True):
        if Atom2.GetIdx() in matched_ids:
            return True
    return False

def Clear_Isotope(Mol):
    [a.SetIsotope(0) for a in Mol.GetAtoms()]

def Get_Changed_Atoms(Reactants, Products):
    err = 0
    Product_Atoms, Product_Atom_Tags = Get_Tagged_Atoms_From_Mols(Products)
    if VERBOSE: 
        print('Products contain {} tagged atoms'.format(len(Product_Atoms)))
        print('Products contain {} unique atom number'.format(len(Product_Atom_Tags)))

    Reactant_Atoms, Reactant_Atom_Tags = Get_Tagged_Atoms_From_Mols(Reactants)
    if len(set(Product_Atom_Tags)) != len(set(Reactant_Atom_Tags)):
        if VERBOSE:
            print('Warning: Different atom tag appear in reactants and products')
        #err = 1
    if len(Product_Atoms) != len(Reactant_Atoms):
        if VERBOSE:
            print('Warning: Total number of tagged atoms differ, stoichometry != 1?')
        #err = 1

    Changed_Atoms     = []
    Changed_Atom_Tags = []
    for i, prod_tag in enumerate(Product_Atom_Tags):
        for j, reac_tag in enumerate(Reactant_Atom_Tags):
            if reac_tag != prod_tag:
                continue
            if reac_tag not in Changed_Atom_Tags:
                if Atoms_Are_Different(Product_Atoms[i], Reactant_Atoms[j]):
                    Changed_Atoms.append(Reactant_Atoms[j])
                    Changed_Atom_Tags.append(reac_tag)
                    break
                if Product_Atom_Tags.count(reac_tag) > 1:
                    Changed_Atoms.append(Reactant_Atoms[j])
                    Changed_Atom_Tags.append(reac_tag)
                    break

    for j, reac_tag in enumerate(Reactant_Atom_Tags):
        if reac_tag not in Changed_Atom_Tags:
            if reac_tag not in Product_Atom_Tags:
                Changed_Atoms.append(Reactant_Atoms[j])
                Changed_Atom_Tags.append(reac_tag)

    Tetra_Atoms = Get_Tetrahedral_Atoms(Reactants, Products)
    if VERBOSE:
        print('Found {} atom-mapped tetrahedral atoms that have chirality specified at least partially'.format(len(Tetra_Atoms)))
    [Set_Isotope_to_equal_MapNumber(reactant) for reactant in Reactants]
    [Set_Isotope_to_equal_MapNumber(product) for product in Products]
    for (atom_tag, ar, ap) in Tetra_Atoms:
        if VERBOSE:
            print('For atom tag {}'.format(atom_tag))
            print('   Reactant: {}'.format(ar.GetChiralTag()))
            print('   Product:  {}'.format(ap.GetChiralTag()))
        if atom_tag in Changed_Atom_Tags:
            if VERBOSE:
                print('-> atoms have changed (by more than just chirality!)')
        else:
            Unchanged = Check_Tetrahedral_Centers_Equivalent(ar, ap) and Chem.rdchem.ChiralType.CHI_UNSPECIFIED not in [ar.GetChiralTag(), ap.GetChiralTag()]
            if Unchanged:
                if VERBOSE:
                    print('-> atoms confirmed to have same chirality, no changed')
            else:
                if VERBOSE:
                    print('-> atom changed chirality!!')
                Tetra_Adjust_to_Rxn = False
                for neighbor in ap.GetNeighbors():
                    if neighbor.HasProp('molAtomMapNumber'):
                        if neighbor.GetProp('molAtomMapNumber') in Changed_Atom_Tags:
                            Tetra_Adjust_to_Rxn = True
                            break
                if Tetra_Adjust_to_Rxn:
                    if VERBOSE:
                        print('-> atom adjust to reaction center, now include')
                    Changed_Atoms.append(ar)
                    Changed_Atom_Tags.append(atom_tag)
                else:
                    if VERBOSE:
                        print('-> adj far from reaction center, not including')
    [Clear_Isotope(reactant) for reactant in Reactants]
    [Clear_Isotope(product) for product in Products]
            
    if VERBOSE:
        print('{} tagged atoms in reactants change 1-atom properties'.format(len(Changed_Atom_Tags)))
        for smarts in [atom.GetSmarts() for atom in Changed_Atoms]:
            print('  {}'.format(smarts))

    return Changed_Atoms, Changed_Atom_Tags, err

def Get_Special_Groups(Mol):
    Group_Templates = [
        (range(3), '[OH0,SH0]=C[O,Cl,I,Br,F]',), # carboxylic acid / halogen
        (range(3), '[OH0,SH0]=CN',), # amide/sulfamide
        (range(4), 'S(O)(O)[Cl]',), # sulfonyl chloride
        (range(3), 'B(O)O',), # boronic acid/ester
        ((0,), '[Si](C)(C)C'), # trialkyl silane
        ((0,), '[Si](OC)(OC)(OC)'), # trialkoxy silane, default to methyl
        (range(3), '[N;H0;$(N-[#6]);D2]-,=[N;D2]-,=[N;D1]',), # azide
        (range(8), 'O=C1N([Br,I,F,Cl])C(=O)CC1',), # NBS brominating agent
        (range(11), 'Cc1ccc(S(=O)(=O)O)cc1'), # Tosyl
        ((7,), 'CC(C)(C)OC(=O)[N]'), # N(boc)
        ((4,), '[CH3][CH0]([CH3])([CH3])O'), # 
        (range(2), '[C,N]=[C,N]',), # alkene/imine
        (range(2), '[C,N]#[C,N]',), # alkyne/nitrile
        ((2,), 'C=C-[*]',), # adj to alkene
        ((2,), 'C#C-[*]',), # adj to alkyne
        ((2,), 'O=C-[*]',), # adj to carbonyl
        ((3,), 'O=C([CH3])-[*]'), # adj to methyl ketone
        ((3,), 'O=C([O,N])-[*]',), # adj to carboxylic acid/amide/ester
        (range(4), 'ClS(Cl)=O',), # thionyl chloride
        (range(2), '[Mg,Li,Zn,Sn][Br,Cl,I,F]',), # grinard/metal (non-disassociated)
        (range(3), 'S(O)(O)',), # SO2 group
        (range(2), 'N~N',), # diazo
        ((1,), '[!#6;R]@[#6;R]',), # adjacency to heteroatom in ring
        ((2,), '[a!c]:a:a',), # two-steps away from heteroatom in aromatic ring
        #((1,), 'c(-,=[*]):c([Cl,I,Br,F])',), # ortho to halogen on ring - too specific?
        #((1,), 'c(-,=[*]):c:c([Cl,I,Br,F])',), # meta to halogen on ring - too specific?
        ((0,), '[B,C](F)(F)F'), # CF3, BF3 should have the F3 included
    ]
    Group_Templates += [
        ((1,2,), '[*]/[CH]=[CH]/[*]'), # trans with two hydrogens
        ((1,2,), '[*]/[CH]=[CH]\[*]'), # cis with two hydrogens
        ((1,2,), '[*]/[CH]=[CH0]([*])\[*]'), # trans with one hydrogens
        ((1,2,), '[*]/[D3;H1]=[!D1]'), # specified on one end, can be N or C
    ]

    Groups = []
    for (add_if_match, template) in Group_Templates:
        Matches = Mol.GetSubstructMatches(Chem.MolFromSmarts(template), useChirality=True)
        for match in Matches:
            add_if = []
            for pattern_idx, atom_idx in enumerate(match):
                if pattern_idx in add_if_match:
                    add_if.append(atom_idx)
            Groups.append((add_if, match))
    return Groups

def Expand_Atoms_to_Use(Mol, Atoms_to_Use, Groups=[], Symbol_Replacements=[]):

    New_Atoms_to_Use = Atoms_to_Use[:]
    for atom in Mol.GetAtoms():
        if atom.GetIdx() not in Atoms_to_Use:
            continue
        for group in Groups:
            if int(atom.GetIdx()) in group[0]:
                if VERBOSE:
                    print('Adding group due to match')
                    try:
                        print('Match from molAtomMapNum {}'.format(atom.GetProp('molAtomMapNumber')))
                    except KeyError:
                        pass
                for idx in group[1]:
                    if idx not in Atoms_to_Use:
                        New_Atoms_to_Use.append(idx)
                        Symbol_Replacements.append((idx, Convert_Atom_to_Wildcard(Mol.GetAtomWithIdx(idx))))

        for neighbor in atom.GetNeighbors():
            New_Atoms_to_Use, Symbol_Replacements = Expand_Atoms_to_Use_Atom(Mol, New_Atoms_to_Use, neighbor.GetIdx(), Groups=Groups, Symbol_Replacements=Symbol_Replacements)
    return New_Atoms_to_Use, Symbol_Replacements

def Expand_Atoms_to_Use_Atom(Mol, Atom_to_Use, Atom_Idx, Groups=[], Symbol_Replacements=[]):
    Found_in_Group = False
    for group in Groups:
        if int(Atom_Idx) in group[0]:
            if VERBOSE:
                print('Adding group due to match')
                try:
                    print('Match from molAtomMapNum {}'.format(Mol.GetAtomWithIdx(Atom_Idx).GetProp('molAtomMapNumber')))
                except KeyError:
                    pass
            for idx in group[1]:
                if idx not in Atom_to_Use:
                    Atom_to_Use.append(idx)
                    Symbol_Replacements.append((idx, Convert_Atom_to_Wildcard(Mol.GetAtomWithIdx(idx))))
            Found_in_Group = True

    if Found_in_Group:
        return Atom_to_Use, Symbol_Replacements

    if Atom_Idx in Atom_to_Use:
        return Atom_to_Use, Symbol_Replacements

    Atom_to_Use.append(Atom_Idx)
    Symbol_Replacements.append((Atom_Idx, Convert_Atom_to_Wildcard(Mol.GetAtomWithIdx(Atom_Idx))))

    return Atom_to_Use, Symbol_Replacements

def Convert_Atom_to_Wildcard(Atom):

    if Atom.GetDegree() == 1:
        Symbol = '[' + Atom.GetSymbol() + ';D1;H{}'.format(Atom.GetTotalNumHs())
        if Atom.GetFormalCharge() != 0:
            Charges = re.search('([-+]+[1-9]?)', Atom.GetSmarts())
            Symbol  = Symbol.replace(';D1', ';{};D1'.format(Charges.group()))

    else:
        Symbol = '['
        if Atom.GetAtomicNum() != 6:
            Symbol += '#{};'.format(Atom.GetAtomicNum())
            if Atom.GetIsAromatic():
                Symbol += 'a;'
        elif Atom.GetIsAromatic():
            Symbol += 'c;'
        else:
            Symbol += 'C;'

        if Atom.GetFormalCharge() != 0:
            Charges = re.search('([-+]+[1-9]?)', Atom.GetSmarts())
            if Charges:
                Symbol += Charges.group() + ';'

        if Symbol[-1] == ';':
            Symbol = Symbol[:-1]

    Label = re.search('\:[0-9]+\]', Atom.GetSmarts())
    if Label:
        Symbol += Label.group()
    else:
        Symbol += ']'

    if VERBOSE:
        if Symbol != Atom.GetSmarts():
            print('Improved generality of atom SMARTS {} -> {}'.format(Atom.GetSmarts(), Symbol))

    return Symbol

def Reassign_Atom_Mapping(Transform):
    
    All_Label        = re.findall('\:([0-9]+)\]', Transform)
    Replacements     = []
    Replacement_Dict = {}
    Counter          = 1
    for label in All_Label:
        if label not in Replacement_Dict:
            Replacement_Dict[label] = str(Counter)
            Counter += 1
        Replacements.append(Replacement_Dict[label])

    Transform_Newmaps = re.sub('\:[0-9]+\]', lambda match: (':' + Replacements.pop(0) + ']'), Transform)
    return Transform_Newmaps

def Get_Strict_SMARTS_for_Atom(Atom):
    
    Symbol = Atom.GetSmarts()
    if Atom.GetSymbol() == 'H':
        Symbol = '[#1]'

    if '[' not in Symbol:
        Symbol = '[' + Symbol + ']'

    if USE_STEREOCHEMISTRY:
        if Atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            if '@' not in Symbol:
                if Atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
                    tag = '@'
                elif Atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
                    tag = '@@'
                if ':' in Symbol:
                    Symbol = Symbol.replace(':', ';{}:'.format(tag))
                else:
                    Symbol = Symbol.replace(']', ';{}]'.format(tag))

    if 'H' not in Symbol:
        H_Symbol = 'H{}'.format(Atom.GetTotalNumHs())
        if ':' in Symbol:
            Symbol = Symbol.replace(':', ';{}:'.format(H_Symbol))
        else:
            Symbol = Symbol.replace(']', ';{}]'.format(H_Symbol))

    if ':' in Symbol:
        Symbol = Symbol.replace(':', ';D{}:'.format(Atom.GetDegree()))
    else:
        Symbol = Symbol.replace(']', ';D{}]'.format(Atom.GetDegree()))

    if '+' not in Symbol and '-' not in Symbol:
        Charge        = Atom.GetFormalCharge()
        Charge_Symbol = '+' if (Charge >= 0) else '-'
        Charge_Symbol += '{}'.format(abs(Charge))
        if ':' in Symbol:
            Symbol = Symbol.replace(':', ';{}:'.format(Charge_Symbol))
        else:
            Symbol = Symbol.replace(']', ';{}]'.format(Charge_Symbol))

    return Symbol

def Expand_Changed_Atom_Tags(Changed_Atom_Tags, Reactant_Fragments):

    Expansion = []
    Atom_Tags_in_Reactant_Fragments = re.findall('\:([0-9]+)\]', Reactant_Fragments)
    for atom_tag in Atom_Tags_in_Reactant_Fragments:
        if atom_tag not in Changed_Atom_Tags:
            Expansion.append(atom_tag)
    if VERBOSE:
        print('After building reactant fragments, additional labels included: {}'.format(Expansion))
    return Expansion

def Get_Fragments_for_Changed_Atoms(Mols, Changed_Atom_Tags, Radius=0, Category='Reactants', Expansion=[]):

    Fragments    = ''
    Mols_Changed = []

    for mol in Mols:

        Symbol_Replacements = []
        if Category == 'Reactants':
            Groups = Get_Special_Groups(mol)
        else:
            Groups = []

        Atoms_to_Use = []
        for atom in mol.GetAtoms():
            if ':' in atom.GetSmarts():
                if atom.GetSmarts().split(':')[1][:-1] in Changed_Atom_Tags:
                    Atoms_to_Use.append(atom.GetIdx())
                    Symbol = Get_Strict_SMARTS_for_Atom(atom)
                    if Symbol != atom.GetSmarts():
                        Symbol_Replacements.append((atom.GetIdx(), Symbol))
                    continue
        
        if INCLUDE_ALL_UNMAPPED_REACTANT_ATOMS and len(Atoms_to_Use) > 0:
            if Category == 'Reactants':
                for atom in mol.GetAtoms():
                    if not atom.HasProp('molAtomMapNumber'):
                        Atoms_to_Use.append(atom.GetIdx())

        for k in range(Radius):
            Atoms_to_Use, Symbol_Replacements = Expand_Atoms_to_Use(mol, Atoms_to_Use, Groups=Groups, Symbol_Replacements=Symbol_Replacements)

        if Category == 'Products':
            if Expansion:
                for atom in mol.GetAtoms():
                    if ':' not in atom.GetSmarts():
                        continue
                    Label = atom.GetSmarts().split(':')[1][:-1]
                    if Label in Expansion and Label not in Changed_Atom_Tags:
                        Atoms_to_Use.append(atom.GetIdx())
                        Symbol_Replacements.append((atom.GetIdx(), Convert_Atom_to_Wildcard(atom)))
                        if VERBOSE:
                            print('Expanded label {} to wildcard in products'.format(Label))
            
            for atom in mol.GetAtoms():
                if not atom.HasProp('molAtomMapNumber'):
                    Atoms_to_Use.append(atom.GetIdx())
                    Symbol = Get_Strict_SMARTS_for_Atom(atom)
                    Symbol_Replacements.append((atom.GetIdx(), Symbol))

        Symbols = [atom.GetSmarts() for atom in mol.GetAtoms()]
        for (i, symbol) in Symbol_Replacements:
            Symbols[i] = symbol

        if not Atoms_to_Use:
            continue

        #Tetra_Consistent = False
        #Num_Tetra_Flips  = 0
        #while not Tetra_Consistent and Num_Tetra_Flips < 100:
        #    Mol_Copy = deepcopy(mol)
        #    [x.ClearProp('molAtomMapNumber') for x in Mol_Copy.GetAtoms()]
        #    This_Fragment = AllChem.MolFragmentToSmiles(Mol_Copy, Atoms_to_Use, atomSymbols=Symbols, allHsExplicit=True, isomericSmiles=USE_STEREOCHEMISTRY, allBondsExplicit=True)
        #    This_Fragment_Mol = AllChem.MolFromSmarts(This_Fragment)
        #    Tetra_Map_Nums = []
        #    for atom in This_Fragment_Mol.GetAtoms():
        #        if atom.HasProp('molAtomMapNumber'):
        #            atom.SetIsotope(int(atom.GetProp('molAtomMapNumber')))
        #            if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
        #                Tetra_Map_Nums.append(atom.GetProp('molAtomMapNumber'))
        #    Map_to_Id = {}
        #    for atom in mol.GetAtoms():
        #        if atom.HasProp('molAtomMapNumber'):
        #            atom.SetIsotope(int(atom.GetProp('molAtomMapNumber')))
        #            Map_to_Id[atom.GetProp('molAtomMapNumber')] = atom.GetIdx()

        #    Tetra_Consistent = True
        #    All_Matched_Ids  = []
        #    Fragment_SMILES  = Chem.MolToSmiles(This_Fragment_Mol)
        #    if Fragment_SMILES.count('.') > 5:
        #        break

        #    for matched_ids in mol.GetSubstructMatches(This_Fragment_Mol, useChirality=True):
        #        All_Matched_Ids.extend(matched_ids)

        #    shuffle(Tetra_Map_Nums)
        #    for tetra_map_num in Tetra_Map_Nums:
        #        if VERBOSE:
        #            print('Checking consistency of tetrahedral {}'.format(tetra_map_num))
        #        if Map_to_Id[tetra_map_num] not in All_Matched_Ids:
        #            Tetra_Consistent = False
        #            if VERBOSE:
        #                print('@@@@@@@@@@ FRAGMENT DOES NOT MATCH PARENT MOL @@@@@@@@@@@')
        #                print('@@@@@@@@@@ FILPPING CHIRALITY SYMBOL NOW      @@@@@@@@@@@')
        #            previous_symbol = Symbols[Map_to_Id[tetra_map_num]]
        #            if '@@' in previous_symbol:
        #                Symbol = previous_symbol.replace('@@', '@')
        #            elif '@' in previous_symbol:
        #                Symbol = previous_symbol.replace('@', '@@')
        #            else:
        #                raise ValueError('Need to modify symbol of tetra atom without @ or @@???')

        #            Symbols[Map_to_Id[tetra_map_num]] = Symbol
        #            Num_Tetra_Flips += 1 
        #            break

        #    for atom in mol.GetAtoms():
        #        atom.SetIsotope(0)

        #if not Tetra_Consistent:
        #    raise ValueError('Could not find consistent tetrahedral mapping, {} centers'.format(len(Tetra_Map_Nums)))
        [x.ClearProp('molAtomMapNumber') for x in mol.GetAtoms()]
        This_Fragment = AllChem.MolFragmentToSmiles(mol, Atoms_to_Use, atomSymbols=Symbols, allHsExplicit=True, isomericSmiles=USE_STEREOCHEMISTRY, allBondsExplicit=True)
        Fragments += '(' + This_Fragment + ').'
        Mols_Changed.append(Chem.MolToSmiles(Clear_MapNumber(Chem.MolFromSmiles(Chem.MolToSmiles(mol, True))), True))

    Intra_Only = (1 == len(Mols_Changed))
    Dimer_Only = (1 == len(set(Mols_Changed))) and (len(Mols_Changed) == 2)

    return Fragments[:-1], Intra_Only, Dimer_Only

def Canonical_Transform(Transform):
    Transform_Reordered = '>>'.join([Canonical_Template(x) for x in Transform.split('>>')])
    return Reassign_Atom_Mapping(Transform_Reordered)

def Canonical_Template(Template):
    Template_Nolabels      = re.sub('\:[0-9]+\]', ']', Template)
    Template_Nolabels_Mols = Template_Nolabels[1:-1].split(').(')
    Template_Mols          = Template[1:-1].split(').(')
    for i in range(len(Template_Mols)):
        Nolabel_Mol_Fragments = Template_Nolabels_Mols[i].split('.')
        Mol_Fragments         = Template_Mols[i].split('.')
        Sortorder             = [j[0] for j in sorted(enumerate(Nolabel_Mol_Fragments), key=lambda x:x[1])]
        Template_Nolabels_Mols[i] = '.'.join([Nolabel_Mol_Fragments[j] for j in Sortorder])
        Template_Mols[i]          = '.'.join([Mol_Fragments[j] for j in Sortorder])

    Sortorder = [j[0] for j in sorted(enumerate(Template_Nolabels_Mols), key=lambda x:x[1])]
    Template  = '(' + ').('.join([Template_Mols[i] for i in Sortorder]) + ')'
    return Template

def Bond_to_Label(Bond):
    Atom1_Label = str(Bond.GetBeginAtom().GetAtomicNum())
    Atom2_Label = str(Bond.GetEndAtom().GetAtomicNum())

    if Bond.GetBeginAtom().HasProp('molAtomMapNumber'):
        Atom1_Label += Bond.GetBeginAtom().GetProp('molAtomMapNumber')
    if Bond.GetEndAtom().HasProp('molAtomMapNumber'):
        Atom2_Label += Bond.GetEndAtom().GetProp('molAtomMapNumber')
    Atoms = sorted([Atom1_Label, Atom2_Label])
    return '{}{}{}'.format(Atoms[0], Bond.GetSmarts(), Atoms[1])

def Get_Reacting_Mols(Mols, Changed_Atom_Tags):
    Mols_SMILES = []
    for mol in Mols:
        for atom in mol.GetAtoms():
            if ':' in atom.GetSmarts():
                if atom.GetSmarts().split(':')[1][:-1] in Changed_Atom_Tags:
                    SMILES = Chem.MolToSmiles(mol)
                    if SMILES not in Mols_SMILES:
                        Mols_SMILES.append(SMILES)
                    continue 
    return Mols_SMILES

def Extract_from_Reaction(Reaction):

    Reactants = Mols_From_SMILES_List(Replace_Deuterated(Reaction['Reactants']).split('.'))
    Products  = Mols_From_SMILES_List(Replace_Deuterated(Reaction['Products']).split('.'))

    if None in Reactants:
        return {'Reaction_ID':Reaction['Id']}
    if None in Products:
        return {'Reaction_ID':Reaction['Id']}

    try:
        for i in range(len(Reactants)):
            Reactants[i] = AllChem.RemoveHs(Reactants[i])
        for i in range(len(Products)):
            Products[i] = AllChem.RemoveHs(Products[i])
        [Chem.SanitizeMol(mol) for mol in Reactants + Products]
        [mol.UpdatePropertyCache() for mol in Reactants + Products]
    except Exception as e:
        if VERBOSE:
            print(e)
            print('Cound not load SMILES or Sanitize')
            print('ID: {}'.format(Reaction['Id']))
        return {'Reaction_ID':Reaction['Id']}

    Are_Unmapped_Product_Atoms = False
    Extra_Reactant_Fragment    = ''
    for product in Products:
        product_atoms = product.GetAtoms()
        if sum([a.HasProp('molAtomMapNumber') for a in product_atoms]) < len(product_atoms):
            if VERBOSE:
                print('Not all product atoms have atom mapping')
                print('ID: {}'.format(Reaction['Id']))
            Are_Unmapped_Product_Atoms = True

    if Are_Unmapped_Product_Atoms:
        for product in Products:
            product_atoms = product.GetAtoms()
            Unmapped_Ids  = [a.GetIdx() for a in product_atoms if not a.HasProp('molAtomMapNumber')]
            if len(Unmapped_Ids) > MAXIMUM_NUMBER_UNMAPPED_PRODUCT_ATOMS:
                if VERBOSE:
                    print('Skip this example - too many unmapped product atoms!')
                    print('ID: {}'.format(Reaction['Id']))
                return {'Reaction_ID':Reaction['Id']}
            Atom_Symbols = ['[{}]'.format(a.GetSymbol()) for a in product_atoms]
            Bond_Symbols = ['~' for b in product.GetBond()]
            if Unmapped_Ids:
                Extra_Reactant_Fragment += AllChem.MolFragmentToSmiles(product, Unmapped_Ids, allHsExplicit=True, isomericSmiles=USE_STEREOCHEMISTRY, atomSymbols=Atom_Symbols, bondSymbols=Bond_Symbols) + '.'

        if Extra_Reactant_Fragment:
            Extra_Reactant_Fragment = Extra_Reactant_Fragment[:-1]
            if VERBOSE:
                print('Extra reactant fragment: {}'.format(Extra_Reactant_Fragment))

        Extra_Reactant_Fragment = '.'.join(sorted(list(set(Extra_Reactant_Fragment.split('.')))))

    if None in Reactants + Products:
        if VERBOSE:
            print('Could not parse all molecules in reaction, skipping')
            print('ID: {}'.format(Reaction['Id']))
        return {'Reaction_ID':Reaction['Id']}

    Changed_Atoms, Changed_Atom_Tags, err = Get_Changed_Atoms(Reactants, Products)
    if err:
        if VERBOSE:
            print('Could not get changed atoms')
            print('ID: {}'.format(Reaction['Id']))
        return {'Reaction_ID':Reaction['Id']}
    if not Changed_Atom_Tags:
        if VERBOSE:
            print('No atom changed?')
            print('ID: {}'.format(Reaction['Id']))
        return {'Reaction_ID':Reaction['Id']}

    Mol_Reactants = Get_Reacting_Mols(Reactants, Changed_Atom_Tags)
    Mol_Products  = Get_Reacting_Mols(Products, Changed_Atom_Tags) 
    try:
        Reactant_Fragments, Intra_only, Dimer_Only = Get_Fragments_for_Changed_Atoms(Reactants, Changed_Atom_Tags, Radius=1, Expansion=[], Category='Reactants')
        Product_Fragments, _, _ = Get_Fragments_for_Changed_Atoms(Products, Changed_Atom_Tags, Radius=0, Expansion=Expand_Changed_Atom_Tags(Changed_Atom_Tags, Reactant_Fragments), Category='Products')
    except ValueError as e:
        if VERBOSE:
            print(e)
            print('ID: {}'.format(Reaction['Id']))
        return {'Reaction_ID':Reaction['Id']}
    except RuntimeError as e:
        if VERBOSE:
            print(e)
            print('ID: {}'.format(Reaction['Id']))
        return {'Reaction_ID':Reaction['Id']}

    Rxn_String          = '{}>>{}'.format(Reactant_Fragments, Product_Fragments)
    Rxn_Canonical       = Canonical_Transform(Rxn_String)
    Rxn_Canonical_Split = Rxn_Canonical.split('>>')
    Rxn_Canonical       = Rxn_Canonical_Split[0][1:-1].replace(').(', '.') + '>>' + Rxn_Canonical_Split[1][1:-1].replace(').(', '.')

    Reactants_String    = Rxn_Canonical.split('>>')[0]
    Products_String     = Rxn_Canonical.split('>>')[1]

    Retro_Canonical     = Products_String + '>>' + Reactants_String
    try:
        rxn                 = AllChem.ReactionFromSmarts(Retro_Canonical)
        if rxn.Validate()[1] != 0:
            if VERBOSE:
                print('Could not validate reaction successfully')
                print('ID: {}'.format(Reaction['Id']))
                print('Retro_Canonical: {}'.format(Retro_Canonical))
            return {'Reaction_ID':Reaction['Id']}
    except Exception as e:
        if VERBOSE:
            print(e)
            print('ID: {}'.format(Reaction['Id']))
        return {'Reaction_ID':Reaction['Id']}

    Template = {
        'Products': Mol_Products,
        'Reactants': Mol_Reactants,
        'Reaction_SMARTS':Retro_Canonical,
        'Intra_Only':Intra_only,
        'Dimer_Only':Dimer_Only,
        'Reaction_ID':Reaction['Id'],
        'Necessary_Reagent':Extra_Reactant_Fragment,
        'Spectators':Reaction['Spectators']
    }
    return Template