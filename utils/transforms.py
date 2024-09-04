import torch
import torch.nn.functional as F
import numpy as np
import pickle

from datasets.pl_data import ProteinLigandData
from utils import data as utils_data

AROMATIC_FEAT_MAP_IDX = utils_data.ATOM_FAMILIES_ID['Aromatic']

# only atomic number 1, 6, 7, 8, 9, 15, 16, 17 exist
MAP_ATOM_TYPE_FULL_TO_INDEX = {
    (1, 'S', False): 0,
    (6, 'SP', False): 1,
    (6, 'SP2', False): 2,
    (6, 'SP2', True): 3,
    (6, 'SP3', False): 4,
    (7, 'SP', False): 5,
    (7, 'SP2', False): 6,
    (7, 'SP2', True): 7,
    (7, 'SP3', False): 8,
    (8, 'SP2', False): 9,
    (8, 'SP2', True): 10,
    (8, 'SP3', False): 11,
    (9, 'SP3', False): 12,
    (15, 'SP2', False): 13,
    (15, 'SP2', True): 14,
    (15, 'SP3', False): 15,
    (15, 'SP3D', False): 16,
    (16, 'SP2', False): 17,
    (16, 'SP2', True): 18,
    (16, 'SP3', False): 19,
    (16, 'SP3D', False): 20,
    (16, 'SP3D2', False): 21,
    (17, 'SP3', False): 22
}

MAP_ATOM_TYPE_ONLY_TO_INDEX = {
    1: 0,
    6: 1,
    7: 2,
    8: 3,
    9: 4,
    15: 5,
    16: 6,
    17: 7,
    35: 8,
}

MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
    (1, False): 0,
    (6, False): 1,
    (6, True): 2,
    (7, False): 3,
    (7, True): 4,
    (8, False): 5,
    (8, True): 6,
    (9, False): 7,
    (15, False): 8,
    (15, True): 9,
    (16, False): 10,
    (16, True): 11,
    (17, False): 12
}

MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX = {
    (1, False): 0,  # 998
    (6, False): 1,  # 803564
    (6, True): 2,   # 1371671
    (7, False): 3,  # 198073
    (7, True): 4,   # 209316
    (8, False): 5,  # 264143
    (8, True): 6,   # 8567
    (9, False): 7,  # 72259 
    (15, False): 8, # 2051
    (16, False): 9, # 21263
    (16, True): 10, # 11529
    (17, False): 11, # 26781
    (35, False): 12, # 3359
}

MAP_ATOM_TYPE_AROMATIC_PDB_ALL_TO_INDEX = {
    (1, False): 0,  # 998
    (6, False): 1,  # 803564
    (6, True): 2,   # 1371671
    (7, False): 3,  # 198073
    (7, True): 4,   # 209316
    (8, False): 5,  # 264143
    (8, True): 6,   # 8567
    (9, False): 7,  # 72259
    (14, False): 8, # 92 -------
    (15, False): 9, # 2051
    (16, False): 10, # 21263
    (16, True): 11, # 11529
    (17, False): 12, # 26781
    (35, False): 13, # 3359
    (53, False): 14, # 852 -----
}

MAP_INDEX_TO_ATOM_TYPE_ONLY = {v: k for k, v in MAP_ATOM_TYPE_ONLY_TO_INDEX.items()}
MAP_INDEX_TO_ATOM_TYPE_AROMATIC = {v: k for k, v in MAP_ATOM_TYPE_AROMATIC_TO_INDEX.items()}
MAP_INDEX_TO_ATOM_TYPE_FULL = {v: k for k, v in MAP_ATOM_TYPE_FULL_TO_INDEX.items()}
MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB = {v: k for k, v in MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX.items()}

def get_atomic_number_from_index(index, mode):
    if mode == 'basic':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_ONLY[i] for i in index.tolist()]
    elif mode == 'add_aromatic':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][0] for i in index.tolist()]
    elif mode == 'add_aromatic_pdb':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB[i][0] for i in index.tolist()]
    elif mode == 'full':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_FULL[i][0] for i in index.tolist()]
    else:
        raise ValueError
    return atomic_number


def is_aromatic_from_index(index, mode):
    if mode == 'add_aromatic':
        is_aromatic = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][1] for i in index.tolist()]
    elif mode == 'add_aromatic_pdb':
        is_aromatic = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB[i][1] for i in index.tolist()]
    elif mode == 'full':
        is_aromatic = [MAP_INDEX_TO_ATOM_TYPE_FULL[i][2] for i in index.tolist()]
    elif mode == 'basic':
        is_aromatic = None
    else:
        raise ValueError
    return is_aromatic


def get_hybridization_from_index(index, mode):
    if mode == 'full':
        hybridization = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][1] for i in index.tolist()]
    else:
        raise ValueError
    return hybridization


def get_index(atom_num, hybridization, is_aromatic, mode):
    if mode == 'basic':
        return MAP_ATOM_TYPE_ONLY_TO_INDEX[int(atom_num)]
    elif mode == 'add_aromatic':
        # self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17])  # H, C, N, O, F, P, S, Cl
        if (int(atom_num), bool(is_aromatic)) in MAP_ATOM_TYPE_AROMATIC_TO_INDEX:
            return MAP_ATOM_TYPE_AROMATIC_TO_INDEX[int(atom_num), bool(is_aromatic)]
        else:
            # print(int(atom_num), bool(is_aromatic)) # Vineeth comment extra atoms lime Br, I are present in pdbbind dataset and not in corossdock
            return MAP_ATOM_TYPE_AROMATIC_TO_INDEX[(1, False)]
    elif mode == 'add_aromatic_pdb':
        if (int(atom_num), bool(is_aromatic)) in MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX:
            return MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX[int(atom_num), bool(is_aromatic)]
        else:
            print(int(atom_num), bool(is_aromatic), 'to', 35, False)
            return MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX[(35, False)]
    elif mode == 'full':
        return MAP_ATOM_TYPE_FULL_TO_INDEX[(int(atom_num), str(hybridization), bool(is_aromatic))]
    else:
        raise NotImplementedError

class FeaturizeProteinAtom(object):

    def __init__(self):
        super().__init__()
        self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])  # H, C, N, O, S, Se
        self.max_num_aa = 20

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1

    def __call__(self, data: ProteinLigandData):
        element = data.protein_element.view(-1, 1) == self.atomic_numbers.view(1, -1)  # (N_atoms, N_elements)
        amino_acid = F.one_hot(data.protein_atom_to_aa_type, num_classes=self.max_num_aa)
        is_backbone = data.protein_is_backbone.view(-1, 1).long()
        x = torch.cat([element, amino_acid, is_backbone], dim=-1)
        data.protein_atom_feature = x
        return data


class FeaturizeLigandAtom(object):

    def __init__(self, mode='basic'):
        super().__init__()
        assert mode in ['basic', 'add_aromatic', 'full']
        self.mode = mode

    @property
    def feature_dim(self):
        if self.mode == 'basic':
            return len(MAP_ATOM_TYPE_ONLY_TO_INDEX)
        elif self.mode == 'add_aromatic':
            return len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX)
        elif self.mode == 'add_aromatic_pdb':
            return len(MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX)
        elif self.mode == 'full':
            return len(MAP_ATOM_TYPE_FULL_TO_INDEX)
        else:
            raise NotImplementedError

    def __call__(self, data: ProteinLigandData):
        element_list = data.ligand_element
        hybridization_list = data.ligand_hybridization
        aromatic_list = [v[AROMATIC_FEAT_MAP_IDX] for v in data.ligand_atom_feature]

        x = [get_index(e, h, a, self.mode) for e, h, a in zip(element_list, hybridization_list, aromatic_list)]
        x = torch.tensor(x)
        data.ligand_atom_feature_full = x
        return data


class FeaturizeLigandBond(object):

    def __init__(self):
        super().__init__()

    def __call__(self, data: ProteinLigandData):
        data.ligand_bond_feature = F.one_hot(data.ligand_bond_type - 1, num_classes=len(utils_data.BOND_TYPES))
        return data


class RandomRotation(object):

    def __init__(self):
        super().__init__()

    def __call__(self,  data: ProteinLigandData):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        Q = torch.from_numpy(Q.astype(np.float32))
        data.ligand_pos = data.ligand_pos @ Q
        data.protein_pos = data.protein_pos @ Q
        return data

class AddScaffoldMask(object):

    def __init__(self, mask_path, change_scaffold):
        super().__init__()
        self.mask_path = mask_path
        with open(mask_path, 'rb') as f:
            self.scaffold_mask = pickle.load(f)
        self.name2mask = {
            item[0]: item[1] for item in self.scaffold_mask
        }
        self.change_scaffold = change_scaffold

    def __call__(self, data: ProteinLigandData):
        ligand_filename = data.ligand_filename
        scaffold_mask = self.name2mask.get(ligand_filename, None)
        if scaffold_mask is None:
            print(f'No scaffold mask found for {ligand_filename}, using zeros shaped {data.ligand_element.shape}')
            scaffold_mask = torch.zeros_like(data.ligand_element, dtype=torch.bool)
        if self.change_scaffold:
            scaffold_mask = ~scaffold_mask
        data.ligand_mask = scaffold_mask
        return data