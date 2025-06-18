import pickle

import numpy as np
import os
import pathlib

from Bio.Data import IUPACData
from tqdm import tqdm
from Bio.PDB import PDBParser, MMCIFParser

def load_structure(path):
    """Read a PDB or mmCIF file and return a Bio.PDB.Structure."""
    path = pathlib.Path(path)
    parser = MMCIFParser(QUIET=True) if path.suffix == ".cif" else PDBParser(QUIET=True)
    return parser.get_structure(path.stem, path)

def get_chain_atoms(structure, chain_id):
    return [atom for residue in structure[0][chain_id] for atom in residue.get_atoms() if atom.element != 'H']

def get_chain_residues(structure, chain_id):
    return list(structure[0][chain_id])

def compute_residue_min_dist(residue, target_atoms):
    return min(np.linalg.norm(atom.coord - target_atom.coord)
               for atom in residue.get_atoms() if atom.element != 'H'
               for target_atom in target_atoms)

def get_binding_mask(peptide_residues, crm1_atoms, top_k=5):
    dists = [compute_residue_min_dist(res, crm1_atoms) for res in peptide_residues]
    top_k_indices = np.argsort(dists)[:top_k]
    mask = np.zeros(len(peptide_residues), dtype=int)
    mask[top_k_indices] = 1
    return mask

def extract_peptides_and_masks(pdb_dir, crm1_chain_id="A", peptide_chain_id="B", top_k=5):
    peptide_seqs = []
    binding_masks = []

    pdb_files = [os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir)
                 if f.endswith(".pdb") or f.endswith(".cif")]

    for pdb_file in tqdm(pdb_files, desc="Processing PDBs"):
        try:
            struct = load_structure(pdb_file)
            peptide_residues = get_chain_residues(struct, peptide_chain_id)
            crm1_atoms = get_chain_atoms(struct, crm1_chain_id)

            # sequence
            three_to_one = IUPACData.protein_letters_3to1_extended
            seq = ''.join(three_to_one[res.get_resname().title()] for res in peptide_residues if res.get_resname().title() in three_to_one)
            mask = get_binding_mask(peptide_residues, crm1_atoms, top_k=top_k)

            assert mask.sum() == top_k, f"{pdb_file}: Found {mask.sum()} pocket residues instead of {top_k}"
            peptide_seqs.append(seq)
            binding_masks.append(mask)
        except Exception as e:
            print(f"⚠️ Skipping {pdb_file}: {e}")

    return peptide_seqs, binding_masks

if __name__ == '__main__':
    pdb_folder = "structures/af_positives"
    crm1_chain_id = "A"  # Update if needed
    peptide_chain_id = "B"

    print("🔍 Extracting peptide sequences and binding pocket masks")
    seqs, masks = extract_peptides_and_masks(pdb_folder, crm1_chain_id=crm1_chain_id, peptide_chain_id=peptide_chain_id)

    for seq, mask in zip(seqs, masks):
        print(seq)
        print(mask)
        print("-")

    save_path = "peptide_data.pkl"
    with open(save_path, "wb") as f:
        pickle.dump((seqs, masks), f)

    print(f"✅ Saved peptide sequences and masks to {save_path}")
