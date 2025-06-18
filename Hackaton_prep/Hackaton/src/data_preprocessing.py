import torch
# import esm
import requests
from bs4 import BeautifulSoup
import re
import json
import csv
import os


from typing import Tuple ,List

"""
Pipeline overview:
This pipeline processes NESDB peptide data and corresponding PDB structures
 to generate a training dataset for peptide binding prediction.
  It parses raw peptide info and pocket chains, extracts 3D coordinates to label residues based on spatial proximity to binding pockets,
and generates per-residue embeddings using the ESM-2 protein language model. The output is a CSV file with residue embeddings and binding labels
ready for downstream machine learning model training.
"""

# All of ESM-2 pre-trained models by embedding size
#TODO take the 1280: esm.pretrained.esm2_t33_650M_UR50D,
#
# ESM_MODELS_DICT = {320: esm.pretrained.esm2_t6_8M_UR50D,
#                    480: esm.pretrained.esm2_t12_35M_UR50D,
#                    640: esm.pretrained.esm2_t30_150M_UR50D,
#                    1280: esm.pretrained.esm2_t33_650M_UR50D,
#                    2560: esm.pretrained.esm2_t36_3B_UR50D,
#                    5120: esm.pretrained.esm2_t48_15B_UR50D}
#
#


def fetch_nes_list(url="http://prodata.swmed.edu/nes_pattern_location/"):
    resp = requests.get(url)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find the table containing NES data
    table = soup.find("table")
    if not table:
        raise RuntimeError("No table found on the page")

    nes_entries = []

    # Extract header columns and find indexes of interest
    headers = [th.get_text(strip=True) for th in table.find_all("th")]

    # Expected headers, find their positions:
    try:
        idx_name = headers.index("name")
        idx_start = headers.index("start#")
        idx_sequence = headers.index("sequence")
    except ValueError as e:
        raise RuntimeError(f"Expected column not found: {e}")

    for tr in table.find_all("tr")[1:]:  # skip header row
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cols) < max(idx_name, idx_start, idx_sequence) + 1:
            continue  # incomplete row, skip

        protein = cols[idx_name]
        start_str = cols[idx_start]
        sequence = cols[idx_sequence]

        if not start_str.isdigit():
            continue  # invalid start number

        start = int(start_str)
        end = start + len(sequence) - 1

        nes_entries.append({
            "protein": protein,
            "start": start,
            "end": end,
            "sequence": sequence
        })
    script_dir = os.path.dirname(os.path.abspath(__file__))  # get the folder where the script is
    file_path = os.path.join(script_dir, "nes_list.csv")
    with open(file_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["protein", "start", "end", "sequence"])
        writer.writeheader()
        writer.writerows(nes_entries)

    return nes_entries

def get_esm_model(embedding_size=1280):
    """
    Retrieves a pre-trained ESM-2 model
    :param embedding_size: The ESM-2 model embedding size
    :return: esm_model, alphabet, batch_converter, device
    """

    if embedding_size not in ESM_MODELS_DICT:
        raise ValueError(f"ERROR: ESM does not have a trained model with embedding size of {embedding_size}.\n "
                         f"Please use one of the following embedding sized: {ESM_MODELS_DICT.keys()}")
    model, alphabet = ESM_MODELS_DICT[embedding_size]()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    # check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"ESM model loaded to {device}")
    return model, alphabet, batch_converter, device


def get_esm_embeddings(pep_tuple_list, esm_model, alphabet, batch_converter, device, embedding_layer=33, sequence_embedding=True):
    """
    This function convert peptide sequence data into ESM sequence embeddings
    :param pep_tuple_list: peptide tuple list of format : [(name_1, seq_1), (name_2, seq_2), ...]
    :param esm_model: Pre-trained ESM-2 model
    :param alphabet: ESM-2 alphabet object
    :param batch_converter: ESM-2 batch_converter object
    :param device: GPU/CPU device
    :param embedding_layer: The desired embedding layer to get
    :param sequence_embedding: Whether to use a sequence embedding (default=True) or amino acid embedding
    :return: List of ESM-2 sequence/amino acids embeddings
    """
    batch_labels, batch_strs, batch_tokens = batch_converter(pep_tuple_list)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations
    with torch.no_grad():
        results = esm_model(batch_tokens.to(device), repr_layers=[embedding_layer])
    token_representations = results["representations"][embedding_layer]

    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    representations = []
    for i, tokens_len in enumerate(batch_lens):
        embedding = token_representations[i, 1: tokens_len - 1]
        # Generate per-sequence representations via averaging
        if sequence_embedding:
            embedding = embedding.mean(0)
        representations.append(embedding.cpu().numpy())

    return representations


def extract_peptide_coords(pdb_path: str, peptide_chain: str = 'P') -> Tuple[List[int], np.ndarray]:
    """
    Extracts residue indices and CA atom coordinates from peptide chain in PDB/AlphaFold structure.
    Args:
        pdb_path: path to pdb file
        peptide_chain: chain ID of peptide in structure (default 'P')
    Returns:
        res_ids: list of residue sequence numbers
        coords: numpy array of shape (L, 3)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    coords = []
    res_ids = []

    for model in structure:
        for chain in model:
            if chain.id == peptide_chain:
                for residue in chain:
                    if 'CA' in residue:
                        coords.append(residue['CA'].get_coord())
                        res_ids.append(residue.get_id()[1])
    coords = np.array(coords)
    return res_ids, coords


def extract_pocket_coords(pdb_path: str, pocket_chains: List[str]) -> np.ndarray:
    """
    Extract CA coords for all residues in given pocket chains.
    Args:
        pdb_path: path to pdb
        pocket_chains: list of chain IDs that form CRM1 pocket
    Returns:
        np.ndarray of shape (N,3)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    coords = []

    for model in structure:
        for chain in model:
            if chain.id in pocket_chains:
                for residue in chain:
                    if 'CA' in residue:
                        coords.append(residue['CA'].get_coord())
    return np.array(coords)


def compute_contact_labels(peptide_coords: np.ndarray, pocket_coords: np.ndarray, threshold=4.0) -> np.ndarray:
    """
    Binary labels for each peptide residue: 1 if residue is within threshold of any pocket atom, else 0
    """
    labels = np.zeros(len(peptide_coords), dtype=int)
    for i, res_coord in enumerate(peptide_coords):
        dists = np.linalg.norm(pocket_coords - res_coord, axis=1)
        if np.any(dists <= threshold):
            labels[i] = 1
    return labels


def process_sample(seq: str, pdb_path: str, peptide_chain: str, pocket_chains: List[str], output_csv: str, esm_embedding_size=1280):
    """
    Main pipeline per peptide:
    - Extract peptide coords and pocket coords from pdb
    - Compute contact labels
    - Get residue embeddings from ESM-2
    - Save CSV: rows = residues with embedding + label
    """
    model, alphabet, batch_converter, device = get_esm_model(esm_embedding_size)

    # Get embeddings (list of np arrays, one per peptide)
    embeddings_list = get_esm_embeddings([( "peptide", seq )], model, alphabet, batch_converter, device,
                                         embedding_layer=33, sequence_embedding=False)
    embeddings = embeddings_list[0]  # shape (L, embedding_dim)

    res_ids, pep_coords = extract_peptide_coords(pdb_path, peptide_chain)
    pocket_coords = extract_pocket_coords(pdb_path, pocket_chains)

    labels = compute_contact_labels(pep_coords, pocket_coords)

    assert len(seq) == embeddings.shape[0], f"Sequence length {len(seq)} vs embeddings length {embeddings.shape[0]}"
    assert len(labels) == embeddings.shape[0], f"Labels length {len(labels)} vs embeddings length {embeddings.shape[0]}"

    # Create dataframe
    df = pd.DataFrame(embeddings, columns=[f"e{i+1}" for i in range(embeddings.shape[1])])
    df["label"] = labels

    df.to_csv(output_csv, index=False)
    print(f"Saved processed data for peptide to {output_csv}")



if __name__ == "__main__":
    # Example usage - adjust peptide sequence and paths to your data
    fetch_nes_list()
    # example_seq = "ACDEFGHIKLMNPQRSTVWY"
    # example_pdb = "example_structure.pdb"
    # peptide_chain = 'P'  # assumed chain id for peptide
    # pocket_chains = ['A', 'B']  # example chains for CRM1 pocket (adjust to your data)
    # output_csv = "peptide_training_data.csv"
    #
    # process_sample(example_seq, example_pdb, peptide_chain, pocket_chains, output_csv)