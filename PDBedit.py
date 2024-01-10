import numpy as np
import pandas as pd
from collections import defaultdict

class PDBedit:
    def __init__(self):
        self.aa = ["GLY", "ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "TYR", "ASP", "ASN",
                   "GLU", "LYS", "GLN", "MET", "SER", "THR", "CYS", "PRO", "HIS", "ARG",
                   "HID", "ASH", "HIE", "HIP", "HSD", "HSE", "HSP"]
        self.one2three = {"G": "GLY", "A": "ALA", "V": "VAL", "L": "LEU", "I": "ILE",
                          "F": "PHE", "W": "TRP", "Y": "TYR", "D": "ASP", "N": "ASN",
                          "E": "GLU", "K": "LYS", "Q": "GLN", "M": "MET", "S": "SER",
                          "T": "THR", "C": "CYS", "P": "PRO", "H": "HIS", "R": "ARG"}
        self.three2one = {"GLY": "G", "ALA": "A", "VAL": "V", "LEU": "L", "ILE": "I",
                          "PHE": "F", "TRP": "W", "TYR": "Y", "ASP": "D", "ASN": "N",
                          "GLU": "E", "LYS": "K", "GLN": "Q", "MET": "M", "SER": "S",
                          "THR": "T", "CYS": "C", "PRO": "P", "HIS": "H", "ARG": "R"}

    def processIon(self, aa) -> str:  # Dealing with protonation conditions
        if aa in ['ASH']:
            return 'ASP'
        if aa in ['HIE', 'HID', 'HIP', 'HSD', 'HSE', 'HSP']:
            return 'HIS'
        return aa
    def readPDB_singleChain(self, path) -> (pd.DataFrame, str):
        # the chainID information is required
        print("Reading:", path)
        pdbinfo = defaultdict(list)

        with open(path, 'r') as file:
            for line in file.readlines():
                record = line.strip()
                ATOM = record[:4].strip()
                if ATOM != "ATOM":  # Detect ATOM start line
                    continue
                resName = self.processIon(record[17:20].strip())  # PRO, Treated protonation conditions
                if resName not in self.aa:
                    continue
                pdbinfo["serial"].append(int(record[6:11].strip()))  # 697
                pdbinfo["name"].append(record[12:16].strip())  # CA
                pdbinfo["resName"].append(resName)  # PRO, Treated protonation conditions
                pdbinfo["resSeq"].append(int(record[22:26].strip()))  # 3
                pdbinfo["x"].append(float(record[30:38].strip()))  # Å
                pdbinfo["y"].append(float(record[38:46].strip()))
                pdbinfo["z"].append(float(record[46:54].strip()))
                pdbinfo["plddt"].append(float(record[60:66].strip()))

        resName = [self.three2one[self.processIon(aa)] for aa in list(pdbinfo["resName"])]
        resSeq = list(pdbinfo["resSeq"])
        fasta = resName[0]
        for idx in range(len(resSeq) - 1):
            if resSeq[idx] != resSeq[idx + 1]:
                fasta += resName[idx + 1]

        return pd.DataFrame(pdbinfo).set_index("serial"), fasta


if __name__ == '__main__':
    pdbedit = PDBedit()
    pdbedit.readPDB_singleChain("/groups/sbinlab/fancao/IDPs_multi/calvados_sim/hecw2_CA.pdb")
    # pdbedit.set_bfactor("/groups/sbinlab/fancao/IDPs_multi", "THB_C2")
    # pdb, fasta = pdbedit.readPDB_singleChain("/groups/sbinlab/fancao/IDPs_multi/AF-Q546U4-F1-model_v4.pdb")
    # print(pdb.plddt)