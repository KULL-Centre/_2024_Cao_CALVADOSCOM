import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import os
import pandas as pd
import MDAnalysis as mda
from Bio import SeqIO, SeqUtils
import yaml

def fasta_from_pdb(pdb):
    """ Generate fasta from pdb entries """
    u = mda.Universe(pdb)
    res3 = "".join(u.residues.resnames)
    fastapdb = SeqUtils.seq1(res3)
    return fastapdb

def modProtein(df,name,**kwargs):
    for key,val in kwargs.items():
        print(key,val)
        if key not in df.columns:
            df[key] = None # initialize property, does not work with lists
        df.loc[name,key] = val
    return df

def delProtein(df,name):
    df = df.drop(name)
    return df

def subset(df,names):
    df2 = df.loc[names]
    return df2

def read_fasta(ffasta):
    records = SeqIO.to_dict(SeqIO.parse(ffasta, "fasta"))
    return records



def get_ssdomains(name,fdomains, output=True):
    with open(fdomains,'r') as f:
        stream = f.read()
        domainbib = yaml.safe_load(stream)

    domains = domainbib[name]
    if output:
        print(f'Using domains {domains}')
    ssdomains = []

    for domain in domains:
        if isinstance(domain[0], int):
            ssdomains.append(list(range(domain[0],domain[1]+1)))
        else:  # corresponding to discreet regions
            tmp = []
            for subdomain in domain:
                tmp += list(range(subdomain[0], subdomain[1]+1))
            ssdomains.append(tmp)
    return ssdomains  # [[1, 2, 3, 4], [5, 6, 7]]

def AllMDpros(fdomains):
    # return all multi domain protein names
    with open(fdomains,'r') as f:
        stream = f.read()
        domainbib = yaml.safe_load(stream)

    return list(domainbib.keys())

def output_fasta(cwd, path2pdbfolder):
    """Output all sequences unber ${path} to every single fasta file in ${multidomain_fasta}"""
    # https://biopython.org/wiki/SeqRecord
    os.system(f"ls {path2pdbfolder} > {path2pdbfolder}/pdbnames.txt")
    with open(f"{path2pdbfolder}/pdbnames.txt", 'r') as file:
        for line in file.readlines():
            record = line.strip().split(".")
            if record[-1] == "pdb":
                fasta = fasta_from_pdb(f"{path2pdbfolder}/{record[0]}.pdb")
                print(record[0], fasta)
                fasta2save = SeqRecord(Seq(fasta), id=record[0], name=record[0], description=record[0])
                with open(f"{cwd}/multidomain_fasta/{record[0]}.fasta", "w") as output_handle:
                    SeqIO.write(fasta2save, output_handle, "fasta")


if __name__ == '__main__':
    # cwd = "/groups/sbinlab/fancao/IDPs_multi"
    # print(get_ssdomains("Gal3", f'{cwd}/domains.yaml'))
    # path2pdbfolder = f"{cwd}/pdbfolder"
    # output_fasta(cwd, path2pdbfolder)
    pass