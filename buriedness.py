#!/usr/bin/env python
"""
Script to calculate the buriedness of amino acids given a pdb file

Usage:

$ python buriedness.py file.pdb > output.csv
"""
from __future__ import print_function

__author__ = "Rodrigo Dorantes-Gilardi"
__email__ = "rodgdor@gmail.com"

import sys
import Bio.PDB
import numpy as np
from scipy.spatial import ConvexHull


def get_buriedness(prot):
    """
    input
    -----
    prot: str. Path to PDB file

    output
    ------
    list: Buriedness of each amino acid.
    """

    BDNESS = {}  # buriedness
    parser = Bio.PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(prot, prot)[0]
    except FileNotFoundError:
        print("Cannot find the file".format(prot))

    # Remove water and hetatoms
    for res in structure.get_residues():
        chain = res.parent
        if res.id[0] != " ":
            chain.detach_child(res.id)

    atoms = [atom for atom in structure.get_atoms()]
    if not atoms:
        raise Exception("Could not parse atoms in the pdb file")

    conv = ConvexHull([atom.coord for atom in atoms])
    for i, atom in enumerate(atoms):
        coord = atom.coord  # atom coordinates
        res = (atom.parent.parent.id  # chain + resname + position
              + atom.parent.resname + str(atom.parent.id[1]))
        BDNESS.setdefault(res, [])
        # Get distance from atom to closer face of `conv'
        if i in conv.vertices:
            dist = 0 
            BDNESS[res].append(dist)
        else:
            dist = np.inf
            for face in conv.equations:
                _dist = abs(np.dot(coord, face[:-1]) + face[-1])
                _dist = _dist / np.linalg.norm(face[:-1])
                if _dist < dist:
                    dist = _dist
            BDNESS[res].append(dist)
    for res in BDNESS:
        BDNESS[res] = np.mean(BDNESS[res])

    return BDNESS


if __name__ == "__main__":
    args = sys.argv
    if args[1][-3:] != "pdb":
        raise Exception("Error: {0} does not finish with '.pdb'".format(args[1][-3:]))
    buriedness = get_buriedness(args[1])
    print("Residue (chain+name+position), buriedness (angstrom)")
    for residue in sorted(buriedness, key= lambda x: x[0] + x[4:]):
        print("{0}, {1:.2f}".format(residue, buriedness[residue]))
