#!/bin/bash
# Parallel PDB download from RCSB
PDB_DIR="/Users/shaharharel/Documents/github/edit-small-mol/data/covbinder_inpdb/PDB"
LIST="/Users/shaharharel/Documents/github/edit-small-mol/data/covbinder_inpdb/to_download.txt"
mkdir -p "$PDB_DIR"

cat "$LIST" | xargs -P 16 -I {} bash -c '
  pdb="$1"
  dest="'$PDB_DIR'/${pdb}.pdb"
  if [ -s "$dest" ]; then
    exit 0
  fi
  curl -sL --max-time 30 "https://files.rcsb.org/download/${pdb}.pdb" -o "$dest"
  if [ ! -s "$dest" ]; then
    # Try .pdb.gz
    curl -sL --max-time 30 "https://files.rcsb.org/download/${pdb}.pdb.gz" -o "${dest}.gz"
    if [ -s "${dest}.gz" ]; then
      gunzip "${dest}.gz"
    fi
  fi
' _ {}

# Report
n_downloaded=$(ls "$PDB_DIR" | wc -l)
n_requested=$(wc -l < "$LIST")
echo "Downloaded $n_downloaded / $n_requested PDBs to $PDB_DIR"
