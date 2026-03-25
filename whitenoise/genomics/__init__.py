# whitenoise.genomics — SWNA bridge for genomic DNA sequences
#
# This submodule provides the DNA-specific pipeline that converts raw
# genomic sequences (FASTA format) into distance arrays suitable for
# SWNA analysis via the core whitenoise package.
#
# Typical workflow:
#   import whitenoise as wn
#
#   records = wn.genomics.read_fasta('genome.fasta')
#   seq = records[0]['sequence']
#
#   # Single transition pair
#   distances = wn.genomics.get_transition_distances(seq, 'A', 'T')
#   lags, msd = wn.compute_msd(distances)
#   result = wn.analyze(distances, model='dna', label='A->T')
#
#   # Full 4x4 transition matrix
#   matrix = wn.genomics.get_all_transition_distances(seq)
#   # matrix[('A', 'T')] -> np.ndarray of distances

from .io import read_fasta, read_fasta_single, filter_sequence, parse_seqrecord
from .distances import get_transition_distances, get_all_transition_distances, summarize_matrix
from .pipeline import extract_pair, compute_pair_msd, fit_pair, plot_pair, analyze_pair, refit_pair

__all__ = [
    'read_fasta',
    'read_fasta_single',
    'filter_sequence',
    'parse_seqrecord',
    'get_transition_distances',
    'get_all_transition_distances',
    'summarize_matrix',
    'extract_pair',
    'compute_pair_msd',
    'fit_pair',
    'plot_pair',
    'analyze_pair',
    'refit_pair',
]
