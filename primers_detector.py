from Bio.Align import PairwiseAligner
import json
from Bio import SeqIO

# задаём праймеры
primers = {
    "PL":    "CTTCATGGATCCTGCTCTCG",
    "PR":    "GGCCCTAAAGCTTAGCACGA",
    "RC_PL": "CGAGAGCAGGATCCATGAAG",
    "RC_PR": "TCGTGCTAAGCTTTAGGGCC",
}

# 1) configure a local (Smith–Waterman) aligner
aligner = PairwiseAligner()
aligner.mode = "local"
aligner.match_score = 1  # +1 for a match
aligner.mismatch_score = 0  # 0 for a mismatch
aligner.open_gap_score = -1  # penalty for opening a gap
aligner.extend_gap_score = -1  # penalty for extending a gap

def fuzzy_search_primer(seq: str, primer: str) -> dict:
    """
    Align `primer` against `seq` locally, then compute:
      - score == number of matched positions
      - percent_identity = 100 * matches / primer_length
    """
    # run the local alignment and pick the best
    best_alignment = aligner.align(primer, seq)[0]

    matches = best_alignment.score
    pct_id = matches / len(primer) * 100

    return {
        "score": int(matches),
        "percent_identity": round(pct_id, 1),
        "primer_length": len(primer),
        "aligned_query_range": best_alignment.aligned[1],  # (start,end) on seq
    }

def analyze(fasta_path, threshold=70.0):
    results = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seq = str(rec.seq)
        hit_info = {
            name: fuzzy_search_primer(seq, p)
            for name, p in primers.items()
        }
        results.append({"seq_id": rec.id, "hits": hit_info})

    # summary:
    total = len(results)
    above_thr = sum(
        1 for r in results
        if any(h["percent_identity"] >= 80 for h in r["hits"].values())
    )
    summary = {
        "total_sequences": total,
        "sequences_≥80%": above_thr,
        "percent_≥80%": round(above_thr / total * 100, 1)
    }

    return results, summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("fasta", help="FASTA file with input sequences")
    parser.add_argument("-t","--threshold", type=float, default=80.0,
                        help="min percent identity to count as hit")
    parser.add_argument("-o","--out", help="JSON output file", default=None)
    args = parser.parse_args()

    results, summary = analyze(args.fasta, threshold=args.threshold)
    #print(json.dumps({"detailed": results, "summary": summary}, indent=2))
    print(results)
    print(summary)