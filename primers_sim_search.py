from Bio.Align import PairwiseAligner
from Bio.Seq import Seq
import json


def parse_primers(primer_args, add_rc=False):
    primers = {}
    for item in primer_args:
        if '=' not in item:
            raise ValueError(f"Invalid primer format: {item}. Expected NAME=SEQUENCE")
        name, seq = item.split('=', 1)
        primers[name] = seq.upper()
    if add_rc:
        rc_primers = {
            f"RC_{name}": str(Seq(seq).reverse_complement())
            for name, seq in primers.items()
        }
        primers.update(rc_primers)
    return primers


# Configure a local (Smith–Waterman) aligner
aligner = PairwiseAligner()
aligner.mode = "local"
aligner.match_score = 1
aligner.mismatch_score = 0
aligner.open_gap_score = -1
aligner.extend_gap_score = -1


def mark_sequence(seq: str, name: str, flat_ranges: list) -> str:
    """
    Mark the input `seq` with XML-style tags using alignment ranges.
    """
    marked_seq = seq
    if not flat_ranges:
        return marked_seq
    start = min(block[0] for block in flat_ranges)
    end = max(block[1] for block in flat_ranges)
    marked_seq = (
            marked_seq[:start]
            + f"<{name}>" + marked_seq[start:end] + f"</{name}>"
            + marked_seq[end:]
    )
    return marked_seq


def format_alignment_graphical(alignment, width: int = 60) -> str:
    """
    Return a graphical alignment as a multi-line string with proper wrapping, containing only the aligned sequences
    and match line, without labels or indices.

    Parameters:
    - alignment: Bio.Align.PairwiseAlignment object
    - width: int, number of characters per line (default 60)

    Returns:
    - str, formatted multi-line alignment
    """
    # Get aligned sequences directly from alignment
    aligned_str = str(alignment).split('\n')
    if len(aligned_str) < 3:
        return "No valid alignment found."

    primer_aligned = aligned_str[0]  # Query sequence (primer)
    seq_aligned = aligned_str[2]     # Target sequence

    # Create match line: '|' for exact matches, ' ' for mismatches or gaps
    match_line = ''
    for p, s in zip(primer_aligned, seq_aligned):
        if p == s and p != '-' and s != '-':
            match_line += '|'
        else:
            match_line += ' '

    # Split into lines of specified width
    lines = []
    for i in range(0, len(primer_aligned), width):
        p_seg = primer_aligned[i:i+width]
        m_seg = match_line[i:i+width]
        s_seg = seq_aligned[i:i+width]

        # Pad segments to ensure consistent length
        max_len = max(len(p_seg), len(m_seg), len(s_seg))
        p_seg = p_seg.ljust(max_len)
        m_seg = m_seg.ljust(max_len)
        s_seg = s_seg.ljust(max_len)

        lines.append(p_seg)
        lines.append(m_seg)
        lines.append(s_seg)
        lines.append("")  # Empty line between blocks

    return "\n".join(lines)

def fuzzy_search_primer(seq: str, name: str, primer: str) -> dict:
    alignment = aligner.align(primer, seq)[0]

    # Extract aligned regions
    primer_aligned = ''
    seq_aligned = ''
    for (i_start, i_end), (j_start, j_end) in zip(*alignment.aligned):
        primer_aligned += primer[i_start:i_end]
        seq_aligned += seq[j_start:j_end]

    # Compute number of matches
    matches = sum(1 for a, b in zip(primer_aligned, seq_aligned) if a == b)
    alignment_len = len(primer_aligned)

    percent_identity = 100 * matches / len(seq) if matches > 0 else 0

    # Store graphical alignment separately
    graphical = format_alignment_graphical(alignment)

    return {
        "primer_sequence": primer,
        "score": matches,
        "percent_identity": round(percent_identity, 1),
        "primer_length": len(primer),
        "alignment_length": alignment_len,
        "aligned_query_range": alignment.aligned[1].tolist(),
        "marked_sequence": mark_sequence(seq, name, alignment.aligned[1].tolist()),
        "alignment": {
            "primer_aligned": primer_aligned,
            "sequence_aligned": seq_aligned
        },
        "_graphical": graphical  # Temporary storage for printing
    }


def analyze_sequence(seq: str, primers: dict, threshold=80.0):
    hit_info = {
        name: fuzzy_search_primer(seq, name, p)
        for name, p in primers.items()
    }

    summary = {
        "sequence_length": len(seq),
        "detected": {
            name: info for name, info in hit_info.items()
            if info["percent_identity"] >= threshold
        },
    }

    return hit_info, summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search primers in a DNA sequence")
    parser.add_argument("sequence", help="Input DNA sequence (as string)")
    parser.add_argument(
        "-p", "--primer", action="append", required=True,
        help="Primer in the format NAME=SEQUENCE. Use multiple times for multiple primers."
    )
    parser.add_argument(
        "--add-rc", action="store_true",
        help="Automatically add reverse complements of primers"
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=80.0,
        help="Minimum percent identity to count as hit"
    )
    parser.add_argument(
        "-o", "--out", help="Optional JSON output file"
    )
    args = parser.parse_args()

    primers = parse_primers(args.primer, add_rc=args.add_rc)
    seq = args.sequence.upper()

    results, summary = analyze_sequence(seq, primers, threshold=args.threshold)

    # Print graphical alignments separately
    print("Graphical Alignments:")
    print("-" * 80)
    for name, info in results.items():
        print(f"Primer: {name}")
        print(info["_graphical"])
        print(info["marked_sequence"])
        print("-" * 80)

    # Prepare JSON output without graphical field
    output = {"hits": {}, "summary": summary}
    for name, info in results.items():
        output["hits"][name] = {k: v for k, v in info.items() if k != "_graphical"}

    if args.out:
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2)
    else:
        print("\nJSON Results:")
        print(json.dumps(output, indent=2))