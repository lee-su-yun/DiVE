import re
import sys

INPUT = "/data/APOBU/beliefmap_low_occlusion_0423/tipped_log.txt"
OUTPUT = "/data/APOBU/beliefmap_low_occlusion_0423/tipped_log_revised.txt"

# pat = re.compile(r"^(episode=\d+)\s+post_(\d+)\s+(fallen=.*)$")
pat = re.compile(r"^(episode=\d+)\s+post_(\d+)\s+(tipped=.*)$")

out_lines = []
with open(INPUT, "r") as f:
    for ln, line in enumerate(f, 1):
        raw = line.rstrip("\n")
        if raw.strip() == "":
            sys.exit(f"ERROR: unexpected blank line at {ln}")
        m = pat.match(raw)
        if not m:
            sys.exit(f"ERROR: unrecognized format at line {ln}: {raw!r}")
        ep, post_str, fallen = m.group(1), m.group(2), m.group(3)
        p = int(post_str)
        if p < 0 or p > 86:
            sys.exit(f"ERROR: post index {p} out of range [0,86] at line {ln}: {raw!r}")
        if p <= 28:
            push_idx, new_post = 1, p
        elif p <= 57:
            push_idx, new_post = 2, p - 29
        else:
            push_idx, new_post = 3, p - 58
        out_lines.append(f"{ep}  push_{push_idx}  post_{new_post:02d}  {fallen}")

with open(OUTPUT, "w") as f:
    f.write("\n".join(out_lines) + "\n")

print(f"OK: wrote {len(out_lines)} lines to {OUTPUT}")
