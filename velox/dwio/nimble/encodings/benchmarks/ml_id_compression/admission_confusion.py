#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Confusion matrix of the SubIntSplit admission heuristic.

Reads CSVs written by nimble_ml_id_admission_benchmark. A column is a positive
when the smallest arm named in --sis_arms is more than --margin smaller than
the smallest arm outside the SubIntSplit family and outside openzl. The
heuristic's prediction is policy_selects_sis (--decision=policy),
estimate_admits (--decision=estimate), or a bit-flip admission row's decision
(--decision=bitflip|bitflip_entropy, at --profile_pairs). Ground-truth encoder
rows may come from different CSVs than the heuristic rows, e.g. an earlier
sweep at the same commit.

  admission_confusion.py out/*.csv --sis_arms SIS/realNested,SIS/hybrid
"""

import argparse
import csv
import sys
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+")
    parser.add_argument("--sis_arms", default="SIS/realNested,SIS/hybrid")
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--decision", choices=["policy", "estimate", "bitflip", "bitflip_entropy"],
                        default="policy")
    parser.add_argument("--profile_pairs", default="0")
    args = parser.parse_args()
    sisArms = args.sis_arms.split(",")

    heuristic = {}
    admission = {}
    arms = defaultdict(dict)
    encodeNanos = defaultdict(dict)
    for path in args.csvs:
        with open(path) as handle:
            for row in csv.DictReader(handle):
                if row.get("skipped") == "1":
                    continue
                dataset = row["dataset"]
                if row["row_kind"] == "heuristic":
                    heuristic[dataset] = row
                elif row["row_kind"] == "admission":
                    admission[(dataset, row["admission_mode"], row["profile_pairs"])] = row
                elif row["row_kind"] == "encoder":
                    arms[dataset][row["encoding"]] = int(row["payload_bytes"])
                    encodeNanos[dataset][row["encoding"]] = int(row["encode_ns"])

    cells = defaultdict(list)
    print("dataset,predicted,actual,sis_arm,sis_bytes,best_other,other_bytes,"
          "heuristic_ns,sis_encode_ns")
    for dataset in sorted(heuristic):
        present = [arm for arm in sisArms if arm in arms[dataset]]
        others = {
            arm: size
            for arm, size in arms[dataset].items()
            if not arm.startswith("SIS/") and not arm.startswith("openzl")
        }
        if not present or not others:
            print(f"{dataset},missing arms", file=sys.stderr)
            continue
        sisArm = min(present, key=lambda arm: arms[dataset][arm])
        otherArm = min(others, key=others.get)
        actual = arms[dataset][sisArm] < (1.0 - args.margin) * others[otherArm]
        if args.decision in ("policy", "estimate"):
            key = "policy_selects_sis" if args.decision == "policy" else "estimate_admits"
            predicted = heuristic[dataset][key] == "1"
            decisionNanos = heuristic[dataset]["heuristic_ns"]
        else:
            row = admission[(dataset, args.decision, args.profile_pairs)]
            predicted = row["decision"] == "1"
            decisionNanos = row["decision_ns"]
        cells[(predicted, actual)].append(dataset)
        print(f"{dataset},{int(predicted)},{int(actual)},{sisArm},"
              f"{arms[dataset][sisArm]},{otherArm},{others[otherArm]},"
              f"{decisionNanos},{encodeNanos[dataset][sisArm]}")

    truePositive = len(cells[(True, True)])
    falsePositive = len(cells[(True, False)])
    falseNegative = len(cells[(False, True)])
    precision = truePositive / (truePositive + falsePositive) if truePositive + falsePositive else float("nan")
    recall = truePositive / (truePositive + falseNegative) if truePositive + falseNegative else float("nan")
    for (predicted, actual), label in [((True, True), "TP"), ((True, False), "FP"),
                                       ((False, True), "FN"), ((False, False), "TN")]:
        print(f"{label} {len(cells[(predicted, actual)])}: {' '.join(cells[(predicted, actual)])}")
    print(f"precision {precision:.3f} recall {recall:.3f}")


if __name__ == "__main__":
    main()
