#!/usr/bin/env python3
"""redteam-sandbox CLI.

    python run.py attack --generations 6 --population 12
    python run.py sweep
    python run.py probe "your prompt here"
"""
from __future__ import annotations

import argparse
import json

from sandbox import defenses, evolve, llm, report, target


def _config(args) -> dict:
    return {
        "use_heuristics": not args.no_heuristics,
        "use_semantic": not args.no_semantic,
        "use_guard": not args.no_guard,
        "use_safeguard": not args.no_safeguard,
        "threshold": args.threshold,
    }


def cmd_attack(args):
    config = _config(args)
    print(f"target: {llm.TARGET_MODEL}   attacker: {llm.ATTACKER_MODEL}")
    print(f"defence: {', '.join(k for k, v in config.items() if v is True) or 'none'} "
          f"(guard threshold {args.threshold})\n")

    # Undefended baseline first. Without it, a low attack success rate could
    # equally mean "strong guardrails" or "the target was never going to leak".
    print("baseline: seed attacks, no guardrails")
    off = {"use_heuristics": False, "use_semantic": False,
           "use_guard": False, "use_safeguard": False}
    baseline = evolve.run(generations=1, population_size=len(evolve.A.SEEDS),
                          config=off, seed=args.seed, verbose=True)
    baseline_summary = {"asr": baseline.asr(), "trials": len(baseline.trials),
                        "leaked": len(baseline.leaks())}
    print(f"  undefended ASR {baseline.asr():.0%} "
          f"({len(baseline.leaks())}/{len(baseline.trials)})\n")

    print(f"evolving {args.generations} generations x {args.population} attacks")
    result = evolve.run(generations=args.generations, population_size=args.population,
                        config=config, seed=args.seed)

    print("\nmeasuring false positives on benign traffic")
    fp = evolve.measure_false_positives(config)
    print(f"  {fp['blocked']}/{fp['n']} benign prompts blocked "
          f"({fp['false_positive_rate']:.0%})")

    summary = report.summarise(result)
    suffix = f"_{args.label}" if args.label else ""
    evolve.save(result, extra={"summary": summary, "false_positives": fp,
                               "baseline": baseline_summary},
                path=evolve.RESULTS.with_name(f"run{suffix}.json"))
    path = report.build(result, summary, fp, baseline=baseline_summary,
                        path=report.REPORT.with_name(f"report{suffix}.html"))

    print(f"\nattack success rate {summary['asr']:.0%} "
          f"({summary['leaked']}/{summary['trials']}) "
          f"vs {baseline.asr():.0%} undefended")
    print(f"blocked {summary['block_rate']:.0%}  |  benign FPR {fp['false_positive_rate']:.0%}")
    print(f"layers: {summary['by_layer']}")
    print(f"report -> {path}")


def cmd_sweep(args):
    """Trace the block-rate / false-positive curve for the Prompt Guard layer."""
    attacks = [evolve.A.Attack(prompt=p, strategy=s) for s, p in evolve.A.SEEDS.items()]
    scores = [defenses.guard_score(a.prompt) for a in attacks]
    benign = [defenses.guard_score(p) for p in defenses.BENIGN_PROMPTS]

    from sandbox import semantic
    sep = semantic.separation([a.prompt for a in attacks], defenses.BENIGN_PROMPTS)
    if sep:
        print("semantic layer (local MiniLM, cosine to attack corpus)")
        print(f"{'thresh':>7} {'attacks blocked':>16} {'benign blocked':>15}")
        for th in (0.35, 0.45, 0.48, 0.55, 0.60, 0.70):
            tp = sum(x >= th for x in sep["attack_scores"]) / len(sep["attack_scores"])
            fp = sum(x >= th for x in sep["benign_scores"]) / len(sep["benign_scores"])
            print(f"{th:>7.2f} {tp:>15.0%} {fp:>15.0%}")
        print(f"  attack min {sep['attack_min']:.3f} vs benign max {sep['benign_max']:.3f}\n")

    print("prompt-guard layer (hosted Llama Prompt Guard 2)")
    rows = []
    for threshold in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
        rows.append({
            "threshold": threshold,
            "block_rate": sum(s >= threshold for s in scores) / len(scores),
            "false_positive_rate": sum(s >= threshold for s in benign) / len(benign),
        })

    print(f"{'thresh':>7} {'attacks blocked':>16} {'benign blocked':>15}")
    for r in rows:
        print(f"{r['threshold']:>7.2f} {r['block_rate']:>15.0%} "
              f"{r['false_positive_rate']:>15.0%}")

    out = evolve.RESULTS.parent / "sweep.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "sweep": rows,
        "attack_scores": dict(zip(evolve.A.SEEDS, scores)),
        "benign_scores": benign,
    }, indent=2))
    print(f"\nsweep -> {out}")


def cmd_probe(args):
    """Run one prompt through the stack and show what each layer thought."""
    verdict = defenses.screen(args.prompt, **_config(args))
    print(f"layers run : {verdict.layers_run}")
    print(f"guard score: {verdict.guard_score}")
    print(f"verdict    : {'BLOCKED by ' + verdict.layer if verdict.blocked else 'ALLOWED'}")
    print(f"reason     : {verdict.reason}")
    if not verdict.blocked:
        response = target.ask(args.prompt)
        print(f"\nleaked: {response.leaked}   refused: {response.refused}")
        print(report.redact(response.text[:500]))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threshold", type=float, default=defenses.GUARD_THRESHOLD)
    p.add_argument("--no-heuristics", action="store_true")
    p.add_argument("--no-semantic", action="store_true")
    p.add_argument("--no-guard", action="store_true")
    p.add_argument("--no-safeguard", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("attack", help="evolve attacks and write the report")
    a.add_argument("--generations", type=int, default=6)
    a.add_argument("--population", type=int, default=12)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--label", default="", help="suffix for report/results filenames")
    a.set_defaults(func=cmd_attack)

    s = sub.add_parser("sweep", help="Prompt Guard threshold vs false positives")
    s.set_defaults(func=cmd_sweep)

    b = sub.add_parser("probe", help="run one prompt through the stack")
    b.add_argument("prompt")
    b.set_defaults(func=cmd_probe)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
