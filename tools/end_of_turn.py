#!/usr/bin/env python3
"""End-of-turn check on short French coding prompts (thinking off), against the running server: how many replies run
to max_tokens instead of ending their turn, and P(end of turn) right after the reply's closing code fence (one token
drawn at T=1, top_p=1, 40 seeds, after the prompt plus the T=0 reply up to its fence, by token ids).

    API_URL=http://127.0.0.1:8888 tools/end_of_turn.py [label]

On 2x GB10 with this recipe: DENSE=q4 12/48 replies cut, P 0.55; fp8 1/48, 0.82; bf16 1/48, 0.76; exl3 1/48."""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = os.environ.get("API_URL", f"http://127.0.0.1:{os.environ.get('PORT', '8888')}")
MODEL = os.environ.get("MODEL", "GLM-5.3-Flash-EXL3")
TASKS = [
    "Écris une fonction Python `fusionne_tries(listes)` qui fusionne une liste de listes triées en une seule liste triée, en O(N log k) avec heapq. Réponds uniquement avec le code.",
    "Écris une classe Python `LRU(capacite)` avec `get(cle)` (renvoie -1 si absente) et `put(cle, valeur)`, éviction du moins récemment utilisé, O(1). Réponds uniquement avec le code.",
    "Écris une fonction Python `vers_romain(n)` (1 ≤ n ≤ 3999) qui renvoie le nombre en chiffres romains. Réponds uniquement avec le code.",
    "Écris une fonction Python `equilibre(s)` qui renvoie True si les parenthèses (), [], {} de la chaîne sont correctement imbriquées. Réponds uniquement avec le code.",
    "Écris une fonction Python `tri_topologique(n, aretes)` qui renvoie un ordre topologique des sommets 0..n-1 (aretes = liste de (u,v) signifiant u avant v), ou None s'il y a un cycle. Réponds uniquement avec le code.",
    "Écris une fonction Python `spirale(m)` qui renvoie les éléments d'une matrice (liste de listes) parcourue en spirale dans le sens horaire depuis le coin haut-gauche. Réponds uniquement avec le code.",
    "Écris deux fonctions Python `rle_encode(s)` et `rle_decode(s)` : encodage par plages sous la forme '3a2b1c' pour 'aaabbc' (compte puis caractère, pas de chiffres dans l'entrée), et l'inverse. Réponds uniquement avec le code.",
    "Écris une fonction Python `duree_secondes(s)` qui convertit une durée ISO 8601 de la forme PnDTnHnMnS (chaque composante optionnelle, entiers) en secondes. Réponds uniquement avec le code.",
]
OFF = {"chat_template_kwargs": {"enable_thinking": False}}


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(API + path, json.dumps({"model": MODEL, **body}).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def cut_runs() -> tuple[int, int]:
    jobs = [(p, 1000 + i, 0.7) for p in TASKS for i in range(5)] + [(p, None, 0.0) for p in TASKS]

    def one(job):
        p, seed, temp = job
        body = {"messages": [{"role": "user", "content": p}], "max_tokens": 1024, "temperature": temp, **OFF}
        if seed is not None:
            body["seed"] = seed
        return post("/v1/chat/completions", body)["choices"][0]["finish_reason"] == "length"

    with ThreadPoolExecutor(4) as ex:
        cut = list(ex.map(one, jobs))
    return sum(cut), len(cut)


def p_stop(prompt: str, draws: int = 40) -> float | None:
    msgs = [{"role": "user", "content": prompt}]
    ans = post("/v1/chat/completions", {"messages": msgs, "max_tokens": 1024, "temperature": 0, "return_token_ids": True, **OFF})
    gen = (ans.get("tensorfold") or {}).get("token_ids") or ans["choices"][0].get("token_ids")
    text, cut = "", None
    for k, t in enumerate(gen):
        text += post("/detokenize", {"tokens": [t]}).get("prompt", "")
        if text.count("```") >= 2:
            cut = k + 1
            break
    if cut is None:
        return None
    ids = post("/tokenize", {"messages": msgs, **OFF})["tokens"] + gen[:cut]
    stops = 0
    for seed in range(draws):
        c = post("/v1/completions", {"prompt": ids, "max_tokens": 1, "temperature": 1.0, "top_p": 1.0, "seed": seed})["choices"][0]
        stops += c["finish_reason"] == "stop" and not c["text"]
    return stops / draws


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else API
    n, total = cut_runs()
    ps = [p for p in (p_stop(t) for t in TASKS) if p is not None]
    print(f"{label}: replies cut by max_tokens {n}/{total}; P(end of turn) after the closing fence: "
          f"mean {sum(ps) / len(ps):.2f}, min {min(ps):.2f} ({len(ps)} tasks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
