def test_dump_titles(capsys):
    import radiance
    cls = dict(radiance.NODE_CLASS_MAPPINGS)
    disp = dict(radiance.NODE_DISPLAY_NAME_MAPPINGS)
    from collections import defaultdict
    groups = defaultdict(list)
    for key, c in cls.items():
        title = disp.get(key, key)
        cat = str(getattr(c, "CATEGORY", "") or "")
        section = cat.split("/")[-1] if cat else "?"
        groups[section].append(title)
    with capsys.disabled():
        total = 0
        # detect duplicate titles
        from collections import Counter
        allt = Counter(disp.values())
        dups = {t: n for t, n in allt.items() if n > 1}
        for sec in sorted(groups):
            names = sorted(groups[sec])
            total += len(names)
            print(f"\n### {sec}  ({len(names)})")
            for n in names:
                print(f"  {n}")
        print(f"\n=== TOTAL {total} nodes ===")
        print("DUPLICATE TITLES:", dups if dups else "none")
