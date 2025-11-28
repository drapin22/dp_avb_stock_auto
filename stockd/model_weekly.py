def load_all_holdings() -> pd.DataFrame:
    """
    Încarcă toate deținerile din fișierele:
      - data/holdings_ro.csv
      - data/holdings_eu.csv
      - data/holdings_us.csv

    și întoarce un DataFrame cu coloanele:
      ['Ticker', 'Region']

    Are și loguri [DEBUG] ca să vezi exact ce se întâmplă în GitHub Actions.
    """
    dfs: list[pd.DataFrame] = []

    files = [
        (settings.HOLDINGS_RO, "RO"),
        (settings.HOLDINGS_EU, "EU"),
        (settings.HOLDINGS_US, "US"),
    ]

    print("[DEBUG] Loading holdings from:")
    for path, region_default in files:
        print(f"  - {path} (default Region={region_default})")

    for path, region_default in files:
        if not path.exists():
            print(f"[DEBUG] {path} NOT FOUND, skip.")
            continue

        df = pd.read_csv(path)
        print(f"[DEBUG] {path.name}: {len(df)} rows RAW")

        # Dacă nu avem coloană Region, o completăm cu implicitul
        if "Region" not in df.columns:
            df["Region"] = region_default

        # Dacă există coloana Active, păstrăm doar rândurile Active == 1
        if "Active" in df.columns:
            before = len(df)
            df = df[df["Active"] == 1]
            print(
                f"[DEBUG] {path.name}: {before} → {len(df)} rows after filter Active=1"
            )

        # Dacă după filtrare nu mai e nimic, trecem la următorul fișier
        if df.empty:
            print(f"[DEBUG] {path.name}: EMPTY after filtering, skip.")
            continue

        # Păstrăm doar coloanele necesare
        df_small = df[["Ticker", "Region"]].copy()
        dfs.append(df_small)

    if not dfs:
        print("[DEBUG] No holdings combined – returning EMPTY DataFrame")
        return pd.DataFrame(columns=["Ticker", "Region"])

    combined = pd.concat(dfs, ignore_index=True).drop_duplicates()
    print(f"[DEBUG] Combined holdings: {len(combined)} rows")
    return combined
