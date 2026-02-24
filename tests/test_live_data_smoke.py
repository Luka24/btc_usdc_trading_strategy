from data_fetcher import DataFetcher


def test_live_data_fetch_smoke():
    df = DataFetcher.fetch_combined_data(days=5, use_real_data=True)
    assert not df.empty
    assert {"date", "btc_price", "hashrate_eh_per_s"}.issubset(df.columns)
    assert (df["btc_price"] > 0).all()
    assert (df["hashrate_eh_per_s"] > 0).all()


def test_synthetic_mode_blocked():
    failed = False
    try:
        DataFetcher.fetch_combined_data(days=5, use_real_data=False)
    except RuntimeError:
        failed = True
    assert failed, "Expected RuntimeError when synthetic mode is requested"


if __name__ == "__main__":
    test_live_data_fetch_smoke()
    test_synthetic_mode_blocked()
    print("test_live_data_smoke.py: PASS")
