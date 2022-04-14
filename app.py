import base64

import numpy as np
import pandas as pd
import requests
import streamlit as st

from crypto_tax_calculator import LotOrdering, Trades, run_trade_allocator


def main():
    st.title("Crypto Tax Lot Analyzer")

    page = st.sidebar.radio(
        "Pages", ["Introduction", "Optimization Setup", "Application"]
    )

    if page == "Introduction":
        write_markdown_file("docs/Introduction.md")
    elif page == "Optimization Setup":
        write_markdown_file("docs/Optimization Setup.md")
    else:
        run_application()


def write_markdown_file(file_name):
    with open(file_name, "r") as f:
        st.markdown(f.read())


def run_application():
    _set_max_width()
    trades = None
    source_data = st.sidebar.radio(
        "Data Source", ["Sample Data1", "Sample Data2", "Upload File"]
    )
    if source_data == "Sample Data1":
        filepath = "sample_data/sample_trades1.csv"
        trades = Trades.from_csv(filepath)
    elif source_data == "Sample Data2":
        filepath = "sample_data/sample_trades2.csv"
        trades = Trades.from_csv(filepath)
    elif source_data == "Upload File":
        data_file = st.sidebar.file_uploader(
            "Upload Trade File", type=["csv", "txt"])
        if data_file is not None:
            trades = Trades.from_csv(data_file)
        else:
            st.write(
                "Data format for trades is a 6 column csv/txt file with these columns names:"
            )
            filepath = "sample_data/sample_trades1.csv"
            sample_trades = Trades.from_csv(
                filepath).assign(hack="").set_index("hack")
            st.write(sample_trades)
            st.markdown(
                get_df_download_link(
                    sample_trades, "Download Sample File", "sample.csv"
                ),
                unsafe_allow_html=True,
            )
            st.write("(Note that the quantity column is always positive)")

    if trades is not None:
        preprocess = st.sidebar.checkbox(
            "Aggregate Trades At Same Timestamp", value=True
        )
        if preprocess:
            trades = Trades(trades.preprocess_trades())

        try:
            import cvxopt
            solvers = [
                "CVXOPT",
                "scipy.optimize.linprog"
            ]
        except:
            solvers = [
                "scipy.optimize.linprog"
                # "CVXOPT"
            ]
        opt_method = st.sidebar.radio(
            "Solver", solvers
        )
        if(opt_method == "scipy.optimize.linprog"):
            opt_method = st.sidebar.radio(
                "scipy.optimize.linprog algorithm",
                [
                    "revised simplex",
                    "simplex",
                    "highs-ds",
                    "highs-ipm",
                    "highs",
                    "interior-point"
                ]
            )

        st.sidebar.markdown(
            """<a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html">scipy.optimize.linprog</a>""",
            unsafe_allow_html=True,
        )
        selected_assets = st.sidebar.multiselect(
            "Assets",
            list(trades.get_asset_names()),
            default=list(trades.get_asset_names()),
        )
        trades = Trades(trades.get_trades(selected_assets))

        sample_trade_expander = st.expander(
            "Trade Data Click Here To Open --------------------->"
        )
        sample_trade_expander.write(trades.assign(hack="").set_index("hack"))
        sample_trade_expander.markdown(
            get_df_download_link(trades, "Download Data Here", "output.csv"),
            unsafe_allow_html=True,
        )
        col10, col11 = st.columns(2)
        ordering = col10.radio(
            "Lot Ordering", list(LotOrdering.__members__.keys()), index=3
        )
        split_holdings_from_sells = col11.radio(
            "How would you like the final Holdings/Transactions displayed?",
            ["Single table", "Split into separate tables"],
        )

        col1, col2 = st.columns(2)
        st_rate = col1.number_input("Short Term Tax Rate", value=0.35)
        lt_rate = col2.number_input("Long Term Tax Rate", value=0.15)

        holdings, transactions = run_trade_allocator(
            trades,
            lot_ordering=LotOrdering(ordering),
            lt_rate=lt_rate,
            st_rate=st_rate,
            method=opt_method,
        )

        results = pd.DataFrame(
            [
                [
                    transactions.get_st_tax_gains(),
                    transactions.get_lt_tax_gains(),
                    transactions.get_st_tax_losses(),
                    transactions.get_lt_tax_losses(),
                    transactions.get_tax_liability(st_rate, lt_rate),
                ]
            ],
            columns=[
                "Short Term Realized Gains",
                "Long Term Realized Gains",
                "Short Term Realized Losses",
                "Long Term Realized Losses",
                "Tax Liability",
            ],
            index=["$"],
        )
        st.dataframe(results.T)

        # Get current prices to compute unrealized gains
        current_prices = get_current_prices(selected_assets)
        for asset in selected_assets:
            open_pos = holdings["asset"] == asset
            if current_prices.get(asset) is not None:
                holdings.loc[open_pos,
                             "current_price"] = current_prices.get(asset)
            else:
                st.error(
                    f"Error getting price of asset: {asset}. Unrealized Gains cannot be displayed"
                )
                holdings.loc[open_pos, "current_price"] = np.nan

        holdings["unrealized_gains"] = holdings.quantity * (
            holdings.current_price - holdings.buy_price
        )
        for date_col in ["buy_date", "sell_date"]:
            transactions[date_col] = transactions[date_col].apply(
                lambda x: x.strftime("%Y-%m-%d") if pd.notnull(x) else np.nan
            )
        holdings["buy_date"] = holdings["buy_date"].apply(
            lambda x: x.strftime("%Y-%m-%d") if pd.notnull(x) else np.nan
        )

        if split_holdings_from_sells != "Single table":
            col6, col7 = st.columns(2)

            data = holdings.dropna(axis=1, how="all").sort_values("buy_date")
            col6.write("Current Holdings")
            col6.write(data)

            data = transactions.dropna(
                axis=1, how="all").sort_values("buy_date")
            col7.write("Past Transactions")
            col7.write(data)
        else:
            st.write("Current Holdings And Past Transactions")
            data = (
                pd.concat([holdings, transactions])
                .reset_index()
                .drop("index", axis=1)
                .sort_values("buy_date")
            )

            st.write(data)
        st.write(
            "THIS IS NOT FINANCIAL ADVICE... PLEASE CONSULT A CPA OR FINANCIAL ADVISOR FOR YOUR OWN TAX SITUATION."
        )


def get_df_download_link(
    df: pd.DataFrame, link_name: str = "Download Link", file_name: str = "download.csv"
) -> str:
    """Generates a download link for dataframe

    Args:
        df (pd.DataFrame): dataframe to write
        link_name (str, optional): Name to display in streamlit. Defaults to "Download Link".
        file_name (str, optional): Name of file. Defaults to "download.csv".

    Returns:
        str: link to write to st.markdown
    """

    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return (
        f'<a href="data:file/csv;base64,{b64}"  download="{file_name}">{link_name}</a>'
    )


@st.cache
def get_current_prices(asset_names: list, currency: str = "usd") -> dict:
    """Get current asset prices from Gemini to compute unrealized gains/losses

    Args:
        asset_names (list): list of assets to obtain prices for
        currency (str, optional): currency for price. Defaults to "usd".

    Returns:
        dict: latest prices from Gemini exchange
    """

    prices = {}

    for asset in asset_names:
        response = requests.get(
            "https://api.gemini.com/v1/trades/" + asset.lower() + currency
        )
        if response.status_code == 200:
            prices[asset] = float(response.json()[0]["price"])
        else:
            prices[asset] = None
    return prices


def _set_max_width():
    st.markdown(
        """
    <style>
    .reportview-container .main .block-container{max-width: 2000px;}
    </style>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
