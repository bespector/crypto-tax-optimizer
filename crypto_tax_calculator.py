import datetime
import math
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from utils import Holdings, LotOrdering, Trades, Transactions


def run_trade_allocator(
    trades: Trades,
    lot_ordering: LotOrdering = LotOrdering.TAX_OPTIMAL,
    lt_rate: float = 0.15,
    st_rate: float = 0.35,
    method: str = "revised simplex",
) -> Tuple[Holdings, Transactions]:
    """Finds optimal allocations based off a set of trades and account method

    Args:
        trades (Trades): list of buys/sells
        lot_ordering (LotOrdering, optional): accounting method. Defaults to LotOrdering.TAX_OPTIMAL.
        lt_rate (float, optional): long term capital gains rate. Defaults to 0.15.
        st_rate (float, optional): short term capital gains rate. Defaults to 0.35.
        method (str, optional): scipy.optimize.linprog algo. Defaults to "highs-ds".

    Raises:
        Exception: if optimization cannot find solution

    Returns:
        Tuple[Holdings, Transactions]: Final Holdings and what to allocate
    """
    holdings = Holdings()
    transactions = Transactions()
    for asset in trades.get_asset_names():
        if(len(trades.get_sells(asset)) > 0):
            problem = setup_linear_optimization(
                asset, trades, lot_method=lot_ordering, lt_rate=lt_rate, st_rate=st_rate, method=method
            )
            if(method == "CVXOPT"):
                import cvxopt
                from cvxopt import matrix
                cvxopt.solvers.options['show_progress'] = False
                solution = cvxopt.solvers.lp(c=matrix(problem['obj']),
                                             G=matrix(problem['A_ub'].values),
                                             h=matrix(problem['b_ub']),
                                             A=matrix(problem["A_eq"].values),
                                             b=matrix(problem['b_eq']))
                if(solution['status'] != 'optimal'):
                    print(solution['status'])
                    raise Exception("Error solving optimization problem!")
                x = list(solution['x'])
            else:
                solution = linprog(
                    problem["obj"],
                    A_ub=problem["A_ub"],
                    b_ub=problem["b_ub"],
                    A_eq=problem["A_eq"],
                    b_eq=problem["b_eq"],
                    bounds=problem["bounds"],
                    method=method,
                )
                if(solution.status != 0):
                    print(solution.status)
                    raise Exception("Error solving optimization problem!")
                x = list(solution.x)
            # print(x)
            column_names = problem["A_eq"].columns
            asset_holdings, asset_transactions = allocate_asset_decision_variables(
                asset, trades, x, column_names
            )

            holdings = asset_holdings + holdings
            transactions = asset_transactions + transactions
        else:
            holdings = holdings + Holdings(trades.get_buys(asset))
    return holdings, transactions


def setup_linear_optimization(
    asset: str,
    trades: Trades = None,
    lot_method: LotOrdering = LotOrdering.TAX_OPTIMAL,
    lt_rate: float = 0.15,
    st_rate: float = 0.35,
    method="CVXOPT"
) -> dict:
    """Creates the inputs into scipy.optimize.linprog

    Args:
        asset (str): asset to setup problem for
        trades (Trades, optional): list of buys sells. Defaults to None.
        lot_method (LotOrdering, optional): Defaults to LotOrdering.TAX_OPTIMAL.
        lt_rate (float, optional): long term tax rate. Defaults to 0.15.
        st_rate (float, optional): short term tax rate. Defaults to 0.35.

    Returns:
        dict: inputs into scipy.optimize.linprog
    """
    obj = []  # coefficients of obj

    A_ub = pd.DataFrame()  # matrix where each row has an upper bound
    b_ub = []  # rhs of upper bounds for linear constraints

    A_eq = pd.DataFrame()  # matrix for linear equalities
    b_eq = []  # rhs linear constraints

    # for sell_index, sell_trade in trades.get_sells(asset).iterrows():
    for sell_index, sell_trade in enumerate(trades.get_sells(asset).itertuples(), 0):
        buys_before = trades.get_buys(asset, end_date=sell_trade.sell_date)
        obj += _get_objective_value(
            sell_trade, buys_before, lot_method, lt_rate=lt_rate, st_rate=st_rate
        )
        b_eq += [float(sell_trade.quantity)]
        column_names = [
            f"Buy_{buy_index}_Sell_{sell_index}"
            # for buy_index, _ in (buys_before).iterrows()
            for buy_index, _ in enumerate(buys_before.itertuples(), 0)
        ]
        A_eq.loc[sell_index, column_names] = 1.0

    # for buy_index, buy_trade in (trades.get_buys(asset)).iterrows():
    for buy_index, buy_trade in enumerate(trades.get_buys(asset).itertuples(), 0):
        b_ub.append(buy_trade.quantity)
        column_names = [
            f"Buy_{buy_index}_Sell_{sell_index}"
            # for sell_index, sell_trade in (trades.get_sells(asset)).iterrows()
            for sell_index, sell_trade in enumerate(trades.get_sells(asset).itertuples(), 0)
            if buy_trade.buy_date < sell_trade.sell_date
        ]
        A_ub.loc[buy_index, column_names] = 1.0
    for item in column_names:
        A_eq = A_eq.dropna(how="all").fillna(0.0)
    A_eq = A_eq.dropna(how="all").fillna(0.0)
    A_ub = A_ub.dropna(how="all").fillna(0.0)
    b_ub = b_ub[: len(A_ub)]

    if(method == "CVXOPT"):
        b_ub = b_ub + len(A_eq.columns)*[0.0]
        df1 = pd.DataFrame(np.diag(-np.ones(len(A_eq.columns))),
                           columns=A_eq.columns)
        A_ub = A_ub.append(df1)
        bounds = None
    else:
        bounds = [(0, None) for i in obj]
    return {
        "obj": obj,
        "A_ub": A_ub[A_eq.columns],
        "b_ub": b_ub,
        "A_eq": A_eq,
        "b_eq": b_eq,
        "bounds": bounds,
    }


def _get_objective_value(
    sell_trade: Trades,
    buys_before: Trades,
    lot_method: LotOrdering,
    lt_rate=0.15,
    st_rate=0.35,
) -> list:
    """Returns list of objective function coefficients

    Args:
        sell_trade (Trades): Single row of sell trade consisiting of price/quantity/buy_date, etc...
        buys_before (Trades): All Buys before single sell
        lot_method (LotOrdering): Accounting method
        lt_rate (float, optional): long term capital gains rate. Defaults to 0.15.
        st_rate (float, optional): short term capital gains rate. Defaults to 0.35.

    Returns:
        list: Objective coefficients
    """

    if lot_method == LotOrdering.FIFO:
        obj = range(sell_trade.Index*len(buys_before),
                    sell_trade.Index*len(buys_before)+len(buys_before), 1)
        obj = [1.0*i for i in obj]
    elif lot_method == LotOrdering.HIFO:
        obj = list(-buys_before.buy_price)
    elif lot_method == LotOrdering.LIFO:
        obj = range(0, -len(buys_before), -1)
        obj = [1.0*i for i in obj]
    elif lot_method == LotOrdering.TAX_OPTIMAL:
        obj = [
            (sell_trade.sell_price - i.buy_price) * lt_rate
            if (sell_trade.sell_date - i.buy_date) > datetime.timedelta(days=365)
            else (sell_trade.sell_price - i.buy_price) * st_rate
            for _, i in buys_before.iterrows()
        ]
    return obj


def allocate_asset_decision_variables(
    asset: str, trades: Trades, solution: dict, column_names: list
) -> Tuple[Holdings, Transactions]:
    """Create Holdings and Transactions from solving optimization

    Args:
        asset (str): Asset to allocate
        trades (Trades): Previous set of trades
        solution (dict): scipy.optimize.linprog solution
        column_names (list): Names of decision variables

    Returns:
        Tuple[Holdings, Transactions]: Gives single set of Holdings/Transactions for a given asset
    """
    buys = {i: pd.Series(j._asdict()) for i, j in enumerate(
        trades.get_buys(asset).itertuples(), 0)}
    sells = {i: pd.Series(j._asdict()) for i, j in enumerate(
        trades.get_sells(asset).itertuples(), 0)}
    new_transactions = []

    for index, variable in enumerate(column_names):

        # Each column is named Buy_{i}_Sell_{j}. Extract i,j:
        _, buy, _, sell = variable.split("_")
        buy = buys[int(buy)]
        sell = sells[int(sell)]

        # If this is a trade > $1, we record it
        #print(solution[index] * sell.sell_price)
        if not math.isclose(solution[index] * sell.sell_price, 0.0, abs_tol=1.0):
            buy.quantity = buy.quantity - solution[index]
            new_transactions.append(
                {
                    "asset": buy.asset,
                    "buy_date": buy.buy_date,
                    "buy_price": buy.buy_price,
                    "sell_date": sell.sell_date,
                    "sell_price": sell.sell_price,
                    "quantity": solution[index],
                    "realized_gains": solution[index]
                    * (sell.sell_price - buy.buy_price),
                }
            )
    transactions = Transactions.from_list(new_transactions)

    new_holdings = []

    # Whatever is left in our portfolio > $1, we record it
    for _, remaining_buys in buys.items():
        if not math.isclose(
            remaining_buys.quantity * remaining_buys.buy_price, 0.0, abs_tol=1.0
        ):
            new_holdings.append(
                {
                    "buy_price": remaining_buys.buy_price,
                    "quantity": remaining_buys.quantity,
                    "buy_date": remaining_buys.buy_date,
                    "asset": remaining_buys.asset,
                }
            )
    holdings = Holdings.from_list(new_holdings)
    return holdings, transactions


if __name__ == "__main__":
    from pathlib import Path

    #sample_data = Path(Path.cwd(), "sample_data", "sample_trades1.csv")
    sample_data = Path(Path.cwd(), "sample_data", "sample_trades2.csv")

    df = Trades.from_csv(str(sample_data))

    df = Trades(df.get_trades("BTC"))
    df = df.preprocess_trades()

    problem = setup_linear_optimization("BTC",
                                        df,
                                        lot_method=LotOrdering.FIFO,
                                        method="CVXOPT")
    print(problem)

    holds, transactions = run_trade_allocator(df,
                                              lot_ordering=LotOrdering.HIFO)
    print(holds)
    print(transactions)

    holds, transactions = run_trade_allocator(df,
                                              lot_ordering=LotOrdering.FIFO)
    print(holds)
    print(transactions)

    holds, transactions = run_trade_allocator(df,
                                              lot_ordering=LotOrdering.LIFO)

    print(holds)
    print(transactions)
