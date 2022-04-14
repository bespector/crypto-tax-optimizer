# Copyright (c) 2022 Brian Spector
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import datetime
from enum import Enum
from typing import Union

import numpy as np
import pandas as pd


class Side(Enum):
    SELL = "SELL"
    BUY = "BUY"

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.value)

    def __str__(self):
        return self.value


class LotOrdering(Enum):
    FIFO = "FIFO"
    HIFO = "HIFO"
    LIFO = "LIFO"
    TAX_OPTIMAL = "TAX_OPTIMAL"

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.value)

    def __str__(self):
        return self.value


class PortfolioBase(pd.DataFrame):
    def __init__(self, data, req_columns=[], opt_columns=[]):
        if data is None or len(data) < 1:
            data = pd.DataFrame(columns=req_columns + opt_columns)
        else:
            data = data.reset_index().drop("index", axis=1)
            for column in req_columns:
                if column not in data.columns:
                    raise Exception("Missing required column: " + column)
            for column in opt_columns:
                if column not in data.columns:
                    data[column] = None
            for column in req_columns + opt_columns:
                if "date" in column:
                    data[column] = pd.to_datetime(data[column])

        super().__init__(data)

    def __add__(self, other):
        cls = self.__class__
        return cls(pd.concat([self, other]))

    @classmethod
    def from_dict(cls, data):
        return cls(pd.Series(data).to_frame().T)

    @classmethod
    def from_list(cls, data):
        return cls(pd.DataFrame(data))

    @classmethod
    def from_csv(cls, filepath):
        df = pd.read_csv(filepath)
        return cls(df)

    def get_asset_names(self):
        return list(self.asset.unique())

    def add_item(self, **args):
        cls = self.__class__
        temp = cls.from_dict(args)
        return self + temp

    def add_list(self, other_list_dict=[]):
        temp = pd.DataFrame(other_list_dict)
        return self + temp


class Trades(PortfolioBase):
    def __init__(self, trades=None):
        req_columns = ["asset", "quantity"]
        opt_columns = ["buy_price", "buy_date", "sell_price", "sell_date"]
        super().__init__(trades, req_columns, opt_columns)

    def preprocess_trades(self):
        """Combine trades at same timestamp."""
        def wm(x): return np.average(x, weights=self.loc[x.index, "quantity"])

        new_transactions = pd.DataFrame()
        buy_transactions = self.groupby([self.asset, self.buy_date.dt.date],
                                        as_index=False).agg(quantity=("quantity", 'sum'),
                                                            buy_price=(
                                                                "buy_price", wm),
                                                            buy_date=("buy_date", "max"))
        sell_transactions = self.groupby([self.asset, self.sell_date.dt.date],
                                         as_index=False).agg(quantity=("quantity", 'sum'),
                                                             sell_price=(
                                                                 "sell_price", wm),
                                                             sell_date=("sell_date", "max"))
        #df = self.sort_values("buy_date")
        new_transactions = pd.concat([buy_transactions, sell_transactions])

        return Trades(new_transactions)

    def get_sells(
        self,
        assets: Union[str, list] = None,
        start_date: datetime.datetime = datetime.datetime(1990, 1, 1),
        end_date: datetime.datetime = datetime.datetime(2100, 1, 1),
    ):
        return self.get_trades(assets, Side.SELL, start_date, end_date)

    def get_buys(
        self,
        assets: Union[str, list] = None,
        start_date: datetime.date = datetime.datetime(1990, 1, 1),
        end_date: datetime.date = datetime.datetime(2100, 1, 1),
    ):
        return self.get_trades(assets, Side.BUY, start_date, end_date)

    def get_trades(
        self,
        assets: Union[str, list] = None,
        trade_type: str = None,
        start_date: datetime.datetime = datetime.datetime(1990, 1, 1),
        end_date: datetime.datetime = datetime.datetime(2100, 1, 1),
    ):
        if isinstance(assets, str):
            assets = [assets]
        elif assets is None:
            assets = self.get_asset_names()

        buy_date_filter = (self.buy_date <= end_date) & (
            self.buy_date >= start_date)
        sell_date_filter = (self.sell_date <= end_date) & (
            self.sell_date >= start_date)

        if trade_type == Side.BUY:
            date_filter = buy_date_filter
        elif trade_type == Side.SELL:
            date_filter = sell_date_filter
        else:
            date_filter = (buy_date_filter) | (sell_date_filter)

        asset_filter = self.asset.isin(assets)
        return self[date_filter & asset_filter]


class Holdings(PortfolioBase):
    def __init__(self, holdings=None):
        req_columns = ["asset", "quantity", "buy_date", "buy_price"]
        opt_columns = ["current_price", "unrealized_gains"]
        super().__init__(holdings, req_columns, opt_columns)

    def to_trades(self) -> Trades:
        return Trades(self[["asset", "quantity", "buy_price", "buy_date"]])


class Transactions(PortfolioBase):
    def __init__(self, transactions=None):
        req_columns = [
            "asset",
            "quantity",
            "buy_date",
            "buy_price",
            "sell_date",
            "sell_price",
        ]
        opt_columns = ["realized_gains"]
        super().__init__(transactions, req_columns, opt_columns)

    def get_long_term_assets(self):
        return (self.sell_date - self.buy_date) > datetime.timedelta(days=365)

    def get_lt_tax_gains(self):
        lt_tax_gains = (
            self[(self.get_long_term_assets()) & (self.realized_gains > 0)]
        ).realized_gains.sum()
        return lt_tax_gains

    def get_st_tax_gains(self):
        st_tax_gains = (
            self[~(self.get_long_term_assets()) & (self.realized_gains > 0)]
        ).realized_gains.sum()
        return st_tax_gains

    def get_lt_tax_losses(self):
        lt_tax_losses = (
            self[(self.get_long_term_assets()) & (self.realized_gains < 0)]
        ).realized_gains.sum()
        return lt_tax_losses

    def get_st_tax_losses(self):
        st_tax_losses = (
            self[~(self.get_long_term_assets()) & (self.realized_gains < 0)]
        ).realized_gains.sum()
        return st_tax_losses

    def get_st_tax_liability(self, st_rate):
        st_tax_liability = self.get_st_tax_losses() + self.get_st_tax_gains()
        return st_tax_liability * st_rate

    def get_lt_tax_liability(self, lt_rate):
        lt_tax_liability = self.get_lt_tax_losses() + self.get_lt_tax_gains()
        return lt_tax_liability * lt_rate

    def get_tax_liability(self, st_rate, lt_rate):
        return self.get_st_tax_liability(st_rate) + self.get_lt_tax_liability(lt_rate)

    def to_trades(self) -> Trades:
        df = pd.DataFrame()
        for i, j in self.iterrows():
            new_row = pd.DataFrame(
                [
                    [j.asset, j.quantity, j.buy_price, j.buy_date, np.nan, np.nan],
                    [j.asset, j.quantity, np.nan, np.nan,
                        j.sell_price, j.sell_date],
                ],
                columns=[
                    "asset",
                    "quantity",
                    "buy_price",
                    "buy_date",
                    "sell_price",
                    "sell_date",
                ],
            )
            df = pd.concat([df, new_row])
        return Trades(df)


if __name__ == "__main__":
    from pathlib import Path

    sample_data2 = Path(Path.cwd(), "sample_data", "sample_trades2.csv")

    df = Trades.from_csv(str(sample_data2))
    df = df.preprocess_trades()
