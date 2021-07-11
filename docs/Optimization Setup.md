
## The Problem Setup
The problem of allocating sells to the corresponding lots can easily be transformed into an optimization problem based on a given accounting method. Let's define:

`Buy_i = The ith buy (or when you acquired an asset)`

`Sell_j = The jth sell (or when you transferred the asset)`

Just note that for each `Buy_i` and `Sell_j`, you have information corresponding to:
- Asset name (BTC, ETH, LTC)
- Quantity (number of coins bought/sold)
- Price (buy or sell price)
- Date (when the transaction occured)

The decision variables will be the amount we're allocating to each Buy trade from a future Sell:

`T_ij = the amount you are netting to Buy_i from Sell_j`

Given an ordered list of trades:

`Buy_0, Buy_1, Sell_0, Buy_2, Sell_1, Sell_2`

we can see what objectives and constraints are needed when arranged into a table:

| Buy_0 | Buy_1 | Buy_2 | ... | Buy_i |            |
| ----- | ----- | ----- | --- | ----- | ---------- |
| T_00  | T_10  | ---   | --- | ---   | **Sell_0** |
| T_01  | T_11  | T_21  | --- | ---   | **Sell_1** |
| T_02  | T_12  | T_22  | --- | ---   | **Sell_2** |
| ---   | ---   | ---   | --- | ---   | ...        |
| T_0j  | T_1j  | T_2j  | --- | T_ij  | **Sell_j** |

## Objective
The objective is given by:

`Min F(x) = c_ij * T_ij`

Where the coefficient `c_ij` is determined by the accounting method:

1. FIFO - Assign higher weights to the earliest Buys
2. HIFO - Use the cost basis for every Buy
3. LIFO - Assign higher weights the most recent Buys
4. TAX_OPTIMAL - Use the cost basis combined with short/long term tax rates

This makes the objective coefficient `c_ij` defined by:

    c = [] # WE ARE MINIMIZING THIS FUNCTION
    for Sell_j in All_Sells:
        Buys_Before = BUYS PRIOR TO Sell_j ORDERED FROM OLDEST TO MOST RECENT
        if(FIFO):
            c += [1, 2, ... , Size(Buys_Before)]
        else if(HIFO):
            c += [-Price(Buy_i) for Buy_i in Buys_Before]
        else if(LIFO):
            c += [-1, -2, ... , -Size(Buys_Before)]
        else if(TAX_OPTIMAL):
            c += [
                (Price(Sell_j) - Price(Buy_i)) * Long_Term_Tax_Rate
                if (Date(Sell_j) - Date(Buy_i)) > 1 YEAR
                else (Price(Sell_j) - Price(Buy_i)) * Short_Term_Tax_Rate 
                for Buy_i in Buys_Before
            ]

## Constraints

We can again look at a table to see what constraints are needed:

| Buy_0 | Buy_1 | Buy_2 | ... | Buy_i |            |
| ----- | ----- | ----- | --- | ----- | ---------- |
| T_00  | T_10  | ---   | --- | ---   | **Sell_0** |
| T_01  | T_11  | T_21  | --- | ---   | **Sell_1** |
| T_02  | T_12  | T_22  | --- | ---   | **Sell_2** |
| ---   | ---   | ---   | --- | ---   | ...        |
| T_0j  | T_1j  | T_2j  | --- | T_ij  | **Sell_j** |

For a given Sell (each ROW) we need to make sure the full amount is allocated to some previous buy. This leads to the equality constraint:

`(1) [Sum(Previous Buy's) T_ij = Sell_j ]      For Each j`

And then since one Buy can be netted across multiple Sells, we need to ensure we do not allocate more than the Buy quantity we have. This corresponds to a constraint for each COLUMN in the above table:

`(2) [Sum(All j) T_ij < Buy_i]  For Each i`

Lastly, we need simple asset bounds on our variables:

`(3) T_ij > 0 For All i,j`

The above objectives and constraints can be input into an LP Solver from [scipy.optimize.linprog](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html) to find the optimal way to allocate your trades.

## NOTE THIS IS NOT FINANCIAL ADVICE... PLEASE CONSULT A CPA OR FINANCIAL ADVISOR FOR YOUR OWN TAX SITUATION.
