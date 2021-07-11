## Introduction

According to the IRS tax guidlines, [crypto currencies are treated as property](https://www.irs.gov/individuals/international-taxpayers/frequently-asked-questions-on-virtual-currency-transactions). When you buy/sell/trade these units, you must recognize and report capital gains or losses coming from the sale of such assets:

> In 2014, the IRS issued Notice 2014-21, 2014-16 I.R.B. 938 PDF, explaining that virtual currency is treated as property for Federal income tax purposes and providing examples of how longstanding tax principles applicable to transactions involving property apply to virtual currency.

...

> You may choose which units of virtual currency are deemed to be sold, exchanged, or otherwise disposed of if you can specifically identify which unit or units of virtual currency are involved in the transaction and substantiate your basis in those units.


 So how do you determine the cost basis for a sale if you have multiple buys prior?  There are a number of common accounting methods:

1. FIFO (First In First Out)
2. HIFO (Highest In First Out)
3. LIFO (Last In First Out)
4. TAX_OPTIMAL (This is defined as the solution that minimizes tax liability using long and short term tax rates)

### A Simple Example
Take the below as an example set of trades:
>Buy 1 BTC on January 1st, 2020 for $15,000 (Lot A)
>
>Buy 1 BTC on January 2st, 2020 for $20,000 (Lot B)
>
>Buy 1 BTC on June 1st, 2020 for $17,500 (Lot C)
>
>Buy 1 BTC on September 15th, 2020 for $12,500 (Lot D)


>Sell 1 BTC on January 31st, 2021 for $15,000

Considering the last transaction is a SELL, you would need to specifiy from which lot you sold from. Using the different accounting methods you would report the following to the IRS:

1. FIFO - use Lot A realizing No GAINS/LOSSES
2. HIFO - use Lot B realizing a Long Term LOSS of $5,000
3. LIFO - use Lot D realizing a Short Term GAIN of $2,500
4. TAX_OPTIMAL - In this case we need to compute the Tax Liability in each scenario. Assuming your Long/Short Term Tax bracket is 15% / 35%:
   - Lot A: $0 Tax Liability     (15% * 0)
   - Lot B: $750 Tax Liability (15% * -5,000)
   - **Lot C: $-875 Tax Liability (35% * -2,500)**
   - Lot D: $875 Tax Liability (35% * 2,500)

TAX_OPTIMAL would then choose Lot C and you'd be able to deduct $875 from your tax bill.**\***


**\***One caveat to the above analysis is that TAX_OPTIMAL needs to be run in an iterative fashion at the end of each tax year. I.E. take the allocations and holdings from one tax year, add a new set of trades for the next year, and re-optimize over that data keeping the prior decision variables constant.

If not then there are cases that when minimizing tax liability, it's advantegous for an asset to change lot assignments year-after-year to reduce the overall tax burden. This is something the IRS doesn't allow - once you declare your gains/losses you need to stick to that! 

## NOTE THIS IS NOT FINANCIAL ADVICE... PLEASE CONSULT A CPA OR FINANCIAL ADVISOR FOR YOUR OWN TAX SITUATION.
