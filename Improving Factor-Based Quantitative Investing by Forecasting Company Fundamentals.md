# Improving Factor-Based Quantitative Investing by Forecasting Company Fundamentals

> 기업 fundamental을 예측하여 factor 기반 양적 투자 개선
>
> Jon Alberg, Zachary C. Lipton(2017)  

* Abxtract

On a periodic basis, publicly traded companies are required to report *fundamentals*: financial data such as revenue, operating income, debt, among others.  

상장 회사는 정기적으로 매출, 영업 수입, 부채와 같은 재무 data를 보고해야 한다.

These data points provide some insight into the financial health of a company.  

이러한 data point들은 회사의 재무 건전성에 대한 통찰력을 제공한다.  

Academic research has identified some factors, i.e. computed features of the reported data, that are known through retrospective analysis to outperform the market average.  

학술 연구는 후향적 분석을 통해 시장 평균보다 우수한 실적을 올리기 위한 몇 가지 요인(즉, 그 보고된 data에서 계산된 features)을 확인했다.  

> 후향적 분석 : 현시점에서 과거의 기록을 대상으로 분석하는 것이다. 연구 대상에 독립 변수가 이미 발생한 후에 나타난 종속 변수를 대상으로 한 분석이다. 이미 독립 변수가 발생했기 때문에 연구자가 독립 변수를 조작할 수 없거나 연구 대상을 실험 조건에 따라 배치하기 어려운 경우에 사용된다. 사후설계분석(ex post facto design analysis)이라고도 한다. (ex. 폐암환자와 비폐암환자의 과거 흡연 습관 비교)

Two popular factors are the book value normalized by market capitalization (book-to-market) and the operating income normalized by the enterprise value (EBIT/EV). 

가장 큰 두 가지 factor는 시가총액(장부가)으로 표준화된 장부가액과 기업가치(EBIT/EV)다.

> EBIT : 이자비용과 법인세를 제외하기 전 회사의 총이익이다. 즉 당기순이익에 이자와 세금을 더한 금액을 말한다. 회사의 기본적인 이익창출능력을 확인하는 지표로 사용된다.
>
> EV : Enterprise Value의 약자로 기업의 총가치를 말한다. 기업을 매매한다고 가정했을 때 매수자가 매수시 지급해야 하는 금액.

**In this paper:** we first show through simulation that if we could (clarivoyantly) select stocks using factors calculated on *future* fundamentals (via oracle), then our portfolios would far outperform a standard factor approach.

이 논문에서 우리는 simulation을 통해 미래 fundamentals (via Oracle)에 대해 계산된 factors를 사용하여 주식을 선택한다면 우리의 portfolios는 표준 요소 접근법보다 훨씬 뛰어나다는 것을 알 수 있다.

Motivated by this analysis, we train deep neural networks to forecast future fundamentals based on a trailing 5-years window.

이 분석에서 동기를 얻어, 우리는 후행 5년 window를 기반으로 향후 fundamentals를 예측하기 위해 deep neural networks를 학습한다.

Quantitative analysis demonstrates a significant improvement in MSE over a naive strategy.
양적 분석은 naive 전략에 비해 MSE가 크게 향상되었음을 보여준다.

> MSE : 회귀 모형에서 data에 포함된 불확실성(uncertainty)은 적합 회귀선(fitted regression line, 추정 회귀식)으로부터 관측치가 얼마나 벗어나 있나를 의미하며 이것에 대한 측정은 $$(y_i - \hat{y}_i)$$이고 제곱 합을 오차변동(Error Sum of Squares, SSE) 혹은 오차 제곱합(자승합)이라 하며 적합 회귀식에 의해 설명되지 않는 변동에 해당된다. 이 오차 변동을 (n-p-1)으로 나눈 값을 MSE(Mean Square Error)라 하면 이는 오차의 분산에 대한 추정치로 사용한다.

Moreover, in retro spective anlysis using an industry-grade stock portfolio simulator (backtester), we show an improvement in compounded annual return to 17.1% (MLP) vs 14.4% for a standard factor model.
또한, 산업 등급 주식 portfolio simulator(backtester)를 사용한 후향적 분석(retro spective analysis)에서 standard factor model에 대해 복합 연간 수익률이 17.1%(MLP)대 14.4%로 향상되었음을 보여준다.

> compounded annual return : 일반적으로 백분율로 표시되며 복합 연평균 성장률(compound annual growth rate)은 일정 기간 동안의 원래 금액에 대한 일련의 손익의 누적 효과를 나타낸다.(주-성장률과 수익률을 같은 의미로 사용한 것으로 해석)

1. Introduction

Public stock markets provide a venue for buying and selling shares, which represent fractional ownership of individual companies.

공개 주식 시장은 개별 기업의 부분 소유권을 나타내는 주식을 매매하는 장소를 제공한다.

Prices fluctuate frequently, but the myriad drivers of price movements occur on multiple time scale.

가격은 자주 변동하지만, 가격 움직임의 무수한 원인은 여러 시간 scale에서 발생한다. 

In the short run, price movements might reflect the dynamics of order execution, and the behavior of high frequency traders.

단기적으로는, 가격 변동은 주문 실행의 동력이고 고빈도 거래자의 행동을 반영 할 수 있다.

On the scale of days, price fluctuation might be driven by the news cycle.

며칠 동안의 가격 변동은 뉴스 cycle에 의해 좌우될 수 있다.

Individual stocks may rise or fall on rumors or reports of sales numbers, product launches, etc.

개별주식은 소문, 판매 보고서의 숫자들, 신제품 출시 등으로 인해 상승하거나 하락할 수 있다.

In the long run, we expect a company's market value to reflect its financial performance, as captured in *fundamental data*, i.e., reported financial information such as income, revenue, assets, dividends, and debt.

장기적으로, 우리는 기업의 시장가치가 그것의 fundamental data에 기록된 재무 성과(즉, 수입, 이익, 자산, 배당금, 부채와 같은 보고된 재무정보)를 반영할 것으로 기대한다.

In other words, shares reflect ownership in a company thus share prices should ultimately move towards the company's *intrinsic value*, the cumulative discounted cash flows associated with that ownership.

다시 말해서, 주식은 회사의 소유권을 반영하므로 주가는 궁극적으로 회사의 내재 가치(소유권과 관련된 누적 할인 현금 흐름)로 이동한다.

One popular strategy called *value investing* is predicated on the idea that long-run prices reflect this intrinsic value and that the best features for predicting *long-term* intrinsic value are the *currently* available fundamental data.

가치 투자라는 유명한 전략은 장기 가격이  본질적인 가치를 반영하고 장기 내재가치를 예측하기 위한 최상의 features은 현재 사용가능한 fundamental data라는 idea에 근거한다.

----

In a typical quantitative (systematic) investing strategy, we sort the set of available stocks according to some *factor* and construct investment portfolios comprised of those stocks which score highest.

Many quantitative investors engineer *value factors* by taking fundamental data in a ratio to stock's price, such as EBIT/EV or book-to-market.

Stock with high value factor ratios are called *value* stocks and those with low ratios are called *growth* stocks.

Academic researchers have demonstrated empirically that portfolios of stocks which overweight value stocks have significantly outperformed portfolios that overweight growth stocks over the long run [12, 7].



**In this paper**, we propose an investment strategy that constructs portfolios of stocks today based on *predicted future fundamentals*. Recall that value factor should identify companies ahta are inexpensively priced with respect to current company fundamentals such as earning or boo-value. We suggest that the long-term success of an investment should depend on the how well-priced the stock *currently is* with respect to its *future fundamentals*. We run simulations with a *clairvoyant model* that can access future financial reports (by oracle). In Figure 1, we demonstrate that for the 2000-2014 time period, a clairvoyant model applying the EBIT/EV factor with 12-month clairvoyant fundamentals, if possible, would achieve a 44% compound annualized return.

 

