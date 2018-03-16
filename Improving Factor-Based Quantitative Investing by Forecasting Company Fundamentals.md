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

Public stock markets provide a venue for buying and selling shares, which representfractional ownership of individual



