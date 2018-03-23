# Improving Factor-Based Quantitative Investing by Forecasting Company Fundamentals

> 기업 fundamental을 예측하여 factor 기반 양적 투자 개선
>
> Jon Alberg, Zachary C. Lipton(2017)  

### Abstract

On a periodic basis, publicly traded companies are required to report *fundamentals*: financial data such as revenue, operating income, debt, among others.  These data points provide some insight into the financial health of a company.  Academic research has identified some factors, i.e. computed features of the reported data, that are known through retrospective analysis to outperform the market average.  Two popular factors are the book value normalized by market capitalization (book-to-market) and the operating income normalized by the enterprise value (EBIT/EV). 

**In this paper:** we first show through simulation that if we could (clarivoyantly) select stocks using factors calculated on *future* fundamentals (via oracle), then our portfolios would far outperform a standard factor approach. Motivated by this analysis, we train deep neural networks to forecast future fundamentals based on a trailing 5-years window. Quantitative analysis demonstrates a significant improvement in MSE over a naive strategy. Moreover, in retro spective anlysis using an industry-grade stock portfolio simulator (backtester), we show an improvement in compounded annual return to 17.1% (MLP) vs 14.4% for a standard factor model.

상장 회사는 정기적으로 매출, 영업 수입, 부채와 같은 재무 data를 보고해야 한다. 이러한 data point들은 회사의 재무 건전성에 대한 통찰력을 제공한다.  학술 연구는 후향적 분석[^1]을 통해 시장 평균보다 우수한 실적을 올리기 위한 몇 가지 요인(즉, 그 보고된 data에서 계산된 features)을 확인했다.  가장 큰 두 가지 factor는 시가총액으로 표준화된 장부가액(book-to-market)[^2]과 기업가치(EBIT[^3]/EV[^4])다. 이 논문에서는 먼저 simulation을 통해 미래의 fundamentals에 대해 계산된 factors를 사용하여 주식을 선택하면(via Oracle), portfolios가 표준 요소 접근법보다 훨씬 우수하다는 것을 보여준다. 이 분석에서 동기를 얻어, 후행 5년 window를 기반으로 향후 fundamentals를 예측하기 위해 deep neural networks를 학습한다. 양적 분석은 naive 전략에 비해 MSE[^5]가 크게 향상되었음을 보여준다. 또한, 산업 등급 주식 portfolio simulator(backtester)를 사용한 후향적 분석(retro spective analysis)에서 standard factor model에 대해 복합 연간 수익률[^6]이 17.1%(MLP)대 14.4%로 향상되었음을 보여준다.

[^1]: 현시점에서 과거의 기록을 대상으로 분석하는 것이다. 연구 대상에 독립 변수가 이미 발생한 후에 나타난 종속 변수를 대상으로 한 분석이다. 이미 독립 변수가 발생했기 때문에 연구자가 독립 변수를 조작할 수 없거나 연구 대상을 실험 조건에 따라 배치하기 어려운 경우에 사용된다. 사후설계분석(ex post facto design analysis)이라고도 한다. (ex. 폐암환자와 비폐암환자의 과거 흡연 습관 비교)

[^2]: 본문의 book-to-market을 Book-to-market ratio로 이해함. Book-to-market ratio = book value of firm / market value of firm
[^3]: EBIT, Earnings Befoer Interest and Tax : 이자비용과 법인세를 제외하기 전 회사의 총이익이다. 즉 당기순이익에 이자와 세금을 더한 금액을 말한다. 회사의 기본적인 이익창출능력을 확인하는 지표로 사용된다.
[^4]: EV, Enterprise Value : 기업의 총가치를 말한다. 기업을 매매한다고 가정했을 때 매수자가 매수시 지급해야 하는 금액.

[^5]: MSE : 회귀 모형에서 data에 포함된 불확실성(uncertainty)은 적합 회귀선(fitted regression line, 추정 회귀식)으로부터 관측치가 얼마나 벗어나 있나를 의미하며 이것에 대한 측정은 $$(y_i - \hat{y}_i)$$이고 제곱 합을 오차변동(Error Sum of Squares, SSE) 혹은 오차 제곱합(자승합)이라 하며 적합 회귀식에 의해 설명되지 않는 변동에 해당된다. 이 오차 변동을 (n-p-1)으로 나눈 값을 MSE(Mean Square Error)라 하면 이는 오차의 분산에 대한 추정치로 사용한다.

[^6]: compounded annual return : 일반적으로 백분율로 표시되며 복합 연평균 성장률(compound annual growth rate)은 일정 기간 동안의 원래 금액에 대한 일련의 손익의 누적 효과를 나타낸다.(주-성장률과 수익률을 같은 의미로 사용한 것으로 해석)

### 1. Introduction

Public stock markets provide a venue for buying and selling shares, which represent fractional ownership of individual companies. Prices fluctuate frequently, but the myriad drivers of price movements occur on multiple time scale. In the short run, price movements might reflect the dynamics of order execution, and the behavior of high frequency traders. On the scale of days, price fluctuation might be driven by the news cycle. Individual stocks may rise or fall on rumors or reports of sales numbers, product launches, etc. In the long run, we expect a company's market value to reflect its financial performance, as captured in *fundamental data*, i.e., reported financial information such as income, revenue, assets, dividends, and debt. In other words, shares reflect ownership in a company thus share prices should ultimately move towards the company's *intrinsic value*, the cumulative discounted cash flows associated with that ownership. One popular strategy called *value investing* is predicated on the idea that long-run prices reflect this intrinsic value and that the best features for predicting *long-term* intrinsic value are the *currently* available fundamental data.

공개 주식 시장은 개별 기업의 부분 소유권을 나타내는 주식을 매매하는 장소를 제공한다.가격은 자주 변동하지만, 가격 움직임의 무수한 원인은 여러 시간 scale에서 발생한다. 단기적으로는, 가격 변동은 주문 실행의 동력이고 고빈도 거래자의 행동을 반영 할 수 있다. 며칠 동안의 가격 변동은 뉴스 cycle에 의해 좌우될 수 있다. 개별주식은 소문, 판매 보고서의 숫자들, 신제품 출시 등으로 인해 상승하거나 하락할 수 있다. 장기적으로, 우리는 기업의 시장가치가 그것의 fundamental data에 기록된 재무 성과(즉, 수입, 이익, 자산, 배당금, 부채와 같은 보고된 재무정보)를 반영할 것으로 기대한다. 다시 말해서, 주식은 회사의 소유권을 반영하므로 주가는 궁극적으로 회사의 내재 가치(소유권과 관련된 누적 할인 현금 흐름)로 이동한다. 가치 투자라는 유명한 전략은 장기 가격이  본질적인 가치를 반영하고 장기 내재 가치를 예측하기 위한 최상의 features은 현재 사용가능한 fundamental data라는 idea에 근거한다.

In a typical quantitative (systematic) investing strategy, we sort the set of available stocks according to some *factor* and construct investment portfolios comprised of those stocks which score highest. Many quantitative investors engineer *value factors* by taking fundamental data in a ratio to stock's price, such as EBIT/EV or book-to-market. Stock with high value factor ratios are called *value* stocks and those with low ratios are called *growth* stocks. Academic researchers have demonstrated empirically that portfolios of stocks which overweight value stocks have significantly outperformed portfolios that overweight growth stocks over the long run [13].

전형적인 양적 (체계적) 투자 전략에서 우리는 어떤 factor에 따라 사용 가능한 주식 세트를 분류하고 가장 높은 점수를 가진 주식으로 구성된 투자 포트폴리오를 구성한다. 많은 양적 투자자는 EBIT/EV 또는 장부가 비율(book-to-market)과 같은 주식 가격 대비 비율로 fundamental data를 취함으로써 가치 factors를 설계한다. 높은 가치 요소 비율을 가진 주식을 가치주라고 하고 낮은 비율을 갖는 것을 성장주라고 한다. 학술 연구자들은 성장주의 비중을 확대한 portfolios보다 가치주의 비중을 확대한 portfolios가 장기적으로 현저하게 성과가 높다는 것을 경험적으로 입증했다. [12, 7]

**In this paper**, we propose an investment strategy that constructs portfolios of stocks today based on *predicted future fundamentals*. Recall that value factor should identify companies that are inexpensively priced with respect to current company fundamentals such as earning or book-value. We suggest that the long-term success of an investment should depend on the how well-priced the stock *currently is* with respect to its *future fundamentals*. We run simulations with a *clairvoyant model* that can access future financial reports (by oracle). In Figure 1, we demonstrate that for the 2000-2014 time period, a clairvoyant model applying the EBIT/EV factor with 12-month clairvoyant fundamentals, if possible, would achieve a 44% compound annualized return.

본고에서, 우리는 예측한 미래의 fundamentals를 기반으로 오늘의 주식 portfolios를 구성하는 투자 전략을 제안한다. 가치 factor는 수입이나 장부가치와 같은 현재의 회사의 fundamentals와 관련하여 저렴한 회사를 식별해야 한다. 우리는 장기적인 투자 성공이 미래의 fundamentals과 관련하여 주식 가격이 얼마나 좋은지에 달려있다고 제안한다. 우리는 미래의 재무 보고서에 접근할 수 있는 날카로운 통찰력이 있는 model을 사용하여 simulations을 실행한다.(oracle을 사용) 그림 1은 2000-2014년의 기간 동안 가능한 12개월의 clarivoyant fundamentals의 EBIT/EV factor를 적용한 clairvoyant model이 44%의 복합 연간 수익률을 달성할 수 있음을 보여준다.

Figure 1: Annualized return for various factor models for different degrees of clairvoyance.

그림 1: 다양한 clairvoyance 정도에 대한 다양한 factor models에 대한 연간 수익률.

![Figure 1](.\image/figure1.svg)

Motivated by the performance of factors applied to clairvoyant future data, we propose to predict future fundamental data based on trailing time series of 5 years of fundamental data. We denote these algorithms as Lookahead Factor Models (LFMs). Both multilayer perceptrons (MLPs) and recurrent neural networks (RNNs) can make informative predictions, achieving out-of-sample MSE of .47, vs .53 for linear regression and .62 for a naive perdictor. Simulations demonstrate that investing with LFMs based on the predicted factors yields a compound annualized return (CAR) of 17.7%, vs 14.4% for a normal factor model and a Sharpe ratio .68 vs .55.

clairvoyance 미래 data에 적용된 factors의 성능에 의한 동기부여를 통해 5년간의 fundamental data의 후행 시계열을 기반으로 미래의 fundamental data를 예측할 것을 제안한다. 이러한 algorithms을 Lookahead Factor Models (LFMs)라고 한다. Multilayer Perceptron (MLP)과 RNN(Recurrent Neural Network) 모두 표본 MSE가 0.47, 선형 회귀 분석에서는 0.53, naive predictor는 0.62를 얻을 수 있다. simulation 결과 예측된 factor를 기반으로 LFM에 투자하면 17.7%의 복합 연평균 수익률(CAR)이 얻어지며 normal factor model의 경우 14.4%, Sharpe ratio[^7]는 LFM은 0.68 normal factor model은 0.55이다. 

[^7]: Sharpe ratio 평균 수익에서 무위험 수익(risk-free return)을 뺀 값을 투자 수익의 표준 편차로 나눈 값

**Related Work**   Deep neural networks models have proven powerful for tasks as diverse as language translations [14, 1], video captioning [11, 16], video recognition [6, 15], and time series modeling [9, 10, 3]. A number of recent paper consider deep learning approaches to predicting stock market performance. [2] evaluates MLPs for stock market prediction. [5] uses recursive tensor nets to extract events from CNN news reports and uses convolutional neural nets to predict future performance from a sequence of extracted events. Several preprinted drafts consider deep learning for stock market prediction [4, 17, 8] however, in all cases, the empirical studies are limited to few stocks and short time periods.

Deep neural networks models은 언어 번역 [1], video captioning [11, 16], video 인식 [6, 15] 및 시계열 modeling[9, 10, 3]과 같은 다양한 작업에 대해 설득력 있는 것으로 입증되었다. 다수의 최근 논문은 주식 시장 성과 예측에 대해 deep learning approaches를 고려하고 있다. [2]는 주식 시장 예측을 위한 MLP를 평가한다. [5]는 recursive tensor nets을 사용하여 CNN news reports에서 events를 추출하고, 추출된 events의 sequence로부터 미래의 성능을 예측하기 위해 convolutional neural nets를 사용한다. 몇몇 preprinted 초안들은 주식 시장 예측에  deep learning을 고려하지만 [4, 17, 8], 대부분의 이런 경우에도 경험적 연구는 소수의 주식과 짧은 기간으로 한정된다.

---

### 2. Deep Learning for Forecasting Fundamentals

**Data**  In this research, we consider all stocks that were publicly traded on the NYSE, NASDAQ or AMEX exchanges for at least 12 consecutive months between betwen January, 1970 and September, 2017. From this list, we exclude non-US-based companies, financial sector companies, and any company with an inflation-adjusted market capitalization value below 100 million dollars. The fianl list contains 11,815 stocks. Our features consist of reported financial information as archived by the *compustat North America* and *Compustat Snapshot* databases. Because reported information arrive intermiittently throughout a financial period, we discretize the raw data to a monthly time step.

이 연구에서는 1970년 1월에서 2017년 9월 중 최소 12개월 동안 연속적으로 NYSE, NASDAQ 또는 AMEX 거래소에서 공개적으로 거래된 모든 주식을 고려한다. 이 목록에서 비 미국계 회사, 금융 부문 회사 및 시가총액이 1억 달러 미만인 모든 회사를 제외한다. 최종 목록에는 11,815개의 주식이 포함되어 있다. feature는 *compustat North America* 와 *Compustat Sanpshot*에 의해 보관된 databases에 있는 보고된 재무 정보로 이루어져있다. 보고된 정보는 재무 기간에 걸쳐 간헐적으로 도착하기 때문에 raw data를 매월 시간 단계로 이산화한다. 우리는 장기 예측에 관심이 있고 data에서 계절성을 평활화 하기 위해 매월마다 시간 frames 사이에 1년 지연한 입력자료를 공급하고 향후 12개월의 fundamentals을 예측한다. 

For eachstock and at each time step *t*, weconsider a total of 20 input features. We engineer16 features from the fundamentals as inputs to our models. Income statement features are cumulative *trailing twelve months*, denoted TTM, and balance sheet features are most recentquarter, denoted MRQ. First we consider These items include *revenue* (TTM); cost of goods sold (TTM);selling, general & and admin expense(TTM); earnings before interest and taxes or EBIT (TTM); net income(TTM); cash and cash equivalents (MRQ); receivables (MRQ); inventories (MRQ);other current assets (MRQ); property plant and equipment (MRQ); other assets(MRQ); debt in current liabilities (MRQ); accounts payable (MRQ); taxes payable(MRQ); other current liabilities (MRQ); total liabilities (MRQ).For all features, we deal with missing valuesby filling forward previously observed values, following the methods of [9]. Additionally we incorporate 4 *momentum features*, which indicate the price movement of the stock over the previous 1, 3, 6, and 9 months respectively. So that our modelpicks up on relative changes and doesn’t focus overly on trends in specifictime periods, we use the percentile among all stocks as a feature (vs absolute numbers).

각 주식 및 각 시간 단계 t에서 총 20개의 input features를 고려한다. 우리의 모델에 입력자료의 fundamentals로부터 16개 feature로 design and build한다. 손익계산서 features는 누적 후행 12개월로 TTM[^8]으로 표기하고, 대차대조표 features는 가장 최근의 분기로 MRQ[^9]로  표기한다. 먼저 우리는 이러한  항목에 수익(TTM); 판매된 상품 원가(TTM); 판매, 일반 및 관리 비용(TTM); 이자 및 세금 전 이익 또는 EBIT(TTM); 순소득(TTM); 현금 및 현금 등가물(MRQ); 미수금(MRQ);, 재고품(MRQ); 다른 현재 자산들(MRQ); 유형자산[^10] (MRQ); 기타자산(MRQ); 유동부채[^11] (MRQ); 채무[^12] (MRQ); 납부할 세금(MRQ); 기타 유동 부채(MRQ); 총 부채(MRQ)가 포함되어 있다고 간주한다. 모든 특징에 대해 우리는 [9]의 방법에 따라 이전에 관측된 값을 채워 누락된 값을 처리한다. 또한 이전 1, 3, 6 및 9개월에 걸쳐 주가 변동을 나타내는 4개의 momentum feature를 통합했다. 우리 model이 상대적인 변화를 예측하고 특정 시기의 추세에 집중하지 않도록 모든 주식의 백분율을 feature로 사용한다(절대 수치와 비교).

[^8]: Trailing Twelve Months(년)
[^9]: Most Recent Quater(최근 분기)
[^10]: Property, Plant and Equipment(PP&E) 유형자산으로 해석함. 회사의 자산, 제조 공장 및 장비를 구매한 모든 시점을 합산한 것
[^11]: 회사의 대차대조표에 나타나고 있는 단기부채, 미지급금, 미지급 부채, 기타부채가 포함된 회사의 채무
[^12]: Accounts Payable(AP) 채권자에게 단기 부채를 상환해야하는 기업의 의무를 나타내는 회계 항목

**Preprocessing** Each of the fundamental features exhibits a wide dynamic range over the universe of considered stocks. For example, Apple’s 52-week revenue as of September 2016 was 215 billion(USD). By contrast, National Presto, which manufactures pressure cookers, had a revenue 340million.  Intuitively,  these statistics are more meaningful when scaled by some measure of a company’s size. In preprocessing, we scale all fundamental features in given time series by the market capitalization in the last input time-step of the series. We scale all time steps by the same value so that the neural network can assess the relative change in fundamental values between time steps. While other notions of size are used, such as enterprise value and book equity, we choose to avoid these measure because they can, although rarely, take negative values. We then further scale the features so that they each individually have zero mean and unit standard deviation.

**전처리** 각 fundamental features는 고려된 주식의 모집단(의역)에 대해 넓은 동적 범위를 나타낸다. 예를 들어, 2016년 9월 현재 Apple의 52주 수익은 215억달러(USD)였다. 대조적으로 pressure cookers를 제조하는 National Presto의 수익은 3억 4천만달러(USD)였다. 직관적으로 이러한 통계는 회사의 규모에 따라 조정될 때 더 의미가 있다. 전처리 과정에서 우리는 주어진 시계열의 모든 fundamental features를 마지막 입력 시간 단계에서 시가 총액으로 확장한다. 신경망이 시간 간격 사이의 fundamental values의 상대적 변화를 평가할 수 있도록 모든 시간 간격을 동일한 값으로 조정한다. 기업 가치, book equity[^13]와 같이 다른 크기의 개념이 사용될 때, 그것들은 드물게 부정적인 값을 가질 수 있기 때문에 이러한 척도를 피하도록 선택한다. 그런 다음 각 feature의 평균 및 단위 표준편차가 0이 되도록 feature의 크기를 조절한다.

[^13]: 특정 기간이 끝날 때 재무 상태 보고서에 표시된 회사 주식의 가치

**Modeling** In our experiments, we divide the timeline in to an *in-sample* and *out-of-sample* period. Then, even within the in-sample period, we need to partition some of the data as a validation set.  In forecasting problems, we face distinct challenges in guarding against overfitting. First, we’re concerned with the traditional form of overfitting. Within the in-sample period, we do not want to over-fit to the finite observed training sample. To protect against and quantify this form of overfitting, we randomly hold out a validation set consisting of 30% of all stocks. On this *in-sample* validation set, we determine all hyperparameters, such as learning rate, model architecture, objective function weighting. We also use the in-sample validation set to determine early stopping criteria. When training, we record the validation set accuracy after each training epoch, saving the model for each best score achieved. When 25 epochs have passed without improving on the best validation set performance, we halt training and selecting the model with the best validation performance. In addition to generalizing well to the in-sample holdout set, we evaluate whether the model can predict the future *out-of-sample* stock performance. Since this research is focused on long-term investing, we chose large in-sample and out-of-sample periods of the years 1970-1999 and 2000-2017, respectively.

**Modeling** 우리의 실험에서 timeline을 sample 내 및 sample 밖 기간으로 나눈다. 그런 다음 sample 내 기간에서도 일부 data를 유효성 검사 집합으로 분할해야 한다. 예측 문제에서 우리는 overfitting을 막는데 뚜렷한 어려움에 직면한다. 첫째, 우리는 전통적인 형태의 overfitting에 관심이 있다. sample 내 기간 동안, 우리는 한정된 관찰된 training sample에 과도하게 맞추고 싶지 않다. 이러한 형태의 overfitting을 방지하고 계량화하기 위해 모든 주식의 30%로 구성된 유효성 검정 set를 임의로 선택한다. 이 sample 내 유효성 검증 세트에서 학습 속도, model architecture, 목적 함수 가중치와 같은 모든 hyperparameters를 결정한다. 또한 초기 sampling 기준을 결정하기 위해 sample 내 검증 set를 사용한다. training을 할 때, 각 training이 끝날 때마다 유효성 검증 정확도를 기록하고 달성된 각 최상의 점수에 대한 모델을 저장한다. 최상의 유효성 검증 set 성능을 개선하지 않고 25개 epoch를 통과한 경우, 최상의 최상의 검증 성능으로 그 모델을 선택하고 training을 중단한다. sample 내 holdout set을 잘 일반화하는 것 외에도, model이 미래의 sample 밖 주식 성능을 예측할 수 있는지 평가한다. 이 연구는 장기 투자에 중점을 두었기 때문에 1970-1999년과 2000-2017년의 대규모 sample 내외의 기간을 선택했다. 

In previous experiments, we tried predicting price movements directly with RNNs and while the RNN outperformed other approaches on the in-sample period, it failed to meaningfully out-perform a linear model (See results in Table 2a).

이전의 실험에서, 우리는 RNN을 사용하여 직접 가격 변동을 예측했는데, RNN이 sample 기간 내에 다른 접근법을 능가했지만 RNN이 선형 모델보다 유의적으로 성능이 좋지 않았다(표 2a의 결과 참조). 

Given only price data, RNN’s easily overfit the training data while failing to improve performance on in-sample validation. **One key benefit of our approach** is that by doing *multi-task learning*, predicting all 16 future fundamentals, we provide the model with considerable training signal and may thus be less susceptible to overfitting.

가격 data만 주어지면 RNN은 training data에 쉽게 overfitting되지만, sample 내 유효성 검사의 성능은 향상시키지 못한다. **우리의 접근 방식의 한 가지 주요 이점**은 *다중 작업 학습*을 수행하여 16가지의 모든 미래의 fundamentals를 예측함으로써 model에 상당한 training signal을 제공하고 따라서 overfitting에 덜 민감할 수 있다는 것이다.  

The price movement of stocks is extremely noisy [13] and so, suspecting that the relationships among fundamental data may have a larger signal to noise ratio than the relationship between fundamentals and price, we set up the problem thusly: For MLPs, at each month $t$, given features for 5 month spaced 1 year apart ($t - 48$, $t - 36$, $t - 24$, $t - 12$), predict the fundamental data at time $t + 12$. For RNNs, the setup is identical but with the small modification that for each input in the sequence, we predict the corresponding month lookahead data.

주식의 가격 움직임은 매우 noisy [13]하기 때문에 fundamental data 간의 관계가 fundamentals와 price 사이의 관계 보다 더 큰 signal to noise ratio[^14]을 가질 수 있다고 의심하기 때문에 우리는 다음과 같이 문제를 설정한다. MLP의 경우 매달 ($t - 48$, $t - 36$, $t - 24$, $t - 12$)의 fundamental data를 예측한다. RNN의 경우 설정은 동일하지만 작은 sequence의 각 입력에 대해 해당 달의 미리보기 data를 예측한다.

[^14]: 다른 신호(signal)의 강도(strength)와 간섭(interference)의 강도의 비율

We evaluated two classes of deep neural networks: MLPs and RNNs. For each of these, we tune hyperparameters on the in-sample period. We then evaluated the resulting model on the out-of-sample period. For both MLPs and RNNs, we consider architectures evaluated with 1, 2, and 4 layers with 64, 128, 256, 512 or 1024 nodes. We also evaluate the use of dropout both on the inputs and between hidden layers. For MLPs we use ReLU activations and apply batch normalization between layers. For RNNs we test both GRU and LSTM cells with layer normalization. We also searched overvarious optimizers (SGD, AdaGrad, AdaDelta), settling on AdaDelta. We also applied L2-norm clipping on RNNs to prevent exploding gradients. Our optimization objective is to minimize square loss.

우리는 두 종류의 deep neural networks를 평가했다: MLPs and RNNs. 이들 각각에 대해 우리는 sample 내 기간에 hyperparameters를 조정한다. 그런 다음 sample 밖의 기간에 model의 결과를 평가했다. MLP와 RNN 모두에 대해 64, 128, 256, 512 또는 1024 nodes로 1, 2 및 4 layers로 평가된 architectures를 고려한다. 또한 input과 숨겨진 layer 사이의 dropout 사용을 평가한다. MLP의 경우 ReLU activation을 사용하고 layer 간에 batch normalization을 적용한다. RNN의 경우 layer nomalization을 통해 GRU 및 LSTM cells을 test한다. AdaDelta에 setting한 다양한 optimazer (SGD, AdaGrad, AdaDelta)도 탐색했다. 또한 exploding gradients를 방지하기 위해 RNN에 L2-norm clipping을 적용했다. 우리의 최적화 목적은 square loss를 최소화하는 것이다.

To account  for the fact that we care more about our prediction of EBIT over the other fundamental values, we up-weight it in the loss (introducing a hyperparameter $\alpha$1). For RNNs, because we care primarily about the accuracy of the prediction at the final time step (of 5), we upweight the loss at the final time step by hyperparameter $\alpha$2(as in [9]). Some results from our hyperparameter search on in-sample data are displayed in Table 1. These hyperparameters resulted in MSE on in-sample validation data of 0*.*6141 for and 0*.*6109 for the MLP and RNN, respectively.

우리가 다른 fundamental values에 대한 EBIT예측에 대해 더 신경쓴다는 사실을 설명하기 위해, 우리는 loss에서 그것의 가중치(의역)를 늘린다(hyperparameter $\alpha$1을 도입함). RNN의 경우 최종 time step (5)에서 예측의 정확성에 대해 주로 염려하기 때문에 [9]에서와 같이 hyperparameter $\alpha$2로 최종 time step에서 loss를 상향 조정한다. sample data의 hyperparameter 검색 결과는 표 1에 나와 있다. 이러한 hyperparameters는 MLP 및 RNN에 대해 각각 0.6141 및 0.6109의 sample 내 유효성 검사 data에서 MSE를 발생시켰다.

![Figure 1](.\image/figure2.svg)

그림 2 (a) 2000-2014 기간 동안의 sample 밖의 성능. All factor model은 EBIT/EV를 사용한다. QFM은 현재 EBIT를 사용하지만 제안된 LFM은 predicted EBIT를 사용한다. Price-LSTM은 가격을 직접 예측할 수 있도록 훈련되었다.

그림 2 (b) MSE(오렌지색) 및 naive predictor(검정색)의 샘플을 벗어난 기간의 MSE.

Table 1 : Final hyperparameters for MLP and RNN

|        Hyperparameter        | MLP  | RNN  |
| :--------------------------: | :--: | :--: |
|         Hidden Units         | 1024 |  64  |
|        Hidden Layers         |  2   |  2   |
|   Inut Dropout Keep Prob.    | 1.0  | 1.0  |
|  Hidden Dropout Keep Prob.   | 0.5  | 1.0  |
| Recurrent Dropout Keep Prob. | n/a  | 0.7  |
|      Max Gradient Norm       | 1.0  | 1.0  |
|          $\alpha_1$          | 0.75 | 0.5  |
|          $\alpha_2$          | n/a  | 0.7  |

**Evaluation** As a first step in evaluating the forecast produced by the neural  networks, we compare the MSE of the predicted fundamental on out-of-sample data with a naive prediction where predicted fundamentals at time $t$ is assumed to be the same as the fundamentals  at $t - 12$. To compare the practical utility of traditional factor models vs lookahead factor models we employ an industry grade investment simulator. The simulator evaluates hypothetical stock portfolios constructed on out-of-sample data. Simulated investment returns reflect how an investor might have performed had they invested in the past according to given strategy.

**평가** neural networks에 의해 생성된 예측을 평가하는 첫 번째 단계로, sample 밖의 data에 대한 예측의 기본의 MSE와 naive prediction($t-12$에서의 fundamentals와 $t$에서 예측된 fundamentals가 같다고 가정한 naive prediction)과 비교한다.

The simulation results reflect assets-under-management at the start of each month that, when adjusted by the S&P 500 Index Price to January 2010, are equal to $100 million. We construct portfolios by ranking all stocks according to the factor EBIT/EV in each month and investing equal amounts of capital into the top 50
stocks holding each stock for one-year. When a stock falls out of the top 50 after one year, it is sold with proceeds reinvested in another highly ranked stock that is not currently
in the simulated portfolio. We  limit the number of shares of a security bought or sold in a month  to no
more than 10% of the monthly volume for a security. Simulated prices for stock purchases and sales
are based on the volume-weighted daily closing price of the security during the first 10 trading days
of each month. If a stock paid a dividend during the period it was held, the dividend was credited to the simulated fund in proportion to the shares held. Transaction costs are factored in as $0.01 per share, plus an additional slippage factor that increases as a square of the simulation’s volume participation in a security. Specifically, if participating at the maximum 10% of monthly volume, the simulation buys at 1% more than the average market price and sells at 1% less than the average market price. Slippage accounts for transaction friction, such as bid/ask spreads, that exists in real life trading.

**Our results demonstrate** a clear advantage for the lookahead factor model. In nearly all months, however
turbulent the market, neural networks outperform the naive predictor (that fundamentals remains unchanged) (Figure [2b]). Simulated portfolios lookahead factor strategies with MLP and RNN perform
similarly, both beating traditional factor models (Table [2a).](#_bookmark1)

### 3. Discussion

In thispaper we demonstrate a new approach for automated stock market prediction basedon time series analysis. Rather than predicting price directly, predictfuture fundamental data from a trailingwindow of values. Retrospective analysis with an oracle motivates the approach,demonstrating the superiority ofLFM over standard factor approaches. In future work we will thoroughlyinvestigate the relative advantages of LFMs vs directly predicting price. We also plan to investigate the effects of the sampling window, input length, and lookahead distance.



### References

[1]: Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio.Neural machine translation by jointly learningto align and translate. *arXiv:1409.0473*, 2014.

[2]     BilbertoBatres-Estrada. Deep learning for multivariate financial time series. 2015.

[3]     Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag, and Yan Liu. Recurrent neural networks for multivariatetime series with missing values. *arXiv:1606.01865*, 2016.

[4]     Kai Chen, Yi Zhou, and Fangyan Dai. A lstm-based method for stock returnsprediction: A case study of china stock market. In *Big Data (Big Data), 2015 IEEE International Conference on*. IEEE, 2015.

[5]     Xiao Ding, Yue Zhang, Ting Liu, and Junwen Duan. Deep learning forevent-driven stock prediction.

[6]     Jeffrey Donahue, Lisa AnneHendricks, Sergio Guadarrama, Marcus Rohrbach, Subhashini Venugopalan, KateSaenko, and Trevor Darrell.Long-term recurrent convolutional networks for visual recognition anddescription. In *CVPR*, 2015.

[7]     Eugene F. Fama and Kenneth R. French. The cross-section of expectedstock returns. *Journal of Finance 47,427-465.*, 1992.

[8]     Hengjian Jia. Investigation into the effectiveness of long short term memorynetworks for stock price prediction. *arXiv:1603.07893*, 2016.

[9]     Zachary C Lipton, David C Kale,Charles Elkan, and Randall Wetzell. Learning to diagnose with lstm recurrentneural networks. *ICLR*, 2016.

[10]      Zachary C Lipton, David C Kale, andRandall Wetzel. Directly modelingmissing data in sequences with rnns: Improved classification of clinical timeseries. *Machine Learning for Healthcare(MLHC)*, 2016.

[11]      Junhua Mao, Wei Xu, Yi Yang, Jiang Wang, Zhiheng Huang,and Alan Yuille. Deep captioningwith multimodal recurrent neural networks (m-rnn). *ICLR*, 2015.

[12]     Eero Pätäri and Timo Leivo. A closer look at the value premium.*Journal of Economic Surveys, Vol.31, Issue 1, pp. 79-168, 2017*, 2017.

[13]     <a id="13"></a> Robert J Shiller. Do stock pricesmove too much to be justified by subsequent changes in dividends?, 1980.

[14]      Ilya Sutskever, Oriol Vinyals, andQuoc V Le. Sequence to sequence learning with neural networks. In *NIPS*,2014.

[15]      Subarna Tripathi, Zachary C Lipton,Serge Belongie, and Truong Nguyen. Context matters: Refining object detectionin video with recurrent neural networks. *BMVC*, 2016.

[16]      Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan.Show and tell: A neural image caption generator. In *CVPR*, 2015.

[17]      Barack Wamkaya Wanjawa and Lawrence Muchemi.Ann model to predict stock prices at stockexchange markets. *arXiv:1502.06434*, 2014.