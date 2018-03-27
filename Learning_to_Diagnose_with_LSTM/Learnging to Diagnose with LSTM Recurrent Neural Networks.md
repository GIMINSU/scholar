# Learnging to Diagnose with LSTM Recurrent Neural Networks

> LSTM RNN으로 진단하는 방법 학습
>
> Zachar C.Lipton, David C.Kale, Charles Elkan, Randall Wetzel(2016)

### Abstract

Clinical medical data, especially in the intensive care unit (ICU), consist of multi- variate time series of observations. For each patient visit (or *episode*), sensor data and lab test results are recorded in the patient’s Electronic Health Record (EHR). While potentially containing a wealth of insights, the data is difficult to mine effectively, owing to varying length, irregular sampling and missing data. Recurrent Neural Networks (RNNs), particularly those using Long Short-Term Memory (LSTM) hidden units, are powerful and increasingly popular models for learning from sequence data. They effectively model varying length sequences and capture long range dependencies. We present the first study to empirically evaluate the ability of LSTMs to recognize patterns in multivariate time series of clinical measurements. Specifically, we consider multilabel classification of diagnoses, training a model to classify 128 diagnoses given 13 frequently but irregularly sampled clinical measurements. First, we establish the effectiveness of a simple LSTM network for modeling clinical data. Then we demonstrate a straightforward and effective training strategy in which we replicate targets at each sequence step. Trained only on raw time series, our models outperform several strong baselines, including a multilayer perceptron trained on hand-engineered features.

임상 의료 data, 특히 중환자실(ICU)에서의 data는 다변량의 시계열 관측치들로 구성된다. 각 환자 방문(또는 episode)마다 sensor data 및 실험실 test 결과가 patient's Electronic Health Recode(EHR)에 기록된다. 많은 시사점을 잠재적으로 포함하고는 있지만 다양한 길이, 불규칙 sampling, missing data 때문에 데이터를 효과적으로 채굴하기 어렵다. 반복적인 신경 회로망(RNNs), 특히 긴 단기 기억(LSTM) hidden unit을 사용하는 RNNs는 sequence data로부터 학습하는가장 강력하고 인기가 많은 모델이다. 이들은 다양한 길이의 sequence를 효과적으로 modeling하고 긴 범위의 종속성을 포착한다. 우리는 임상 측정의 다변량 시계열에서 pattern을 인식하는 LSTM의 능력을 경험적으로 평가하기 위한 첫 번째 연구를 제시한다. 특히 우리는 13개의 빈번하지만 불규칙적으로 표본화된 임상 측정치를 제공하는 128개의 진단을 분류하기 위한 model을 훈련하면서 진단의 다중 라벨 분류를 고려한다. 첫째, 우리는 임상 데이터 모델링을 위한 간단한 LSTM network의 효과성을 입증한다. 그런 다음 각 sequence 단계에서 목표를 복제하는 간단하고 효과적인 교육 전략을 보여준다.  원시 시계열에 대해서만 훈련된 이 모델은 수작업으로 설계된 feature에 대한 교육을 받은 다층 퍼셉트론을 포함하여 몇 가지 강력한 기준선보다 우수한 성능을 보인다.

