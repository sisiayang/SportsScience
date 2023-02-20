# SportsScience
根據排球賽事中選手的行為預測是否會得分 (失分)  
further work: 分析影響得分的關鍵因素

* use tensorflow to implement

### Model 架構
* space embedding layer (參考paper [1])
* action embedding layer (參考 paper [1])
* input concat layer ([embeded space, embeded action, other features])
* cnn layer
* dense layer


### Reference
paper [1]: https://arxiv.org/abs/2109.06431
