# Mask Checker 簡易臉部口罩檢測

![image](https://raw.githubusercontent.com/jason9075/mask_checker/master/img/demo.gif)

最近武漢肺癌很嚴重，公司同事聊到說可不可能做一個口罩偵測的模型，就做了這個Side Project。

模型用 tf.keras 內建的預訓練模型 MobileNetV2 來訓練，資料集從網路上找有/無口罩各400張的臉而成。

[模型下載](https://drive.google.com/file/d/1MZsr7YLeGIBNrZCqSKQuKX5fioPs9HCY/view?usp=sharing)

## 使用方法

準備好資料集後，執行 train.py 可進行訓練。
將預訓練模型放到 exported 底下，執行 webcam_demo.py 可開啟鏡頭測試。

## 備註
本專案仍非常簡陋，在遇到用手遮擋嘴巴的狀況下，人會辨識為有戴口罩，若要加強此部分，請在訓練資料集中多加入特殊樣本。
