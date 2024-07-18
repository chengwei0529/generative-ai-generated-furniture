# generative-ai-generated-furniture-master
This is the final project of COMPUTER VISION IN DEEP LEARNING IMPLEMENTATION AND ITS APPLICATIONS.

# Stable Diffusion UI with PyQt5

這是一個使用 PyQt5 製作的 GUI 應用程式，結合了 Stable Diffusion 模型和即時串流捕獲。此專案允許用戶使用提示詞生成圖片，並且可以對生成的圖片進行即時處理和顯示。

## 功能

- 即時串流或攝影機捕獲
- 使用 Stable Diffusion 生成圖片
- 即時圖片處理：位置、縮放、透明度、旋轉、翻轉
- 去除圖片背景
- GUI 控制界面

## 流程圖

![流程圖](images/flowchart.png)

1. 開啟環控攝影機並開始錄影
2. Streaming Server 開始即時串流
3. Jetson Xavier NX 拉流
4. 進行即時影像生成
5. 使用 rembg 去背
6. 將結果及時生成於串流影像之中
7. 使用者可自行調整圖片

## 安裝

1. 創建並激活 Conda 環境：

    ```bash
    conda create --name diffusers-gui python=3.8
    conda activate diffusers-gui
    ```

2. 安裝 CUDA 工具包和 CuDNN：

    ```bash
    conda install -c conda-forge cudatoolkit cudnn -y
    ```

3. 安裝所需的 Python 套件：

    ```bash
    pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
    pip install PyQt5==5.15.10 pillow==10.4.0 rembg==2.0.57 numpy==1.24.4 diffusers==0.29.2 transformers==4.42.3 accelerate==0.32.1
    ```

## 使用

1. 執行主程式：

    ```bash
    python main.py
    ```

2. 在彈出的 GUI 中，可以通過輸入提示詞並點擊 "Enter" 按鈕生成圖片。使用下方的控制按鈕可以調整圖片的位置、大小、透明度、旋轉角度和翻轉。

## 結果展示

### UI 界面

![UI 界面](images/ui_interface.png)

- 加載模型並生成圖片
- 輸入提示詞並重置
- 調整圖片位置、透明度、大小、旋轉和翻轉

### Text to Image（With ChatGPT）

![Text to Image](images/text_to_image.png)

- "A very handsome man is sitting on a chair."
- "Turn into a very ugly man."

### Image and Text to Image

![Image and Text to Image](images/image_text_to_image.png)

- 根據提示詞生成圖片並進行處理

## 問題

- Jetson Xavier NX 的運行環境可以執行程式碼，但計算資源不足導致機器崩潰。
- 每次運行時，屏幕都會凍結。
- Jetson Xavier NX 在全性能運行時經常崩潰，因為計算資源不足。
- 出現 CPU 問題。

## 依賴

- Python 3.8
- PyQt5 5.15.10
- Pillow 10.4.0
- rembg 2.0.57
- numpy 1.24.4
- torch 2.0.0+cu118
- torchaudio 2.0.0+cu118
- torchvision 0.15.0+cu118
- diffusers 0.29.2
- transformers 4.42.3
- accelerate 0.32.1

## 參考

- [diffusers](https://github.com/huggingface/diffusers)

## 授權

此專案採用 MIT 授權。

## 感謝您的關注！
