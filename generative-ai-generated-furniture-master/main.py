import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
import torch
from diffusers import StableDiffusionPipeline
import cv2
import numpy as np
from PIL import Image
from rembg import remove
from PyQt5.QtGui import QImage, QPixmap


# 處理串流捕獲
class VideoThread(QThread):

    # 將訊號轉成 nparray
    change_pixmap_signal = pyqtSignal(np.ndarray)

    # 初始化設定
    def __init__(self):
        super().__init__()
        self._run_flag = True

    # 開啟攝影機或串流
    def run(self):
        cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture("rtmp://140.116.56.6:1935/live")
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        cap.release()

    # 停止攝影機或串流
    def stop(self):
        self._run_flag = False
        self.wait()

# 處理圖片生成、去背、UI Interface
class ImageApp(QWidget):

    # 初始化設定
    def __init__(self):

        # 初始化界面
        super().__init__()
        self.initUI()

        # 載入 Stable Diffusion 模型
        self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")

        # 初始化圖片處理參數
        self.image_np = None
        self.x_offset = 50
        self.y_offset = 50
        self.scale_factor = 2 / 3
        self.alpha_factor = 1.0
        self.angle = 0
        self.flip_horizontal = False

        # 啟動串流或攝影機
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    # 初始化界面
    def initUI(self):

        # UI 介面名稱
        self.setWindowTitle('Stable Diffusion UI')
        vbox = QVBoxLayout()

        # 圖片顯示標籤
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter) # 圖片居中
        vbox.addWidget(self.image_label)
        hbox_prompt = QHBoxLayout()
        
        # 提示標籤
        self.prompt_label = QLabel('Prompt:')
        hbox_prompt.addWidget(self.prompt_label)

        # 提示輸入框
        self.prompt_entry = QLineEdit(self)
        hbox_prompt.addWidget(self.prompt_entry)

        # 生成圖片按鈕
        self.enter_button = QPushButton('Enter', self)
        self.enter_button.clicked.connect(self.generate_image)
        hbox_prompt.addWidget(self.enter_button)

        # 重置按鈕
        self.reset_button = QPushButton('Reset', self)
        self.reset_button.clicked.connect(self.reset_image)
        hbox_prompt.addWidget(self.reset_button)
        vbox.addLayout(hbox_prompt)

        # 控制信息標籤
        self.info_label = QLabel('Use the buttons below to control the image:')
        vbox.addWidget(self.info_label)
        hbox_controls = QHBoxLayout()

        # 控制按鈕設置
        buttons = [
            ('Up', 'w'), ('Down', 's'), ('Left', 'a'), ('Right', 'd'),
            ('Smaller', '1'), ('Larger', '2'), ('More Transparent', '3'),
            ('Less Transparent', '4'), ('Rotate Left', '5'), ('Rotate Right', '6'), ('Flip Horizontal', '7')
        ]
        for (text, key) in buttons:
            button = QPushButton(text, self)
            button.clicked.connect(lambda _, k=key: self.key_pressed(k))
            hbox_controls.addWidget(button)
        vbox.addLayout(hbox_controls)
        self.setLayout(vbox)

    # 按鍵控制
    def key_pressed(self, key):
        if key == 'w':
            self.y_offset -= 10
        elif key == 's':
            self.y_offset += 10
        elif key == 'a':
            self.x_offset -= 10
        elif key == 'd':
            self.x_offset += 10
        elif key == '1':
            self.scale_factor = max(0.1, self.scale_factor - 0.1)
        elif key == '2':
            self.scale_factor += 0.1
        elif key == '3':
            self.alpha_factor = max(0.0, self.alpha_factor - 0.1)
        elif key == '4':
            self.alpha_factor = min(1.0, self.alpha_factor + 0.1)
        elif key == '5':
            self.angle -= 10
        elif key == '6':
            self.angle += 10
        elif key == '7':
            self.flip_horizontal = not self.flip_horizontal

    # 生成圖片
    def generate_image(self):

        # 獲取用戶輸入的提示詞
        prompt = self.prompt_entry.text()

        # 生成圖片
        if prompt:

            # 使用Stable Diffusion生成圖片
            with torch.autocast("cuda"):
                image = self.pipe(prompt).images[0]
            
            # 將生成的圖片轉換為numpy數組
            self.image_np = np.array(image)
            
            # 移除背景
            self.image_np = remove(self.image_np)
            
            # 保存生成的圖片到文件
            Image.fromarray(self.image_np).save("generated_image.png")
            
            # 根據圖片的通道數進行轉換
            if self.image_np.shape[2] == 3:
                self.image_np = cv2.cvtColor(self.image_np, cv2.COLOR_BGR2BGRA)
            elif self.image_np.shape[2] == 4:
                self.image_np = cv2.cvtColor(self.image_np, cv2.COLOR_RGBA2BGRA)


    # 重置圖片
    def reset_image(self):
        self.prompt_entry.clear()
        self.x_offset = 50
        self.y_offset = 50
        self.scale_factor = 2 / 3
        self.alpha_factor = 1.0
        self.angle = 0
        self.flip_horizontal = False
        self.image_np = None

    # 更新圖片顯示
    def update_image(self, cv_img):

        # 更新圖片
        if self.image_np is not None:

            # 計算新的高度並保持原圖片的縱橫比
            new_height = int(cv_img.shape[0] * self.scale_factor)
            aspect_ratio = self.image_np.shape[1] / self.image_np.shape[0]
            new_width = int(new_height * aspect_ratio)
            
            # 調整圖片大小
            resized_image = cv2.resize(self.image_np, (new_width, new_height))

            # 旋轉圖片
            M = cv2.getRotationMatrix2D((new_width // 2, new_height // 2), self.angle, 1.0)
            rotated_image = cv2.warpAffine(resized_image, M, (new_width, new_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            # 水平翻轉圖片
            if self.flip_horizontal:
                rotated_image = cv2.flip(rotated_image, 1)

            # 計算透明度
            alpha_s = (rotated_image[:, :, 3] / 255.0) * self.alpha_factor
            alpha_l = 1.0 - alpha_s

            # 確保圖片不超出邊界
            self.x_offset = max(0, min(self.x_offset, cv_img.shape[1] - new_width))
            self.y_offset = max(0, min(self.y_offset, cv_img.shape[0] - new_height))

            # 合併背景和前景圖片
            for c in range(0, 3):
                cv_img[self.y_offset:self.y_offset+rotated_image.shape[0], self.x_offset:self.x_offset+rotated_image.shape[1], c] = (
                    alpha_s * rotated_image[:, :, c] + alpha_l * cv_img[self.y_offset:self.y_offset+rotated_image.shape[0], self.x_offset:self.x_offset+rotated_image.shape[1], c]
                )

        # 將更新後的cv圖片轉換為Qt圖片並顯示
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    # 將 opencv 圖片轉換為 QPixmap
    def convert_cv_qt(self, cv_img):
        
        # 將 BGR 圖片轉換為 RGB 圖片
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        # 獲取圖像的高度、寬度和通道數
        h, w, ch = rgb_image.shape
        
        # 計算每行的字節數
        bytes_per_line = ch * w
        
        # 將 numpy 數組轉換為 QImage
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 調整 QImage 大小，保持縱橫比
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        
        # 將 QImage 轉換為 QPixmap 並返回
        return QPixmap.fromImage(p)

    # 關閉事件處理
    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

# 主程式
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageApp()
    ex.show()
    sys.exit(app.exec_())