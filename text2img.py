import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# 載入模型
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# 設定生成圖片的提示詞
prompt = "A bustling cityscape at night, illuminated by the vibrant lights of skyscrapers and neon signs. The streets are filled with a mix of cars, bicycles, and pedestrians, all moving with a sense of purpose. Street vendors line the sidewalks, their stalls offering a variety of foods and goods. Above, the sky is a deep indigo, dotted with a few stars visible through the urban glow. In the distance, a river reflects the shimmering lights of the buildings, creating a dazzling mirror effect. The air is filled with the sounds of honking horns, chatter, and occasional music from street performers, encapsulating the lively energy of the city."

# 生成圖片
with torch.autocast("cuda"):
    image = pipe(prompt).images[0]

# 保存圖片
image.save("image.png")

# 顯示圖片
image.show()

print("Image saved as 'image.png'")