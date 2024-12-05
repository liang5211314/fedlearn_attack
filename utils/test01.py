import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# 加载模型
model = models.resnet50(pretrained=True)
model.eval()


# 预处理输入图像
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor, img  # 返回原始图像


# 获取特征图和梯度
activations = None
gradients = None


def get_gradcam_layers(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]


def get_activations(module, input, output):
    global activations
    activations = output


# 生成Grad-CAM热图
def generate_gradcam(model, image_tensor):
    target_layer = model.layer4[2].conv3

    # 注册hook
    hook_activation = target_layer.register_forward_hook(get_activations)
    hook_gradient = target_layer.register_backward_hook(get_gradcam_layers)

    # 前向传播
    output = model(image_tensor)
    pred = output.argmax(dim=1)

    # 反向传播
    model.zero_grad()
    output[0][pred].backward()

    # 计算热图
    weights = F.adaptive_avg_pool2d(gradients, 1)
    gradcam = torch.relu(torch.sum(weights * activations, dim=1)).squeeze().cpu().detach().numpy()
    gradcam = np.maximum(gradcam, 0)
    gradcam /= gradcam.max()  # 归一化

    # 重新调整大小
    gradcam = cv2.resize(gradcam, (image_tensor.size(3), image_tensor.size(2)))

    # 移除hook
    hook_activation.remove()
    hook_gradient.remove()

    return gradcam


# 将热力图叠加到原始图像上
def overlay_heatmap(original_image, heatmap, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 确保原始图像是NumPy数组并且大小匹配
    original_image = np.array(original_image)
    original_image = cv2.resize(original_image, (heatmap.shape[1], heatmap.shape[0]))

    overlay = cv2.addWeighted(original_image, alpha, heatmap, 1 - alpha, 0)
    return overlay


# 示例用法
image_paths = ['./images/image_ori1.png', './images/image_ori2.png', './images/image_ori3.png',
               './images/image_ori4.png', './images/image_ori5.png', './images/image_ori6.png']
# image_paths = ['./images/image_1.png', './images/image_2.png', './images/image_3.png',
#                './images/image_4.png', './images/image_5.png', './images/image_6.png']

fig, axes = plt.subplots(2, len(image_paths), figsize=(12,4))

for i, image_path in enumerate(image_paths):
    image_tensor, original_image = preprocess_image(image_path)
    heatmap = generate_gradcam(model, image_tensor)
    overlay_image = overlay_heatmap(original_image, heatmap)

    # 显示原始图像
    axes[0, i].imshow(original_image)
    axes[0, i].axis('off')

    # 显示叠加后的图像
    axes[1, i].imshow(overlay_image)
    axes[1, i].axis('off')
plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, hspace=0.01, wspace=0.01)  # 减小边距
plt.savefig('./images/RL/gradcam_ori.png', dpi=300)
plt.show()

