__vit transformer__

使用 mnist 数据对模型进行训练、测试

训练 `100` 个 `epoch`, 测试集中使用分类精度为 `0.993`

```python
import torch
from model import VIT

net = VIT(image_size=224,channels=3, patch_size=16, num_classes=10, 
          dim=256, transLayer=6, multiheads=8)

img = torch.randn(2, 3, 224, 224)

out = net(img) # (1, num_classes)
print(out.shape)
torch.Size([2, 10])

# 保存为 onnx 模型
net.eval()
torch.onnx.export(net, img, "vit.onnx", verbose=1, training=torch.onnx.TrainingMode.EVAL,
                    input_names=["inputNode"], output_names=["outNode1"], opset_version=11, )
```
`VIT2` 为测试将模型转为 `onnx` 时, `einops` 中一些维度变换的函数能否被转换



参考：
>https://github.com/lucidrains/vit-pytorch