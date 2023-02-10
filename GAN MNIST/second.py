import torch
import matplotlib.pyplot as plt

model = torch.jit.load("model_gan_opt_trained.pt")

output = model(torch.full((1, 100, 1, 1), 0.5))

output = (output + 1) / 2.0

plt.imshow(torch.permute(output[0], [1, 2, 0]), cmap='gray')
plt.show()