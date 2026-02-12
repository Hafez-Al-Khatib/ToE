
import torch
from kan import KAN

print("Testing KAN gradients...")
model = KAN(layers_hidden=[8, 1])
x = torch.randn(2, 8, requires_grad=True)
y = model(x)
print(f"y requires_grad: {y.requires_grad}")
print(f"y grad_fn: {y.grad_fn}")

try:
    loss = y.sum()
    loss.backward()
    print("Backward pass successful.")
    print(f"x.grad norm: {x.grad.norm().item()}")
except Exception as e:
    print(f"Backward failed: {e}")
    import traceback
    traceback.print_exc()
