
print("Importing torch...")
import torch
print(f"Torch version: {torch.__version__}")

print("Importing KAN...")
try:
    from kan import KAN
    print("KAN imported successfully.")
    
    model = KAN(layers_hidden=[8, 1])
    print("KAN instantiated.")
    
    x = torch.randn(2, 8)
    y = model(x)
    print("KAN forward pass successful.")
except Exception as e:
    print(f"KAN failed: {e}")
    import traceback
    traceback.print_exc()

print("Done.")
