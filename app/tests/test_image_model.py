from inference.image_predict import model, THRESHOLD

print("=" * 50)
print("Fusion V2 Model Loaded Successfully")
print("=" * 50)

print("Threshold:", THRESHOLD)

total_params = sum(p.numel() for p in model.parameters())

print("Total Parameters:", f"{total_params:,}")

print("=" * 50)