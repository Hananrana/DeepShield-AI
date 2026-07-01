from inference.image_predict import predict_image

result = predict_image("test.jpg")

print("\nPrediction Result")
print("=" * 50)

print("Label      :", result["label"])
print("Confidence :", result["confidence"], "%")

print("=" * 50)