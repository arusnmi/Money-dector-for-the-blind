import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

onnx_model = onnx.load("Money_lite/best_money_model.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("Money_lite/tf_model")

converter = tf.lite.TFLiteConverter.from_saved_model("Money_lite/tf_model")
tflite_model = converter.convert()
with open("Money_lite/best_money_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Done! Saved Money_lite/best_money_model.tflite")
