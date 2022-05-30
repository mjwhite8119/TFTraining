import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

#### STEP 1 - Load the h5 model
model = tf.keras.models.load_model('model.h5')

#### STEP 2 - Save as Simple SavedModel
tf.saved_model.save(model, 'simple_model')

### STEP 3 - Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

### STEP 4 - Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

### STEP 5 - Save frozen graph from frozen ConcreteFunction
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir="./",
                    name="simple_frozen_graph.pb",
                    as_text=False)

### STEP 6 - Use blobconverter.luxonis.com to convert to blob
# --input_shape=[1,66,200,3] must be specified in the blobconverter Advanced options 
# which is on the right hand side of the page.