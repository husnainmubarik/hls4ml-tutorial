#!/usr/bin/env python
# coding: utf-8

# # Part 1: Getting started

# In[ ]:


from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

#from IPython import get_ipython

#get_ipython().run_line_magic('matplotlib', 'inline')


# ## Fetch the jet tagging dataset from Open ML

# In[ ]:


data = fetch_openml('hls4ml_lhc_jets_hlf')
X, y = data['data'], data['target']


# ### Let's print some information about the dataset
# Print the feature names and the dataset shape

# In[ ]:


print(data['feature_names'])
print(X.shape, y.shape)
print(X[:5])
print(y[:5])


# As you saw above, the `y` target is an array of strings, e.g. \['g', 'w',...\] etc.
# We need to make this a "One Hot" encoding for the training.
# Then, split the dataset into training and validation sets

# In[ ]:


le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y, 5)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(y[:5])


# In[ ]:


scaler = StandardScaler()
X_train_val = scaler.fit_transform(X_train_val)
X_test = scaler.transform(X_test)


# ## Now construct a model
# We'll use 3 hidden layers with 64, then 32, then 32 neurons. Each layer will use `relu` activation.
# Add an output layer with 5 neurons (one for each class), then finish with Softmax activation.

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from callbacks import all_callbacks


# In[ ]:


model = Sequential()
model.add(Dense(64, input_shape=(16,), name='fc1', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
model.add(Activation(activation='relu', name='relu1'))
model.add(Dense(32, name='fc2', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
model.add(Activation(activation='relu', name='relu2'))
model.add(Dense(32, name='fc3', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
model.add(Activation(activation='relu', name='relu3'))
model.add(Dense(5, name='output', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
model.add(Activation(activation='softmax', name='softmax'))


# ## Train the model
# We'll use Adam optimizer with categorical crossentropy loss.
# The callbacks will decay the learning rate and save the model into a directory 'model_1'
# The model isn't very complex, so this should just take a few minutes even on the CPU.
# If you've restarted the notebook kernel after training once, set `train = False` to load the trained model.

# In[ ]:


train = True
if train:
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
    callbacks = all_callbacks(stop_patience = 1000,
                              lr_factor = 0.5,
                              lr_patience = 10,
                              lr_epsilon = 0.000001,
                              lr_cooldown = 2,
                              lr_minimum = 0.0000001,
                              outputDir = 'test_model')
    model.fit(X_train_val, y_train_val, batch_size=1024,
              epochs=30, validation_split=0.25, shuffle=True,
              callbacks = callbacks.callbacks)
else:
    from tensorflow.keras.models import load_model
    model = load_model('test_model/KERAS_check_best_model.h5')


# ## Check performance
# Check the accuracy and make a ROC curve

# In[ ]:


import plotting
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
y_keras = model.predict(X_test)
print("Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))))
plt.figure(figsize=(9,9))
_ = plotting.makeRoc(y_test, y_keras, le.classes_)


# # Convert the model to FPGA firmware with hls4ml
# Now we will go through the steps to convert the model we trained to a low-latency optimized FPGA firmware with hls4ml.
# First, we will evaluate its classification performance to make sure we haven't lost accuracy using the fixed-point data types. 
# Then we will synthesize the model with Vivado HLS and check the metrics of latency and FPGA resource usage.
# 
# ## Make an hls4ml config & model
# The hls4ml Neural Network inference library is controlled through a configuration dictionary.
# In this example we'll use the most simple variation, later exercises will look at more advanced configuration.

# In[ ]:


import hls4ml
config = hls4ml.utils.config_from_keras_model(model, granularity='model')
print(config)
hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir='test_model/hls4ml_prj')


# Let's visualise what we created. The model architecture is shown, annotated with the shape and data types

# In[ ]:


hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)


# ## Compile, predict
# Now we need to check that this model performance is still good. We compile the hls_model, and then use `hls_model.predict` to execute the FPGA firmware with bit-accurate emulation on the CPU.

# In[ ]:


hls_model.compile()
y_hls = hls_model.predict(X_test)


# ## Compare
# That was easy! Now let's see how the performance compares to Keras:

# In[ ]:


print("Keras  Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))))
print("hls4ml Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls, axis=1))))

fig, ax = plt.subplots(figsize=(9, 9))
_ = plotting.makeRoc(y_test, y_keras, le.classes_)
plt.gca().set_prop_cycle(None) # reset the colors
_ = plotting.makeRoc(y_test, y_hls, le.classes_, linestyle='--')

from matplotlib.lines import Line2D
lines = [Line2D([0], [0], ls='-'),
         Line2D([0], [0], ls='--')]
from matplotlib.legend import Legend
leg = Legend(ax, lines, labels=['keras', 'hls4ml'],
            loc='lower right', frameon=False)
ax.add_artist(leg)


# ## Synthesize
# Now we'll actually use Vivado HLS to synthesize the model. We can run the build using a method of our `hls_model` object.
# After running this step, we can integrate the generated IP into a workflow to compile for a specific FPGA board.
# In this case, we'll just review the reports that Vivado HLS generates, checking the latency and resource usage.
# 
# **This can take several minutes.**

# In[ ]:


#hls_model.build()


# ## Check the reports
# Print out the reports generated by Vivado HLS. Pay attention to the Latency and the 'Utilization Estimates' sections

# In[ ]:


#hls4ml.report.read_vivado_report('model_1/hls4ml_prj/')


# ## Exercise
# Since `ReuseFactor = 1` we expect each multiplication used in the inference of our neural network to use 1 DSP. Is this what we see? (Note that the Softmax layer should use 5 DSPs, or 1 per class)
# Calculate how many multiplications are performed for the inference of this network...
# (We'll discuss the outcome)

# In[ ]:




