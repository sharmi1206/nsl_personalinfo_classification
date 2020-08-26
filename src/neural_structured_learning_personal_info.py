import tensorflow as tf
import neural_structured_learning as nsl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("../data/adult.all.txt", sep=", ")

print(df.shape)
print(df.columns)

df.columns = ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education-num', 'Marital-status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-gain', 'Capital-loss',
              'Hours-per-week', 'Native-country', 'Salary']


df['Age'] = df['Age'].astype(np.float32)

df['fnlwgt'] = df['fnlwgt'].astype(np.float32)
df['Education-num'] = df['Education-num'].astype(np.float32)
df['Capital-gain'] = df['Capital-gain'].astype(np.float32)
df['Capital-loss'] = df['Capital-loss'].astype(np.float32)
df['Hours-per-week'] = df['Hours-per-week'].astype(np.float32)

df['Workclass'] = df['Workclass'].astype('category').cat.codes.astype(np.float32)
df['Education'] = df['Education'].astype('category').cat.codes.astype(np.float32)
df['Marital-status'] = df['Marital-status'].astype('category').cat.codes.astype(np.float32)
df['Occupation'] = df['Occupation'].astype('category').cat.codes.astype(np.float32)
df['Relationship'] = df['Relationship'].astype('category').cat.codes.astype(np.float32)
df['Race'] = df['Race'].astype('category').cat.codes.astype(np.float32)
df['Sex'] = df['Sex'].astype('category').cat.codes.astype(np.float32)
df['Native-country'] = df['Native-country'].astype('category').cat.codes.astype(np.float32)
df['Salary'] = df['Salary'].astype('category').cat.codes.astype(np.float32)

df_train = df[['Age', 'Workclass', 'fnlwgt', 'Education', 'Education-num', 'Marital-status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-gain', 'Capital-loss',
              'Hours-per-week', 'Native-country']]

df_test= df[['Salary']]

print(df.head())
df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df_train, df_test, test_size=0.3, random_state=42)

x_train = df_x_train.values
x_test = df_x_test.values


y_train = np.asarray(df_y_train.values.reshape((-1,))).astype(np.float32)
y_test = np.asarray(df_y_test.values.reshape((-1,))).astype(np.float32)

# Prepare data.
print(np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test), np.unique(y_train), np.unique(y_test))

# # Create a base model -- sequential, functional, or subclass.
model = tf.keras.Sequential([
    tf.keras.Input((x_train.shape[1]), name='feature'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

# Wrap the model with adversarial regularization.
adv_config = nsl.configs.make_adv_reg_config(multiplier=0.1, adv_step_size=0.01)
adv_model = nsl.keras.AdversarialRegularization(model, adv_config=adv_config, base_with_labels_in_features=True)

# Compile, train, and evaluate.
adv_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
adv_model.fit({'feature': x_train, 'label': y_train}, batch_size=32, epochs=5)
prediction = adv_model.predict({'feature': x_test, 'label': y_test})
results = adv_model.evaluate({'feature': x_test, 'label': y_test}, return_dict=True)

print("Prediction", adv_model.predict({'feature': x_test, 'label': y_test}))
print(results)
