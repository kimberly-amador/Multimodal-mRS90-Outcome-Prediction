import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import random
import config as cfg
import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from utils import ENCODER, SelfAttention, MultimodalFusion
from data_generator import DataGenerator_CTP
from lr_scheduler import CosineDecayRestarts
from tensorflow.keras import Model, optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Concatenate, Input, Dense, Lambda, Embedding, Dropout, MultiHeadAttention


# Set seeds for reproducible results
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


# -----------------------------------
#            PARAMETERS
# -----------------------------------

# Print config parameters
print('TRANSFORMER PARAMETERS')
for x in cfg.transformer_params:
    print(x, ':', cfg.transformer_params[x])
print('\nTRAINING PARAMETERS')
for x in cfg.train_params:
    print(x, ':', cfg.train_params[x])
print('\nDATA GENERATOR PARAMETERS')
for x in cfg.params:
    print(x, ':', cfg.params[x])


# -----------------------------------
#            IMPORT DATA
# -----------------------------------

# Import the clinical metadata
print("[INFO] importing datasets...")
clinical_names = [*cfg.continuous_features, *cfg.categorical_features['names']]
metadata_csv = pd.read_csv(cfg.params['clinicalFile'], index_col='PATIENT_ID')[clinical_names]
mRs_scores = pd.read_csv(cfg.params['clinicalFile'], index_col='PATIENT_ID')[cfg.target_feature['name']]
IDs = metadata_csv.index.values.tolist()

# Preprocess continuous variables
scaler = MinMaxScaler()  # perform min-max scaling for each variable. Features are now in the range [0, 1]
continuous_preprocessed = scaler.fit_transform(metadata_csv[cfg.continuous_features])
continuous_preprocessed = pd.DataFrame(continuous_preprocessed, index=IDs, columns=[*cfg.continuous_features], dtype='float32')  # convert numpy array back to pandas dataframe
metadata_csv[cfg.continuous_features] = continuous_preprocessed[cfg.continuous_features]  # replace the original values with the preprocessed values

# No need to preprocess the categorical variables

# Importing patient dictionary
print("[INFO] loading patient dictionary...")
with open(cfg.params['dictFile_3D'], 'rb') as output:
    partition = pickle.load(output)

# Calling training generator
train_generator = DataGenerator_CTP(partition['training'], metadata_csv, mRs_scores, shuffle=True, **cfg.params)
val_generator = DataGenerator_CTP(partition['testing'], metadata_csv, mRs_scores, shuffle=True, **cfg.params)
test_generator = DataGenerator_CTP(partition['testing'], metadata_csv, mRs_scores, shuffle=False, **cfg.params)


# -----------------------------------
#             CALLBACKS
# -----------------------------------

# ---------- LR SCHEDULER -----------
print("[INFO] using a 'cosine' learning rate decay with periodic 'restarts'...")
n_cycles = 16
t_mul = 1
m_mul = 0.75
total_steps = len(partition['training'])//cfg.params['batch_size']*cfg.train_params['n_epochs']
sch = CosineDecayRestarts(initial_learning_rate=cfg.train_params['learning_rate'], first_decay_steps=total_steps//int(n_cycles), t_mul=float(t_mul), m_mul=float(m_mul), alpha=0.0)


# -----------------------------------
#            BUILD MODEL
# -----------------------------------

# ------ (A.1) IMAGE ENCODER -------
base_network = ENCODER.build(reg=l2(0.00005), shape=cfg.params['dim'])
absolute_diff = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))  # compute the absolute difference between tensors

# Create CTP encoders
inputs, outputs, skip = [], [], []
for w in range(cfg.params['timepoints']):
    # Create inputs and get model outputs
    i = Input(shape=(*cfg.params['dim'], 1))
    o = base_network(i)
    # Append results
    inputs.append(i)
    outputs.append(o[0])

# Concatenate latent vectors and skip connections
imaging_encoded = Concatenate(axis=1)(outputs)


# ----- (A.2) CLINICAL ENCODER -----
# Import metadata and create embeddings
# Embed categorical data with no ordinal relationship, using a fixed dim for all features
embed_dim = imaging_encoded.shape[-1]
C_inputs = []
C_embedding_outputs = []
for i in range(len(cfg.categorical_features['names'])):
    categorical_i = Input(shape=(1, ), dtype='int32', name=cfg.categorical_features['names'][i])
    categorical_i_ = Embedding(input_dim=cfg.categorical_features['categories'][i], output_dim=embed_dim)(categorical_i)
    C_inputs.append(categorical_i)
    C_embedding_outputs.append(categorical_i_)
categorical_inputs = Concatenate(axis=1)(C_embedding_outputs)

# Numerical feature tokenizer: Continuous inputs are transformed to tokens (embeddings) instead of used as-is
N_inputs = []
N_embedding_outputs = []
for feature_name in cfg.continuous_features:
    continuous_i = Input(shape=(1, ), dtype='float32', name=feature_name)
    continuous_i_ = Dense(embed_dim, activation='relu')(continuous_i)
    N_inputs.append(continuous_i)
    N_embedding_outputs.append(tf.expand_dims(continuous_i_, axis=1))
continuous_inputs = Concatenate(axis=1)(N_embedding_outputs)

# Concatenate numerical and categorical features
metadata_encoded = Concatenate(axis=1)([continuous_inputs, categorical_inputs])


# -------- (B) MULTIMODAL FUSION --------

# Self-attention: modeling temporal and clinical information via transformers
self_imaging = SelfAttention.build(imaging_encoded, num_heads=cfg.transformer_params['n_heads'], num_layers=cfg.transformer_params['n_layers'])
self_metadata = SelfAttention.build(metadata_encoded, num_heads=cfg.transformer_params['n_heads'], num_layers=cfg.transformer_params['n_layers'])

# Cross-attention: fuse the contextualized representations from images and clinical metadata; merge two embedding sequences regardless of modality
features = MultimodalFusion.build(self_imaging, self_metadata, num_heads=cfg.transformer_params['n_heads'], embed_dim=cfg.transformer_params['projection_dim'])


# ------- (C) OUTCOME PREDICTION ---------
# Stack an MLP before the last layer
mlp_hidden_units_factors = [2, 1]  # MLP hidden layer units, as factors of the number of inputs
mlp_hidden_units = [factor * features.shape[-1] for factor in mlp_hidden_units_factors]
for units in mlp_hidden_units:
    features = Dense(units, activation='relu')(features)
    features = Dropout(0.2)(features)

# Classify outputs
mRS_prediction = Dense(1, activation='sigmoid')(features)

# Build model and plot its summary
mRS_model = Model(inputs=[inputs, C_inputs, N_inputs], outputs=mRS_prediction)
mRS_model.summary()


# -----------------------------------
#            TRAIN MODEL
# -----------------------------------

# Define loss function
binary_crossentropy = tf.keras.losses.BinaryCrossentropy()

# Define optimizer
adam_optimizer = optimizers.Adam(learning_rate=sch)

# Compile model
mRS_model.compile(loss=binary_crossentropy, optimizer=adam_optimizer, metrics=['accuracy'])

# Train model
print("[INFO] training network for {} epochs...".format(cfg.train_params['n_epochs']))
H = mRS_model.fit(x=train_generator,
                  validation_data=val_generator,
                  epochs=cfg.train_params['n_epochs'])

# Save model weights
# mRS_model.save_weights(cfg.params['resultsPath'] + 'model_weights')

# Save training history
# np.save(cfg.params['resultsPath'] + 'trainHistoryDic.npy', H.history)


# -----------------------------------
#          EVALUATE MODEL
# -----------------------------------

predictions = mRS_model.predict(test_generator, steps=int(len(partition['testing'])/cfg.params['batch_size']))
np.savez_compressed(cfg.params['resultsPath'] + 'predictions', pred=predictions)
