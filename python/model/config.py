# Transformer configuration
transformer_params = {'n_layers': 1,
                      'n_heads': 8,
                      'dropout_rate': 0.2}

# Clinical metadata
categorical_features = {'names': ['Sex', 'Hypertension', 'Smoking', 'Atrial_Fibrillation', 'mTICI'], 'categories': [2, 2, 2, 2, 3]}
continuous_features = ['Age', 'Onset2CTP', 'CTP2Recanalization', 'NIHSS00']
target_feature = {'name': ['mRs90_binary'], 'categories': [2]}  # mRs90_binary = 0-2 (good), 3-6 (severe)

# Training configuration
train_params = {'n_epochs': 100,
                'learning_rate': 0.001}

# Data generator parameters
params = {'imagePath': './datasets/',
          'dictFile': './datasets/patient_dictionary.pickle',
          'clinicalFile': './datasets/clinical_metadata.csv',  # PATIENT_ID column must exist
          'resultsPath': './results/',
          'dim': (512, 512, 16),
          'batch_size': 1,
          'timepoints': 32,
          'n_classes': 2,
          'features': [continuous_features, categorical_features['names']]}
