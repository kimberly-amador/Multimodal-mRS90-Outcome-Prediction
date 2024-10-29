import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv3D, Dropout, Dense, Reshape, Embedding, MultiHeadAttention, LayerNormalization, Concatenate, Layer


# -----------------------------------
#        MODEL ARCHITECTURE
# -----------------------------------

class ENCODER:
    @staticmethod
    def build(reg=l2(), shape=(512, 512, 16), init='he_normal', name='Encoder'):
        # Create the model
        i = Input(shape=(*shape, 1))

        # The first two layers will learn a total of 16 filters with a 3x3x3 kernel size
        o = Conv3D(16, (3, 3, 3), strides=(1, 1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(i)
        d1 = Conv3D(16, (3, 3, 3), strides=(1, 1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg, name='d1')(o)
        o = Conv3D(16, (2, 2, 2), strides=(2, 2, 2))(d1)  # Down-sampling

        # Stack two more layers, keeping the size of each filter as 3x3x3 but increasing to 32 total learned filters
        o = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)
        d2 = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg, name='d2')(o)
        o = Conv3D(32, (2, 2, 2), strides=(2, 2, 2))(d2)  # Down-sampling

        # Stack two more layers, keeping the size of each filter as 3x3x3 but increasing to 64 total learned filters
        o = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)
        d3 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg, name='d3')(o)
        o = Conv3D(64, (2, 2, 2), strides=(2, 2, 2))(d3)  # Down-sampling

        # Stack two more layers, keeping the size of each filter as 3x3x3 but increasing to 128 total learned filters
        o = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)
        d4 = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg, name='d4')(o)
        o = Conv3D(1, (2, 2, 2), strides=(2, 2, 2))(d4)  # Down-sampling

        # Encoder outputs
        size = o.shape[1] * o.shape[2] * o.shape[3]
        output = Reshape((1, size))(o)
        return Model(inputs=i, outputs=[output], name=name)


class SelfAttention:
    @staticmethod
    def build(inputTensor, num_heads=8, num_layers=1, dropout=0.2):
        # Define embed_dim
        embed_dim = inputTensor.shape[2]

        # Add a Positional Embedding layer
        sequence_length = inputTensor.shape[1]  # inputs are of shape: (batch_size, n_features, filter_size)
        positions = tf.range(start=0, limit=sequence_length, delta=1)
        embedded_positions = Embedding(input_dim=sequence_length, output_dim=embed_dim)(positions)
        inputTensor += embedded_positions

        # Add CLStoken
        inputTensor = AddCLSToken()(inputTensor)

        # TransformerEncoder layer
        for _ in range(num_layers):
            attention_output, attention_scores = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout)(inputTensor, inputTensor, return_attention_scores=True)
            inputTensor = LayerNormalization(epsilon=1e-6)(inputTensor + attention_output)  # res connection provides a direct path for the gradient, while the norm maintains a reasonable scale for outputs
        return inputTensor


class MultimodalFusion:
    @staticmethod
    def build(image_embed, clinical_embed, embed_dim=1024, num_heads=8, dropout=0.2):

        # Co-attention: Query - Imaging; Key & Value - Metadata
        A_attention_output, A_attention_scores = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout)(image_embed, clinical_embed, return_attention_scores=True)
        A_proj_input = LayerNormalization(epsilon=1e-6)(image_embed + A_attention_output)
        A_proj_output = Sequential([Dense(embed_dim, activation=tf.nn.gelu), Dropout(dropout), ])(A_proj_input)
        A_x = LayerNormalization(epsilon=1e-6)(A_proj_input + A_proj_output)

        # Reduce output sequence through pooling layer
        cross_imaging = A_x[:, 0]  # extracting cls token

        # Co-attention: Query - Metadata; Key & Value - Imaging  (LayerNorm first)
        B_attention_output, B_attention_scores = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout)(clinical_embed, image_embed, return_attention_scores=True)
        B_proj_input = LayerNormalization(epsilon=1e-6)(clinical_embed + B_attention_output)
        B_proj_output = Sequential([Dense(embed_dim, activation=tf.nn.gelu), Dropout(dropout), ])(B_proj_input)
        B_x = LayerNormalization(epsilon=1e-6)(B_proj_input + B_proj_output)

        # Reduce output sequence through pooling layer
        cross_metadata = B_x[:, 0]  # extracting cls token

        # Concatenate CLS tokens
        features = Concatenate()([cross_imaging, cross_metadata])
        return features


class AddCLSToken(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cls_token = None

    def build(self, inputs_shape):
        initial_value = tf.zeros([1, 1, inputs_shape[2]])  # input.shape is [batch_size, num_patches, projection_dim]
        self.cls_token = tf.Variable(initial_value=initial_value, trainable=True, name="cls")

    def call(self, inputs, **kwargs):
        # Replace batch size with the appropriate value
        cls_token = tf.tile(self.cls_token, [tf.shape(inputs)[0], 1, 1])
        # Extra - cast to a new dtype for mixed precision training
        # cls_token = tf.cast(cls_token, dtype=inputs.dtype)
        # Append learnable parameter [CLS] class token
        concat = tf.concat([cls_token, inputs], axis=1)  # cls token placed at the start of the sequence
        return concat
