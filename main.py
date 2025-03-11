import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2DTranspose, ReLU, Reshape, Conv2D, LeakyReLU, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

#Configuração da API do Kaggle
api = KaggleApi()
api.authenticate()
api.dataset_download_files('chetankv/dogs-cats-images', path='./', unzip=True)

#Hiperparâmetros
LATENT_DIM = 100
IMAGE_SIZE = 64
CHANNELS = 3
BATCH_SIZE = 128  
EPOCHS = 100    
LEARNING_RATE = 0.0002
BETA_1 = 0.5

DATASET_PATH = "./dog vs cat"
TRAIN_DIR = os.path.join(DATASET_PATH, "dataset", "training_set", "cats")

#Gerador e Discriminador
def build_generator(latent_dim):
    model = Sequential([
        Dense(4 * 4 * 1024, input_dim=latent_dim),
        ReLU(),
        Reshape((4, 4, 1024)),
        Conv2DTranspose(512, kernel_size=5, strides=2, padding='same'),
        BatchNormalization(),
        ReLU(),
        Conv2DTranspose(256, kernel_size=5, strides=2, padding='same'),
        BatchNormalization(),
        ReLU(),
        Conv2DTranspose(128, kernel_size=5, strides=2, padding='same'),
        BatchNormalization(),
        ReLU(),
        Conv2DTranspose(CHANNELS, kernel_size=5, strides=2, padding='same', activation='tanh'),
    ])
    return model

def build_discriminator(image_shape):
    model = Sequential([
        Conv2D(64, kernel_size=5, strides=2, padding='same', input_shape=image_shape),
        LeakyReLU(alpha=0.2),
        Conv2D(128, kernel_size=5, strides=2, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Conv2D(256, kernel_size=5, strides=2, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Conv2D(512, kernel_size=5, strides=2, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Flatten(),
        Dense(1, activation='sigmoid'),
    ])
    return model


#Carregamento de Dados
def load_and_preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=CHANNELS, expand_animations=False)  # Decodifica
    img = tf.image.convert_image_dtype(img, dtype=tf.float32) # Converte para float32 e normaliza para [0, 1]
    img = (img - 0.5) / 0.5                                     # Normaliza para [-1, 1]
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    return img

def get_dataset(data_dir, batch_size):
    list_ds = tf.data.Dataset.list_files(os.path.join(data_dir, "*.jpg"), shuffle=True) 
    image_ds = list_ds.interleave(
        lambda x: tf.data.Dataset.from_tensors(load_and_preprocess_image(x)),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    # image_ds = image_ds.cache()
    image_ds = image_ds.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    return image_ds


if __name__ == '__main__':
    dataset = get_dataset(TRAIN_DIR, BATCH_SIZE)

    generator = build_generator(LATENT_DIM)
    discriminator = build_discriminator((IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

    discriminator_optimizer = Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)
    gan_optimizer = Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)

    loss_fn = tf.keras.losses.BinaryCrossentropy()

    #Função de Treinamento
    @tf.function
    def train_step(real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, LATENT_DIM])

        with tf.GradientTape(persistent=True) as tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(real_images, training=True)
            fake_output = discriminator(generated_images, training=False)

            real_loss = loss_fn(tf.ones_like(real_output), real_output)
            fake_loss = loss_fn(tf.zeros_like(fake_output), fake_output)
            disc_loss = real_loss + fake_loss

            gen_loss = loss_fn(tf.ones_like(fake_output), fake_output)

        gradients_of_discriminator = tape.gradient(disc_loss, discriminator.trainable_variables)
        gradients_of_generator = tape.gradient(gen_loss, generator.trainable_variables)
        del tape

        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        gan_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

        return disc_loss, gen_loss

    #Cria os diretórios
    checkpoint_dir = './training_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs('./generated_images', exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=gan_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    # Treinamento
    for epoch in range(EPOCHS):
        for batch in dataset:
            disc_loss, gen_loss = train_step(batch)
        print(f"Epoch {epoch + 1}, Discriminator Loss: {disc_loss:.4f}, Generator Loss: {gen_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            num_examples = 16
            noise = tf.random.normal([num_examples, LATENT_DIM])
            generated_images = generator(noise, training=False)
            generated_images = (generated_images + 1) / 2.0

            plt.figure(figsize=(8, 8))
            for i in range(num_examples):
                plt.subplot(4, 4, i + 1)
                plt.imshow(generated_images[i])
                plt.axis('off')
            plt.savefig(f'./generated_images/epoch_{epoch+1}.png')
            plt.show()

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

    #Gera as imagens e as exibe
    num_final_images = 64
    final_noise = tf.random.normal([num_final_images, LATENT_DIM])
    final_generated_images = generator(final_noise, training=False)
    final_generated_images = (final_generated_images + 1) / 2.0

    plt.figure(figsize=(10, 10))
    for i in range(num_final_images):
        plt.subplot(8, 8, i + 1)
        plt.imshow(final_generated_images[i])
        plt.axis('off')
    plt.show()
