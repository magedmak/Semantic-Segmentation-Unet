from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
import segmentation_model as sm
import matplotlib.pyplot as plt
from utils_func import *
from model_arch import get_model_arch

EPOCHS = 10
BATCH_SIZE = 10
HEIGHT = 256
WIDTH = 256
N_CLASSES = 13

train_folder = "./images/train"
valid_folder = "./images/val"

num_of_training_samples = len(os.listdir(train_folder))
num_of_valid_samples = len(os.listdir(valid_folder))


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable all debugging logs (info, error, and warning messages)

    train_imgs = data_generator(train_folder, batch_size=BATCH_SIZE)
    val_imgs = data_generator(valid_folder, batch_size=BATCH_SIZE)

    TRAIN_STEPS = num_of_training_samples // BATCH_SIZE + 1
    VAL_STEPS = num_of_valid_samples // BATCH_SIZE + 1

    # define model
    model = sm.Unet()

    # save the model or weights (in a checkpoint file) at some interval
    checkpoint = ModelCheckpoint('./output/unet_model.hdf5',
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    # model.summary()
    # get_model_arch(model)

    history = model.fit(train_imgs,
                        epochs=EPOCHS,
                        validation_data=val_imgs,
                        steps_per_epoch=TRAIN_STEPS,
                        validation_steps=VAL_STEPS,
                        callbacks=checkpoint)

    model.load_weights("./output/unet_model.hdf5")
    get_acc(history.history["val_loss"], history.history["val_acc"])  # plot training and validation accuracy

    img_show_num = 1
    imgs, segs = next(val_imgs)  # The next() function returns the next item in an iterator.
    prediction = model.predict(imgs)

    for i in range(img_show_num):
        show_predications(imgs, segs, prediction, i)


def data_generator(path, batch_size=BATCH_SIZE, classes=N_CLASSES):
    files = os.listdir(path)
    while True:
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            imgs = []
            segs = []
            for file in batch_files:
                image, mask = load_images(file, path)
                mask_binned = bin_image(mask)
                labels = get_segmentation_arr(mask_binned, classes)

                imgs.append(image)
                segs.append(labels)

            yield np.array(imgs), np.array(segs)


def show_predications(imgs, segs, prediction, i):
    _p = give_color_to_seg_img(np.argmax(prediction[i], axis=-1))
    _s = give_color_to_seg_img(np.argmax(segs[i], axis=-1))

    pred_img = cv2.addWeighted(imgs[i] / 255, 0.5, _p, 0.5, 0)
    true_img = cv2.addWeighted(imgs[i] / 255, 0.5, _s, 0.5, 0)

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title("Prediction")
    plt.imshow(pred_img)
    plt.axis("off")
    plt.subplot(122)
    plt.title("Original")
    plt.imshow(true_img)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("./output/pred_" + str(i+1) + ".png", dpi=150)
    plt.show()


def get_acc(loss, acc):
    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plt.title("Val. Loss")
    plt.plot(loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(212)
    plt.title("Val. Accuracy")
    plt.plot(acc)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig("./output/acc.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
