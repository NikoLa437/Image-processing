# import libraries here
import gc
import os
import numpy as np
import cv2  # OpenCV
from sklearn.svm import SVC  # SVM klasifikator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier  # KNN
from joblib import dump, load
import matplotlib
import matplotlib.pyplot as plt

# prikaz vecih slika
matplotlib.rcParams['figure.figsize'] = 16, 12
from imutils import face_utils
import argparse
import imutils
import dlib


def loadArrayFromImages():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./serialization_folder/shape_predictor_68_face_landmarks.dat')

    train_array = []
    train_labels = []
    train_dir = './dataset/slike'
    j = 0
    img_path = 0
    for img_name in os.listdir(train_dir):
        img_path = os.path.join(train_dir, img_name)
        img_path_splited = img_path.split('_')
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        rects = detector(img, 1)
        if (len(rects) < 1):
            continue
        shape = predictor(img, rects[0])
        shape = face_utils.shape_to_np(shape)
        print(j)
        j = j + 1
        (x, y, w, h) = cv2.boundingRect(np.array([shape[1:17]]))
        pf = w * h
        image2 = img
        if (x < 0):
            if (y < 0):
                image2 = img[0: y + h - 25, 0 + 20:x + w - 35]
            else:
                image2 = img[y: y + h - 25, 0 + 20:x + w - 35]
        elif (y < 0):
            image2 = img[0: y + h - 25, x + 20:x + w - 35]
        else:
            image2 = img[y: y + h - 25, x + 20:x + w - 35]
        # plt.imshow(image2, 'gray')  # prikazivanje slike
        # plt.show()
        parametar1 = np.mean(image2)

        # print("PARAMETAR1: " + str(parametar1))

        # povrsina nosa

        (x1, y1, w1, h1) = cv2.boundingRect(np.array([shape[28:38]]))
        pn = w1 * h1
        parametar2 = pn / pf * 600
        print("PARAMETAR2: " + str(parametar2))

        # povrsina usana

        (x1, y1, w1, h1) = cv2.boundingRect(np.array([shape[49:60]]))
        pn = w1 * h1
        parametar3 = pn / pf * 1500
        print("PARAMETAR3: " + str(parametar3))

        # debljina trebavica

        (x1, y1, w1, h1) = cv2.boundingRect(np.array([shape[18:22]]))
        pn = h1
        parametar4 = h1 / h * 1300
        # print("PARAMETAR4: " + str(parametar4))

        # odnos faece

        (x1, y1, w1, h1) = cv2.boundingRect(np.array([shape[1:27]]))
        pn = h1
        parametar5 = w1 / h1 * 100
        # print("PARAMETAR5: " + str(parametar5))

        # debljina trebavica

        (x1, y1, w1, h1) = cv2.boundingRect(np.array([shape[37:40]]))
        pn = w1 * h1
        parametar6 = pn / pf * 15000
        # print("PARAMETAR6: " + str(parametar6))

        # odnos duzine i sirine nosa
        (x1, y1, w1, h1) = cv2.boundingRect(np.array([shape[28:36]]))
        pn = w1 / h1
        parametar7 = pn * 150
        print("PARAMETAR7: " + str(parametar7))
        array = []
        array.append(parametar1)
        array.append(parametar2)
        array.append(parametar3)
        #array.append(parametar4)
        #array.append(parametar5)
        #array.append(parametar6)
        array.append(parametar7)
        if(img_path_splited[2]=='0' or img_path_splited[2]=='1'):
            train_array.append(array)
            train_labels.append(img_path_splited[2])

        # print(num)

    return train_array, train_labels

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)


def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx * ny))


def getAge(age):
    age = int(age)
    if 0 <= age <= 8:
        return 5
    elif 9 <= age <= 17:
        return 13
    elif 18 <= age <= 26:
        return 22
    elif 27 <= age <= 35:
        return 31
    elif 36 <= age <= 44:
        return 40
    elif 45 <= age <= 53:
        return 49
    elif 54 <= age <= 62:
        return 58
    elif 63 <= age <= 71:
        return 67
    elif 72 <= age <= 80:
        return 76
    elif 81 <= age <= 89:
        return 85
    elif 90 <= age <= 98:
        return 94
    elif 99 <= age <= 107:
        return 103
    elif 108 <= age <= 116:
        return 112
    return 27


def load_images_and_labels_for_age():
    train_dir = './dataset/slike'
    images = []
    labels_for_age = []
    i = 0

    for img_name in os.listdir(train_dir):
        img_path = os.path.join(train_dir, img_name)
        print(img_path)
        img_path_splited = img_path.split('_')
        if i % 8 == 0:
            images.append(load_image(img_path))
            #age= img_path_splited[0].split('/')[3]
            labels_for_age.append(img_path_splited[0].split('/')[3])
        i = i + 1

    return images, labels_for_age


def load_images_and_labels_for_race():
    train_dir = './dataset/slike'
    images = []
    labels_for_race = []
    race0 = 0
    race1 = 0
    race2 = 0
    race3 = 0
    race4 = 0
    i = 0
    j = 0
    k = 0
    l = 0

    for img_name in os.listdir(train_dir):
        img_path = os.path.join(train_dir, img_name)
        img_path_splited = img_path.split('_')
        if img_path_splited[2] == '0':
            # if i % 6 == 0:
            images.append(load_image(img_path))
            labels_for_race.append(img_path_splited[2])
            race0 = race0 + 1
            i = i + 1
        elif img_path_splited[2] == '1':
            # if j % 3 == 0 or j % 17 == 0:
            images.append(load_image(img_path))
            labels_for_race.append(img_path_splited[2])
            race1 = race1 + 1
            j = j + 1
        elif img_path_splited[2] == '2':
            # if k % 2 == 0:
            images.append(load_image(img_path))
            labels_for_race.append(img_path_splited[2])
            race2 = race2 + 1
            k = k + 1
        elif img_path_splited[2] == '3':
            # if l % 3 == 0 or l % 7 == 0:
            images.append(load_image(img_path))
            labels_for_race.append(img_path_splited[2])
            race3 = race3 + 1
            l = l + 1
        elif img_path_splited[2] == '4':
            images.append(load_image(img_path))
            labels_for_race.append(img_path_splited[2])
            race4 = race4 + 1

    return images, labels_for_race, race0, race1, race2, race3, race4


def load_images_and_labels_for_gender():
    train_dir = './dataset/slike'
    images = []
    labels_for_gender = []
    men = 0
    women = 0
    i = 0
    j = 0
    for img_name in os.listdir(train_dir):
        img_path = os.path.join(train_dir, img_name)
        img_path_splited = img_path.split('_')
        if i % 1 == 0:  # svaki drugi
            if j % 222223 != 0 and img_path_splited[1] == '0':  # svaki 11 muskarac preskoci
                images.append(load_image(img_path))
                labels_for_gender.append(img_path_splited[1])
                men = men + 1
            elif img_path_splited[1] == '1':
                images.append(load_image(img_path))
                labels_for_gender.append(img_path_splited[1])
                women = women + 1

            j = j + 1
        i = i + 1

    return images, labels_for_gender, men, women


def train_or_load_age_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija), liste
    labela za svaku fotografiju iz prethodne liste, kao i putanju do foldera u koji treba sacuvati model nakon sto se
    istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju
    # return 
    #PRIVREMENO STOJI AGE
    #print("IMAGES: " + str(len(images)))
    #print("LABELS: " + str(len(labels)))

    model = None
    return model
    #model = load('./serialization_folder/clf_svm_age.joblib')

    if model is None:
        images, labels = load_images_and_labels_for_age()
        print("IMAGES: " + str(len(images)))
        print("LABELS: " + str(len(labels)))
        nbins = 5  # broj binova
        cell_size = (12, 12)  # broj piksela po celiji
        block_size = (4, 4)  # broj celija po bloku

        imagesForTrain = []
        for img in images:
            hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                              img.shape[0] // cell_size[0] * cell_size[0]),
                                    _blockSize=(block_size[1] * cell_size[1],
                                                block_size[0] * cell_size[0]),
                                    _blockStride=(cell_size[1], cell_size[0]),
                                    _cellSize=(cell_size[1], cell_size[0]),
                                    _nbins=nbins)
            imagesForTrain.append(hog.compute(img))

        images = None
        print('Zavrsen hog')
        gc.collect()
        print(len(imagesForTrain))
        x = np.array(imagesForTrain)
        y = np.array(labels)
        imagesForTrain = None
        labels = None
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
        x = None
        y = None
        gc.collect()
        print('Train shape: ', x_train.shape, y_train.shape)
        print('Test shape: ', x_test.shape, y_test.shape)

        x_train = reshape_data(x_train)
        x_test = reshape_data(x_test)
        gc.collect()

        # probati sa RBF i probati za godine SVC
        clf_svm = SVC(kernel='linear', probability=True, verbose=True, cache_size=900)
        clf_svm.fit(x_train, y_train)
        y_train_pred = clf_svm.predict(x_train)
        y_test_pred = clf_svm.predict(x_test)
        print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
        print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))

        #clf_lr = LogisticRegression()
        #clf_lr = clf_lr.fit(x_train,y_train)
        #y_train_pred = clf_lr.predict(x_train)
        #y_test_pred = clf_lr.predict(x_test)
        #print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
        #print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))

        #probati knn za gender
        #clf_knn = KNeighborsClassifier(n_neighbors=3)
        #clf_knn = clf_knn.fit(x_train, y_train)
        #y_train_pred = clf_knn.predict(x_train)
        #y_test_pred = clf_knn.predict(x_test)
        #print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
        #print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))

        model = clf_svm
        dump(clf_svm, 'serialization_folder/clf_svm_age1.joblib')

    return model



def train_or_load_gender_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija), liste
    labela za svaku fotografiju iz prethodne liste, kao i putanju do foldera u koji treba sacuvati model nakon sto se
    istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju

    model = load('./serialization_folder/clfsvm.joblib')

    if model is None:
        images, labels, men, women = load_images_and_labels_for_gender()
        print("MEN: " + str(men))
        print("WOMEN: " + str(women))
        nbins = 3  # broj binova
        cell_size = (12, 12)  # broj piksela po celiji
        block_size = (4, 4)  # broj celija po bloku

        imagesForTrain = []
        for img in images:
            hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                              img.shape[0] // cell_size[0] * cell_size[0]),
                                    _blockSize=(block_size[1] * cell_size[1],
                                                block_size[0] * cell_size[0]),
                                    _blockStride=(cell_size[1], cell_size[0]),
                                    _cellSize=(cell_size[1], cell_size[0]),
                                    _nbins=nbins)
            imagesForTrain.append(hog.compute(img))

        images = None
        print('Zavrsen hog')
        gc.collect()
        print(len(imagesForTrain))
        x = np.array(imagesForTrain)
        y = np.array(labels)
        imagesForTrain = None
        labels = None
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        x = None
        y = None
        gc.collect()
        print('Train shape: ', x_train.shape, y_train.shape)
        print('Test shape: ', x_test.shape, y_test.shape)

        x_train = reshape_data(x_train)
        x_test = reshape_data(x_test)
        gc.collect()

        # probati sa RBF i probati za godine SVC
        clf_svm = SVC(kernel='linear', probability=True, verbose=True, cache_size=900)
        clf_svm.fit(x_train, y_train)
        y_train_pred = clf_svm.predict(x_train)
        y_test_pred = clf_svm.predict(x_test)
        print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
        print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))

        ''' clf_lr = LogisticRegression()
        clf_lr = clf_lr.fit(x_train,y_train)
        y_train_pred = clf_lr.predict(x_train)
        y_test_pred = clf_lr.predict(x_test)
        print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
        print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))'''

        ''' probati knn za gender
        clf_knn = KNeighborsClassifier(n_neighbors=5)
        clf_knn = clf_knn.fit(x_train, y_train)
        y_train_pred = clf_knn.predict(x_train)
        y_test_pred = clf_knn.predict(x_test)
        print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
        print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))'''

        model = clf_svm
        dump(clf_svm, 'clfsvm.joblib')

    return model


def train_or_load_race_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija), liste
    labela za svaku fotografiju iz prethodne liste, kao i putanju do foldera u koji treba sacuvati model nakon sto se
    istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju

    model = None
    return model

    model = load('./serialization_folder/clf_knn.joblib')

    if model is None:
        images, labels = loadArrayFromImages()
        print("IMAGES: " + str(len(images)))
        print("LABELS: " + str(len(labels)))

        gc.collect()
        print(len(images))
        x = np.array(images)
        y = np.array(labels)
        imagesForTrain = None
        labels = None
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        x = None
        y = None
        gc.collect()
        print('Train shape: ', x_train.shape, y_train.shape)
        print('Test shape: ', x_test.shape, y_test.shape)

        # x_train = reshape_data(x_train)
        # x_test = reshape_data(x_test)
        gc.collect()

        # probati sa RBF i probati za godine SVC
        # clf_svm = SVC(kernel='linear', probability=True, verbose=True, cache_size=900)
        # clf_svm.fit(x_train, y_train)
        # y_train_pred = clf_svm.predict(x_train)
        # y_test_pred = clf_svm.predict(x_test)
        # print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
        # print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))

        # clf_lr = LogisticRegression()
        # clf_lr = clf_lr.fit(x_train,y_train)
        # y_train_pred = clf_lr.predict(x_train)
        # y_test_pred = clf_lr.predict(x_test)
        # print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
        # print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))

        # probati knn za gender
        clf_knn = KNeighborsClassifier(n_neighbors=7)
        clf_knn = clf_knn.fit(x_train, y_train)
        y_train_pred = clf_knn.predict(x_train)
        y_test_pred = clf_knn.predict(x_test)
        print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
        print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))

        model = clf_knn
        dump(clf_knn, 'clf_knn2.joblib')

        clf_knn = KNeighborsClassifier(n_neighbors=9)
        clf_knn = clf_knn.fit(x_train, y_train)
        y_train_pred = clf_knn.predict(x_train)
        y_test_pred = clf_knn.predict(x_test)
        print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
        print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))

        model = clf_knn
        dump(clf_knn, 'clf_knn3.joblib')

        clf_knn = KNeighborsClassifier(n_neighbors=12)
        clf_knn = clf_knn.fit(x_train, y_train)
        y_train_pred = clf_knn.predict(x_train)
        y_test_pred = clf_knn.predict(x_test)
        print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
        print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))

        model = clf_knn
        dump(clf_knn, 'clf_knn4.joblib')

        clf_knn = KNeighborsClassifier(n_neighbors=15)
        clf_knn = clf_knn.fit(x_train, y_train)
        y_train_pred = clf_knn.predict(x_train)
        y_test_pred = clf_knn.predict(x_test)
        print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
        print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))

        model = clf_knn
        dump(clf_knn, 'clf_knn5.joblib')

        clf_knn = KNeighborsClassifier(n_neighbors=3)
        clf_knn = clf_knn.fit(x_train, y_train)
        y_train_pred = clf_knn.predict(x_train)
        y_test_pred = clf_knn.predict(x_test)
        print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
        print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))

        model = clf_knn
        dump(clf_knn, 'clf_knn6.joblib')

        clf_knn = KNeighborsClassifier(n_neighbors=8)
        clf_knn = clf_knn.fit(x_train, y_train)
        y_train_pred = clf_knn.predict(x_train)
        y_test_pred = clf_knn.predict(x_test)
        print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
        print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))

        model = clf_knn
        dump(clf_knn, 'clf_knn8.joblib')

    return model


def predict_age(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje godina i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati godine.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje godina
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati godine lica
    :return: <Int> Prediktovanu vrednost za goinde  od 0 do 116
    """
    return 28
    # TODO - Prepoznati ekspresiju lica i vratiti njen naziv (kao string, iz skupa mogucih vrednosti)
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor('./serialization_folder/shape_predictor_68_face_landmarks.dat')
    array = []
    rects = detector(img, 1)
    # for (i, rect) in enumerate(rects):

    # shape = predictor(img, rect)
    # shape predstavlja 68 koordinata
    # shape = face_utils.shape_to_np(shape)  # konverzija u NumPy niz
    if len(rects)<1:
        return 27
    (x, y, w, h) = face_utils.rect_to_bb(rects[0])
    # crtanje pravougaonika oko detektovanog lica
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print(image_path)
    # print("X: " + str(x) + "   Y:" + str(y) + "   w:" + str(w) + "    h" + str(h))
    if (x < 0):
        if (y < 0):
            image2 = img[0:y + h, 0:x + w]
        else:
            image2 = img[y: y + h, 0:x + w]
    elif (y < 0):
        image2 = img[0: y + h, x:x + w]
    else:
        image2 = img[y: y + h, x:x + w]

    img = resize_region(image2)
    # plt.imshow(img, 'gray')  # prikazivanje slike
    # plt.show()
    nbins = 5  # broj binova
    cell_size = (12, 12)  # broj piksela po celiji
    block_size = (4, 4)  # broj celija po bloku

    predicted = 0
    if img is not None:
        hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                          img.shape[0] // cell_size[0] * cell_size[0]),
                                _blockSize=(block_size[1] * cell_size[1],
                                            block_size[0] * cell_size[0]),
                                _blockStride=(cell_size[1], cell_size[0]),
                                _cellSize=(cell_size[1], cell_size[0]),
                                _nbins=nbins)
        array.append(hog.compute(img))

        x = np.array(array)

        x_train = reshape_data(x)
        print('Train shape: ', x_train.shape)

        predicted = trained_model.predict(x_train)

    # plt.imshow(image2, 'gray')  # prikazivanje slike
    # plt.show()
    print(str(predicted[0]))
    if(predicted[0]=='1' or predicted[0]=='2' or predicted[0]=='3' or predicted[0]=='4' or predicted[0]=='5'):
        #print('test')
        return 5
    return predicted[0]




def predict_gender(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje pola na osnovu lica i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati pol.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <Int>  Prepoznata klasa pola (0 - musko, 1 - zensko)
    """
    #TODO: izmeniti za gender pre stavljanja na drive
    # TODO - Prepoznati ekspresiju lica i vratiti njen naziv (kao string, iz skupa mogucih vrednosti)
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor('./serialization_folder/shape_predictor_68_face_landmarks.dat')
    array = []
    rects = detector(img, 1)
    # for (i, rect) in enumerate(rects):

    # shape = predictor(img, rect)
    # shape predstavlja 68 koordinata
    # shape = face_utils.shape_to_np(shape)  # konverzija u NumPy niz
    if len(rects)<1:
        return 0
    (x, y, w, h) = face_utils.rect_to_bb(rects[0])
    # crtanje pravougaonika oko detektovanog lica
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print(image_path)
    # print("X: " + str(x) + "   Y:" + str(y) + "   w:" + str(w) + "    h" + str(h))
    if (x < 0):
        if (y < 0):
            image2 = img[0:y + h, 0:x + w]
        else:
            image2 = img[y: y + h, 0:x + w]
    elif (y < 0):
        image2 = img[0: y + h, x:x + w]
    else:
        image2 = img[y: y + h, x:x + w]

    img = resize_region(image2)
    # plt.imshow(img, 'gray')  # prikazivanje slike
    # plt.show()
    nbins = 3  # broj binova
    cell_size = (12, 12)  # broj piksela po celiji
    block_size = (4, 4)  # broj celija po bloku

    predicted = 0
    if img is not None:
        hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                          img.shape[0] // cell_size[0] * cell_size[0]),
                                _blockSize=(block_size[1] * cell_size[1],
                                            block_size[0] * cell_size[0]),
                                _blockStride=(cell_size[1], cell_size[0]),
                                _cellSize=(cell_size[1], cell_size[0]),
                                _nbins=nbins)
        array.append(hog.compute(img))

        x = np.array(array)

        x_train = reshape_data(x)
        print('Train shape: ', x_train.shape)

        predicted = trained_model.predict(x_train)

    # plt.imshow(image2, 'gray')  # prikazivanje slike
    # plt.show()
    print(str(predicted[0]))
    return predicted[0]


def resize_region(region):
    if region.shape[0] != 0 and region.shape[1] != 0:
        return cv2.resize(region, (200, 200), interpolation=cv2.INTER_NEAREST)


def predict_race(trained_model, image_path):
    """

    Procedura prima objekat istreniranog modela za prepoznavanje rase lica i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati rasu.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <Int>  Prepoznata klasa (0 - Bela, 1 - Crna, 2 - Azijati, 3- Indijci, 4 - Ostali)
    """
    #TODO: Zakomentarisati return pre stavljanja na drive
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2LAB)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./serialization_folder/shape_predictor_68_face_landmarks.dat')
    print(image_path)
    rects = detector(img, 1)
    if (len(rects) < 1):
        return 0
    shape = predictor(img, rects[0])
    shape = face_utils.shape_to_np(shape)
    #(x, y, w, h) = face_utils.rect_to_bb(rects[0])
    #pf = w * h
    (x, y, w, h) = cv2.boundingRect(np.array([shape[1:17]]))
    pf = w * h
    image2 = img
    if (x < 0):
        if (y < 0 and h<25 and w<35):
            image2 = img[0: y + h - 25, 0 + 20:x + w - 35]
        elif h<25 and w<35:
            image2 = img[y: y + h - 25, 0 + 20:x + w - 35]
        else:
            return 0
    elif (y < 0 and h>25 and w>35):
        image2 = img[0: y + h - 25, x + 20:x + w - 35]
    elif h>25 and w>35:
        image2 = img[y: y + h - 25, x + 20:x + w - 35]
    else:
        return 0
    # plt.imshow(image2, 'gray')  # prikazivanje slike
    # plt.show()
    parametar1 = np.mean(image2)
    print(parametar1)
    ret_val=0
    if(parametar1>140):
        ret_val= 0
    else:
        ret_val=1
    print(ret_val)
    # print("PARAMETAR1: " + str(parametar1))
    return ret_val
