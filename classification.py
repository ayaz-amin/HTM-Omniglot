import cv2
import numpy as np
import copy
from model import OmniglotHTM

nrun = 20 
fname_label = 'class_labels.txt' 

def load_data(file_path):
    image = cv2.imread(file_path, 0)
    image = cv2.resize(image, (64, 64))
    image = cv2.bitwise_not(image)
    return image.reshape(-1)

def classification_run(folder, epochs):
    model = OmniglotHTM(64 * 64, 4096)

    root = 'all_runs'
    file_path = root + '/' + folder
    with open(file_path + '/' + fname_label) as f:
        content = f.read().splitlines()

        pairs = [line.split() for line in content]

        test_files  = [pair[0] for pair in pairs]
        train_files = [pair[1] for pair in pairs]
        answers_files = copy.copy(train_files)

        test_files.sort()
        train_files.sort()	
	
        ntrain = len(train_files)
        ntest = len(test_files)

        # load the images (and, if needed, extract features)
        train_items = [load_data(root + '/' + f) for f in train_files]
        test_items  = [load_data(root + '/' + f) for f in test_files]

        for _ in range(epochs):
            model.learn(train_items)

        # compute the error rate
        correct = 0.0
        for i in range(ntest):
            y_pred = model.predict(test_items[i])
            if train_files[y_pred] == answers_files[i]:
                correct += 1.0
            pcorrect = 100 * correct / ntest
            perror = 100 - pcorrect
            return perror

if __name__ == "__main__":
    print('One-shot classification')
    perror = np.zeros(nrun)
    for r in range(1, nrun+1):
        rs = str(r)
        if len(rs) == 1:
            rs = '0' + rs		
            perror[r-1] = classification_run('run' + rs, epochs=10)
            print(" run " + str(r) + " (error " + str(perror[r-1]) + "%)")		
    total = np.mean(perror)
    print(" average error " + str(total) + "%")