from sklearn.svm import SVC

def get_svm_model():
    return SVC(kernel="rbf", probability=True)