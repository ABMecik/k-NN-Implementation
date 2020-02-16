#Date: 2.16.2020
#Description: In this code, the variation of the k-NN algorithm in different distance metrics and k (neighbor number) value is examined.
#In this process, iris dataset was used.


"""
It is benefited from matplotlib libraries to draw numpy and graphics for the dataset.
The Skearn library was imported separately in the section where the comparison was made.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


"""
This class implements all relevant operations of the k-NN algorithm.
It was designed in a similar way to the sklearn.neighbors.KNeighborsClassifier library.
While defining Class, it takes the number of neighbors (k) and distance metric as standard.
The default k value is set to 3 and the default distance metric is set to euclidean.
"""
class KNNClassifier:
    def __init__(self, k=3, method="euclidean"):
        self.k=k
        self.method=method

    """
    This function imports training data.
    """
    def fit(self, X_train, y_train):
        self.X_train=X_train
        self.y_train=y_train


    """
    The help of this function of distance function finds the distance of the desired point to the data in the training data.
    The distances found are listed.
    The closest neighbor is taken as much as the number of neighbors (k) and the most affiliate value is selected.
    In this case, the k value should be an odd number in order to prevent equality.
    However, when an even number is entered, the system selects the closest value in case of equality.
    The system takes a dataset as input and returns the prediction list.
    It can work in different data sets except for the draw and split_list method.
    """
    def predict(self, X_test):
        predictions = []
        for i in X_test:
            distances = []
            for k in range(len(self.X_train)):
                distances.append((self.y_train[k],self.distance(i,self.X_train[k])))#(i,self.X_train[k])
            distances = sorted(distances, key=lambda x: x[1])
            neighbors = distances[:self.k]
            predictions.append(self.response(neighbors))
        return predictions

    """
    In accordance with the desired metric, the distance between the given data is calculated.
    """
    def distance(self, data1, data2):
        length = len(data1)
        if self.method == "euclidean":
            distance = 0
            for x in range(length):
                distance += pow((float(data1[x])-float(data2[x])),2)
            return pow(distance,1/2)
        elif self.method == "manhattan":
            distance = 0
            for x in range(length):
                distance += abs(float(data1[x])-float(data2[x]))
            return distance

    """
    It counts the possible options in the given data and returns the option that gives the highest count value.
    """
    def response(self, neighbors):
        dss={}
        for (k,l) in neighbors:
            if k in dss:
                dss[k] = dss[k]+1
            else:
                dss[k] = 1
        v = list(dss.values())
        k = list(dss.keys())
        return k[v.index(max(v))]

    """
    It turns accuracy rate by comparing predictions and real results.
    """
    def getAccuracy(self, y_prend, y_test):
        length = len(y_prend)
        crr=0
        for i in range(length):
            if y_prend[i] == y_test[i]:
                crr += 1
        if crr == 0:
            return 0
        else:
            return (crr/length)*100

    """
    It turns error count by comparing predictions and real results.
    """
    def getErrorcount(self, y_prend, y_test):
        length = len(y_prend)
        crr=0
        for i in range(length):
            if y_prend[i] != y_test[i]:
                crr += 1
        return crr


    """
    With the help of the Matplotlib library, he draws decision boundaries.
    This section is designed to operate only in the iris dataset and in line with certain parameters, unlike other sections in the k-NN algorithm.
    It takes a single subplot as input.
    """
    #only for iris dataset
    def draw(self, ax):
        iris_dict = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
        X_x, X_y = self.split_list(self.X_train)

        d1_min,d2_min,d1_max,d2_max = min(X_x)-.3,min(X_y)-.3,max(X_x)+.3,max(X_y)+.3
        resolution=.03

        map_x, map_y = np.meshgrid(np.arange(d1_min, d1_max, resolution), np.arange(d2_min, d2_max, resolution))
        res = self.predict(np.c_[map_x.ravel(), map_y.ravel()])

        y, boundary = [], []
        [boundary.append(iris_dict[i]) for i in res]
        boundary = np.array(boundary).reshape(map_x.shape)
        [y.append(iris_dict[i]) for i in self.y_train]
        y = np.array(y)

        ax.set_title("(Method = " + self.method + ") (k = " + str(self.k) + ")", fontsize=14)
        ax.axis('tight')
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_xlim(map_x.min(), map_x.max())
        ax.set_ylim(map_y.min(), map_y.max())
        ax.pcolormesh(map_x, map_y, boundary, cmap=ListedColormap(['#FFAAAA', '#AAFFAA','#AAAAFF']))
        ax.scatter(X_x, X_y, c=y, cmap=ListedColormap(['#FF0000', '#00FF00','#0000FF']), marker="o", s=12, edgecolors=(0, 0, 0, 1), linewidths=.5)

    """
    Subdividing to dataset.
    """
    def split_list(self, X):
        x=[]
        y=[]
        for [k1,k2] in X:
            x.append(float(k1))
            y.append(float(k2))
        return x,y

    def setK(self, k):
        self.k=k

"""
Preprocess
The process of reading and splitting the data set is applied.
"""
class Data:
    def __init__(self):
        super(self)


    """
    It reads the data from the desired file.
    """
    @classmethod
    def readlocal(cls, path):
        all=[]
        with open(path, 'r', encoding='utf8') as f:
            for i in f.readlines():
                arr = i.split(',')
                if arr[-1][-1] == "\n":
                    arr[-1] = arr[-1][:-1]
                all.append(arr)
        return np.array(all)

    """
    It divides the data based on test|train and input|output.
    The distribution of the data is received from the user.
    The distribution can be done numerically or as a percentage.
    The data is divided into 70% education and 30% test data by default.
    The data is divided numerically as requested in this code. (30 training and 20 test data)
    """
    @classmethod
    def split_data(cls, data, type="percentage", train_size=.7, test_size=.3):

        X_train, X_test, y_train, y_test = [], [], [], []

        vertigo = []
        tp,ls=[],""
        for a in data:
            if a[-1] != ls:
                if tp:
                    vertigo.append(tp)
                tp=[]
            tp.append(a)
            ls=a[-1]
        vertigo.append(tp)

        for t in vertigo:
            if type=="percentage":
                train_size = len(t)*train_size
                test_size = len(t)-train_size
            try:
                train,test = t[:train_size],t[train_size:train_size+test_size]
            except:
                print("Not enough size")
                return None

            for k in train:
                X_train.append(k[:-1])
                y_train.append(k[-1])
            for k in test:
                X_test.append(k[:-1])
                y_test.append(k[-1])

        return X_train, X_test, y_train, y_test

def main():
    try:
        """
        He reads the data with the help of libraries.
        It then separates the data into test and train data.
        """
        data = Data.readlocal(path="iris_data.txt")
        X_train, X_test, y_train, y_test = Data.split_data(data=data,type="numeric",train_size=30,test_size=20)
        """
        Preprocessor
        It takes the 1st and 4th values in the input values of the code.
        These values are particularly distinctive in the iris dataset.
        The main purpose of this process is to increase the working speed of the code.
        """
        X_train = [[x[0],x[3]] for x in X_train]
        X_test = [[x[0],x[3]] for x in X_test]

        k = int(input("Pls enter k value: "))
        print("*" * 75)
    except Exception as error:
        print(error)

    """
    Using the k-NN class, he finds the accuracy value based on the euclidean metric and the k value received from the user.
    Test data is used when calculating accuracy.
    Decision boundaries are drawn.
    """
    knn_euclidean = KNNClassifier(k=k,method="euclidean")#euclidean, manhattan
    knn_euclidean.fit(X_train, y_train)
    y_prend = knn_euclidean.predict(X_test)
    accuracy = knn_euclidean.getAccuracy(y_prend,y_test)
    errorcount = knn_euclidean.getErrorcount(y_prend, y_test)
    print("For k="+str(k)+" and euclidean method || Accuracy : ", accuracy, " Error Count: " + str(errorcount)+"/"+str(len(y_test)))

    """
    Using the k-NN class, he finds the accuracy value based on the manhattan metric and the k value received from the user.
    Test data is used when calculating accuracy.
    Decision boundaries are drawn.
    """
    knn_manhattan = KNNClassifier(k=k, method="manhattan")  # euclidean, manhattan
    knn_manhattan.fit(X_train, y_train)
    y_prend = knn_manhattan.predict(X_test)
    accuracy = knn_manhattan.getAccuracy(y_prend, y_test)
    errorcount = knn_manhattan.getErrorcount(y_prend, y_test)
    print("For k=" + str(k) + " and manhattan method || Accuracy : ", accuracy, " Error Count: " + str(errorcount) + "/" + str(len(y_test)))

    """
    With the Sklearn library, the accuracy rate is calculated using the test data in line with the k value received from the user.
    """
    #Sklearn prediction:
    from sklearn.neighbors import KNeighborsClassifier
    import warnings
    warnings.filterwarnings("ignore")

    knn_sklearn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_sklearn_classifier.fit(X_train, y_train)  # Train
    y_prend = knn_sklearn_classifier.predict(X_test)
    knn_sklearn_acc = KNNClassifier()
    accuracy = knn_sklearn_acc.getAccuracy(y_prend, y_test)
    errorcount = knn_sklearn_acc.getErrorcount(y_prend, y_test)
    print("For sklearn dictionary (k="+str(k)+") || Accuracy : ", accuracy, " Error Count: " + str(errorcount) + "/" + str(len(y_test)))
    #********************************

    try:

        print("*" * 75)
        while True:
            print("0: Exit || 1: Draw Graph || 2: User Prediction")

            ans = int(input("Chose: "))
            print("*" * 75)
            while ans>2 or ans<0:
                print("0: Exit || 1: Draw Graph || 2: User Prediction")
                ans = int(input("Chose: "))
                print("*" * 75)

            if ans == 0:
                break
            elif ans == 1:
                """
                Chart window basic settings.
                Decision boundaries are drawn.
                """
                fig, (ax1, ax2) = plt.subplots(1,2,figsize=(6, 3), dpi=150, facecolor='w', edgecolor='k')
                knn_euclidean.draw(ax1)
                knn_manhattan.draw(ax2)

                Iris_setosa_d = mpatches.Patch(color='#FFAAAA', label='Iris-setosa')
                Iris_versicolor_d = mpatches.Patch(color='#00FF00', label='Iris-versicolor')
                Iris_virginica_d = mpatches.Patch(color='#0000FF', label='Iris-virginica')
                plt.tight_layout()
                plt.legend(bbox_to_anchor=(-0.25, -0.1), loc=9, borderaxespad=0.,handles=[Iris_setosa_d, Iris_versicolor_d, Iris_virginica_d],ncol=3)
                plt.savefig('Description', bbox_inches='tight')

                plt.show()
            elif ans == 2:
                """
                Based on the data received from the user, an estimate is obtained from the trained systems.
                """
                # https://archive.ics.uci.edu/ml/datasets/iris
                sepal_length = float(input("Pls enter sepal length as cm: "))
                sepal_width = float(input("Pls enter sepal length as cm: "))
                petal_length = float(input("Pls enter sepal length as cm: "))
                petal_width = float(input("Pls enter sepal length as cm: "))
                pre_data = [sepal_length, sepal_width, petal_length, petal_width]
                print("Euclidean pretiction: ", knn_euclidean.predict([pre_data])[0])
                print("Manhattan pretiction: ", knn_manhattan.predict([pre_data])[0])
                print("Sklearn pretiction: ", knn_sklearn_classifier.predict([pre_data])[0])
                print("*" * 75)
    except Exception as error:
        print(error)

if __name__ == '__main__':
    main()
