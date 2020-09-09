import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"

def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    plt.scatter(X[:,0],X[:,1],2,y)
    plt.savefig('1_f_2D_Scatter_plot.png')
    ### END YOUR CODE

def visualize_result(X, y, W):
    '''This function is used to plot the sigmoid model after training.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].

    Returns:
        No return. Save the plot to 'train_result_sigmoid.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    plt.scatter(X[:,0],X[:,1],2,y)
    x_points = np.linspace(-0.35,-0.25,100)
    y_points = - (W[0]/(1e-10 + W[2])) - x_points * (W[1]/(1e-10 + W[2]))
    plt.plot(x_points,y_points,'-r')
    plt.savefig('3_d.png')
    ### END YOUR CODE

def visualize_result_multi(X, y, W):
    '''This function is used to plot the softmax model after training.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 0,1,2.
        W: An array of shape [n_features, 3].

    Returns:
        No return. Save the plot to 'train_result_softmax.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    W = W.T
    plt.figure()
    plt.scatter(X[:,0],X[:,1],2,y)
    x0_points = np.linspace(-5,5,100)
    y0_points = - (W[0,0]/W[0,2]) - x0_points * (W[0,1]/W[0,2])
    plt.plot(x0_points, y0_points,'-r')
    x1_points = np.linspace(-5,5,100)
    y1_points = - (W[1,0]/W[1,2]) - x1_points * (W[1,1]/W[1,2])
    plt.plot(x1_points, y1_points,'-b')
    x2_points = np.linspace(-5,5,100)
    y2_points = - (W[2,0]/W[2,2]) - x2_points * (W[2,1]/W[2,2])
    plt.plot(x2_points, y2_points,'-g')
    plt.savefig('4_d.png')
    ### END YOUR CODE

def main():
    # ------------Data Preprocessing------------
    # Read data for training.

    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)


    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)

    ####### For binary case, only use data from '1' and '2'
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training.
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class.
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    data_shape= train_y.shape[0]

    # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)


   # ------------Logistic Regression Sigmoid Case------------

   ##### Check GD, SGD, BGD
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

    logisticR_classifier.fit_GD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_BGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_BGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_BGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))


    # Explore different hyper-parameters.
    ### YOUR CODE HERE

#    learning_rate = [0.03, 0.1, 0.3]
#    max_iter = [100, 1000 ]
    learning_rate = [0.3]
    max_iter = [1000]

    best_model_acc = 0
    best_params = np.zeros(train_X.shape[0])
    best_model = None
    for lr in learning_rate:
        for iter in max_iter:
            logisticR_classifier = logistic_regression(learning_rate=lr, max_iter=iter)

            logisticR_classifier.fit_GD(train_X, train_y)
            if logisticR_classifier.score(valid_X, valid_y) > best_model_acc :
                best_model_acc = logisticR_classifier.score(valid_X, valid_y)
                best_params = logisticR_classifier.get_params()
                best_model = logisticR_classifier

            logisticR_classifier.fit_BGD(train_X, train_y, 10)
            if logisticR_classifier.score(valid_X, valid_y) > best_model_acc :
                best_model_acc = logisticR_classifier.score(valid_X, valid_y)
                best_params = logisticR_classifier.get_params()
                best_model = logisticR_classifier

            logisticR_classifier.fit_SGD(train_X, train_y)
            if logisticR_classifier.score(valid_X, valid_y) > best_model_acc :
                best_model_acc = logisticR_classifier.score(valid_X, valid_y)
                best_params = logisticR_classifier.get_params()
                best_model = logisticR_classifier

            print(iter, lr)
            print(best_model_acc, best_params)


    print(best_model_acc, best_params)
    ### END YOUR CODE

    # Visualize the your 'best' model after training.
    ### YOUR CODE HERE
    visualize_result(train_X[:, 1:3], train_y, best_params)
    ### END YOUR CODE

    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE
    test_raw_data, test_labels = load_data(os.path.join(data_dir, test_filename))
    test_X_all = prepare_X(test_raw_data)
    test_y_all, test_idx = prepare_y(test_labels)
    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    test_y[np.where(test_y == 2)] = -1

    print("----------------------------------------------")
    print("Best model testing accuracy logistic regression: ", best_model.score(test_X,test_y))
    print("----------------------------------------------")

    ### END YOUR CODE


    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all

    #########  BGD for multiclass Logistic Regression
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_BGD(train_X, train_y, 10)
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    # Explore different hyper-parameters.
    ### YOUR CODE HERE
#    learning_rate = [0.03, 0.1, 0.3]
#    max_iter = [100, 1000 ]
    learning_rate = [0.3]
    max_iter = [100]

    best_model_acc_multi = 0
    best_params_multi = np.zeros(train_X.shape[0])
    best_model_multi = None
    for lr in learning_rate:
        for iter in max_iter:
            logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=lr, max_iter=iter, k = 3)

            logisticR_classifier_multiclass.fit_BGD(train_X, train_y, 10)
            if logisticR_classifier_multiclass.score(train_X, train_y) > best_model_acc_multi :
                best_model_acc_multi = logisticR_classifier_multiclass.score(train_X, train_y)
                best_params_multi = logisticR_classifier_multiclass.get_params()
                best_model_multi = logisticR_classifier_multiclass

    print(best_model_acc_multi)
    ### END YOUR CODE

    # Visualize the your 'best' model after training.
    visualize_result_multi(train_X[:, 1:3], train_y, best_params_multi)

    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE
    test_X = test_X_all
    test_y = test_y_all
    print("----------------------------------------------")
    print("Best model testing accuracy multiclass : ", best_model_multi.score(test_X,test_y))
    print("----------------------------------------------")

    ### END YOUR CODE


    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2'

    #####  set labels to 0,1 for softmax classifer
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0

    ###### First, fit softmax classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=10000, k = 2)
    logisticR_classifier_multiclass.fit_BGD_Convergence(train_X,train_y,10)
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    ### END YOUR CODE






    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    #####       set lables to -1 and 1 for sigmoid classifer
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1

    ###### Next, fit sigmoid classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=10000)

    logisticR_classifier.fit_BGD_Convergence(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    ### END YOUR CODE


    ################Compare and report the observations/prediction accuracy


    '''
    Explore the training of these two classifiers and monitor the graidents/weights for each step.
    Hint: First, set two learning rates the same, check the graidents/weights for the first batch in the first epoch. What are the relationships between these two models?
    Then, for what leaning rates, we can obtain w_1-w_2= w for all training steps so that these two models are equivalent for each training step.
    '''
    ### YOUR CODE HERE
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    #####       set lables to -1 and 1 for sigmoid classifer
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1

    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=3)
    logisticR_classifier.fit_BGD(train_X, train_y, train_X.shape[0])
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0

    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=3, k = 2)
    logisticR_classifier_multiclass.fit_BGD(train_X,train_y,train_X.shape[0])
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    ### END YOUR CODE

    # ------------End------------


if __name__ == '__main__':
    main()


