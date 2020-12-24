from DataReader import prepare_data
from model import Model

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"

def main():
    # ------------Data Preprocessing------------
    train_X, train_y, valid_X, valid_y, train_valid_X, train_valid_y, test_X, test_y = prepare_data(data_dir, train_filename, test_filename)

    # ------------Kernel Logistic Regression Case------------
    ### YOUR CODE HERE
    # Run your kernel logistic regression model here
    learning_rate = [0.01]
    max_epoch = [50]
    batch_size = [32, 128]
    sigma = [1.0, 2.0, 3.0]
    
    ans = []
    
    for lr in learning_rate:
        for ep in max_epoch:
            for bs in batch_size:
                for sig in sigma:

                    model = Model('Kernel_LR', train_X.shape[0], sig)
                    model.train(train_X, train_y, valid_X, valid_y, ep, lr, bs)
                
                    model = Model('Kernel_LR', train_valid_X.shape[0], sig)
                    model.train(train_valid_X, train_valid_y, None, None, ep, lr, bs)
                    score = model.score(test_X, test_y)
                    print("score = {} in test set.\n".format(score))
                        
                    ans.append((lr,ep,bs,sig,score))

    for a in ans:
        print(a)                    
    ### END YOUR CODE

    # ------------RBF Network Case------------
    ### YOUR CODE HERE
    # Run your radial basis function network model here
    hidden_dims = [16, 32, 64] 
    learning_rate = [0.01]
    max_epoch = [50]
    batch_size = [128]
    sigma = [0.1, 1.0, 3.0]
    
    ans = []
    
    for lr in learning_rate:
        for ep in max_epoch:
            for bs in batch_size:
                for sig in sigma:
                  for hidden_dim in hidden_dims:

                    model = Model('RBF', hidden_dim, sig)
                    model.train(train_X, train_y, valid_X, valid_y, ep, lr, bs)

                    model = Model('RBF', hidden_dim, sig)
                    model.train(train_valid_X, train_valid_y, None, None, ep, lr, bs)
                    score = model.score(test_X, test_y)
                    print("score = {} in test set.\n".format(score))
                    ans.append((lr,ep,bs,sig,hidden_dim,score))

    for a in ans:
        print(a)                    
    # ### END YOUR CODE

    # # ------------Feed-Forward Network Case------------
    # ### YOUR CODE HERE
    # # Run your feed-forward network model here
    hidden_dims = [32] 
    learning_rate = [0.01]
    max_epoch = [200]
    batch_size = [128]

    ans = []
    
    for lr in learning_rate:
        for ep in max_epoch:
            for bs in batch_size:
                for hidden_dim in hidden_dims:

                  model = Model('FFN', hidden_dim)
                  model.train(train_X, train_y, valid_X, valid_y, ep, lr, bs)

                  model = Model('FFN', hidden_dim)
                  model.train(train_valid_X, train_valid_y, None, None, ep, lr, bs)
                  score = model.score(test_X, test_y)
                  print("score = {} in test set\n".format(score))

                  ans.append((lr,ep,bs,hidden_dim,score))

    for a in ans:
        print(a)                    

    ### END YOUR CODE
    
if __name__ == '__main__':
    
    main()