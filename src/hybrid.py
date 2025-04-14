from Library.Build_Model import *

# We declare this function here and not in the
# function-storing python file to modify it easily
# as it can change the printouts of the methods
def printout(filename, Stats, model, time):
    # printing Stats
    print('Stats for %s CPU-time %.4f' % (filename, time))
    print('R2 = %.4f (+/- %.4f) Constraint = %.4f (+/- %.4f)' % \
          (Stats.train_objective[0], Stats.train_objective[1],
           Stats.train_loss[0], Stats.train_loss[1]))
    print('Q2 = %.4f (+/- %.4f) Constraint = %.4f (+/- %.4f)' % \
          (Stats.test_objective[0], Stats.test_objective[1],
           Stats.test_loss[0], Stats.test_loss[1]))


# Create, train and evaluate AMN_QP o models with FBA simulated training set for E. coli core
# with EB or UB with a mechanistic layer
DIRECTORY = "./"
# What you can change
seed = 10
ratio_test = 0.1 # part of the training set removed for test
np.random.seed(seed=seed)
# trainname = 'e_coli_core_UB_50' # can change EB by UB
trainname = 'iLC858_core_with_varmed_UB_1000'

timestep = 4
# End of What you can change

# Create model 90% for training 10% for testing
trainingfile = DIRECTORY+'Dataset_model/'+trainname
model = Neural_Model(trainingfile = trainingfile,
                     # objective=['BIOMASS_Ecoli_core_w_GAM'],
                     objective=['bio1'],
                     model_type='AMN_QP',
                     timestep = timestep, learn_rate=1.0,
                     scaler=True,
                     n_hidden = 2, hidden_dim = 25,
                     epochs=500, xfold=5,
                     verbose=True)
ID = np.random.choice(model.X.shape[0],
                      size=int(model.X.shape[0]*ratio_test), replace=False)
Xtest,  Ytest  = model.X[ID,:], model.Y[ID,:]
Xtrain, Ytrain = np.delete(model.X, ID, axis=0), np.delete(model.Y, ID, axis=0)
model.printout()

# Train and evaluate
reservoirname = trainname+'_'+model.model_type
reservoirfile = DIRECTORY+'Reservoir/'+reservoirname
start_time = time.time()
model.X, model.Y = Xtrain, Ytrain
reservoir, pred, stats, _ = train_evaluate_model(model, verbose=False)
delta_time = time.time() - start_time

# Printing cross-validation results
printout(reservoirname, stats, model, delta_time)

# Save, reload and run idependent test set
reservoir.save(reservoirfile)
reservoir.load(reservoirfile)
reservoir.printout()
if len(Xtest) > 0:
    start_time = time.time()
    reservoir.X, reservoir.Y = Xtest, Ytest
    X, Y = model_input(reservoir, verbose=False)
    pred, stats = evaluate_model(reservoir.model, X, Y, reservoir, verbose=False)
    delta_time = time.time() - start_time
    printout('Test set', stats, model, delta_time)