from Library.Build_Model import *
from vibrio_modeling import pretty_print_section

# Step 4: Model Training & Evaluation
pretty_print_section("MECHANISTIC MODEL TRAINING & EVALUATION")
# Run Mechanistic model (no training) QP (quadratic program) or LP (linear program)
# using E. coli core simulation training sets and EB (or UB) bounds

# What you can change
seed = 10
np.random.seed(seed=seed)

trainname = 'iLC858_core_with_varmed_UB_1000'
# trainname = 'e_coli_core_UB_50' # the training set file name (generated at previous step)


size = 5 # number of runs must be lower than the number of element in trainname
# timestep = int(1.0e4) # LP 1.0e4 QP 1.0e5
timestep = int(1e4)
learn_rate = 1.0 # for LP: 0.3; for QP: 1.0
decay_rate = 0.9 # only in QP, 0.9
solver = 'MM_QP' # choose between MM_LP or MM_QP
# End of What you can change

DIRECTORY = "./"    

# Create model and run GD for X and Y randomly drawn from trainingfile
trainingfile = DIRECTORY+'Dataset_model/'+trainname
model = Neural_Model(trainingfile = trainingfile,
            objective=['bio1'], #
            # objective=['BIOMASS_Ecoli_core_w_GAM'],
            model_type = solver,
            timestep = timestep,
            learn_rate = learn_rate,
            decay_rate = decay_rate)
# Prints a summary of the model before running

model.printout()

def printout(V, Stats, model):
    # printing Stats
    print("R2 = %.2f (+/- %.2f) Constraint = %.2f (+/- %.2f)" % \
        (Stats.train_objective[0], Stats.train_objective[1],
        Stats.train_loss[0], Stats.train_loss[1]))
    Vout = tf.convert_to_tensor(np.float32(model.Y))
    Loss_norm, dLoss = Loss_Vout(V, model.Pout, Vout)
    print('Loss Targets', np.mean(Loss_norm))
    Loss_norm, dLoss = Loss_SV(V, model.S)
    print('Loss SV', np.mean(Loss_norm))
    Vin = tf.convert_to_tensor(np.float32(model.X))
    Pin = tf.convert_to_tensor(np.float32(model.Pin))
    if Vin.shape[1] == model.S.shape[1]: # special case
        Vin  = tf.linalg.matmul(Vin, tf.transpose(Pin), b_is_sparse=True)
    # Loss_norm, dLoss = Loss_Vin(V, model.Pin, Vin, model.mediumbound)
    # print('Loss Vin bound', np.mean(Loss_norm))
    Loss_norm, dLoss = Loss_Vpos(V, model)
    print('Loss V positive', np.mean(Loss_norm))

    # Runs the appropriate method

if model.model_type is 'MM_QP':
    Ypred, Stats = MM_QP(model, verbose=True)
if model.model_type is 'MM_LP':
    Ypred, Stats = MM_LP(model, verbose=True)

# Printing results
printout(Ypred, Stats, model)


