#!/usr/bin/env python
"""
Refactored code for metabolic model duplication, training set generation, and model evaluation using PyTorch.

This script performs:
1. **Model Duplication**: Converts a metabolic model to a duplicated version for further processing.
2. **Training Set Generation**: Generates training datasets for metabolic simulations.
3. **Model Training & Evaluation**: Trains mechanistic, ANN, and AMN models and evaluates their performance.

Authors: [Your Name]
Date: [Insert Date]
"""

import time
import numpy as np
import pandas as pd

import cobra
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Style

# Import project-specific functions
from Library.Duplicate_Model import *
from Library.Build_Dataset import *
from Library.Build_Model import *

# =============================================================================
# Utility Functions
# =============================================================================

def pretty_print_section(title):
    """Print a formatted section header in the console."""
    print("\n" + Fore.CYAN + "=" * 80)
    print(f"{title.center(80)}")
    print("=" * 80 + Style.RESET_ALL)

def pretty_print_subsection(title):
    """Print a formatted subsection header in the console."""
    print("\n" + Fore.YELLOW + "-" * 60)
    print(f"{title.center(60)}")
    print("-" * 60 + Style.RESET_ALL)

def load_model(sbml_path: str) -> cobra.Model:
    """
    Load a metabolic model from an SBML file.

    Args:
    - sbml_path (str): Path to the SBML file.

    Returns:
    - cobra.Model: Loaded metabolic model.
    """
    pretty_print_subsection("Loading Metabolic Model")
    print(f"📂 Loading model from: {sbml_path}")
    model = cobra.io.read_sbml_model(sbml_path)
    print(Fore.GREEN + "✅ Model loaded successfully!" + Style.RESET_ALL)
    return model

def print_reactions(model: cobra.Model) -> None:
    """
    Print reaction IDs and names from a metabolic model.

    Args:
    - model (cobra.Model): A loaded COBRA model.
    """
    pretty_print_subsection("Printing Model Reactions")
    for reaction in model.reactions[:10]:  # Show only first 10 reactions for readability
        print(f"🔹 {reaction.id}: {reaction.name}")
    print(Fore.GREEN + f"✅ Displayed first {min(10, len(model.reactions))} reactions." + Style.RESET_ALL)

def duplicate_model_workflow(model: cobra.Model, io_dict: dict) -> cobra.Model:
    """
    Duplicate the metabolic model's reactions and update the medium.

    Args:
    - model (cobra.Model): The original metabolic model.
    - io_dict (dict): Dictionary mapping input/output reaction categories.
    - unsignificant_mols (list): List of molecules to exclude from duplication.

    Returns:
    - cobra.Model: Duplicated metabolic model.
    """
    # Print section header for clarity in command-line output
    pretty_print_section("Duplicating Metabolic Model")

    # Start timing for performance monitoring
    start_time = time.time()

    # Override io_dict with fixed standard values (remove these overrides if you want to use the passed parameters)
    io_dict = {
        "_i": [(None, "e"), (None, "c"), ("e", "p"), ("p", "c"), ("e", "c"), ("c", "m"), ("p", "m")],
        "_o": [("c", None), ("e", None), ("p", "e"), ("c", "p"), ("c", "e"), ("m", "c"), ("m", "p")]
    }

    # Debug: confirm that io_dict is indeed a dictionary
    print("io_dict type:", type(io_dict))  # Expected: <class 'dict'>

    # Override unsignificant_mols with fixed values (remove these if you want to use passed parameter)
    unsignificant_mols = ["h_p", "h_c", "pi_c", "pi_p", "adp_c", "h2o_c", "atp_c"]

    # Screen the original model to get mapping information for duplication
    reac_id_to_io_count_and_way = screen_out_in(model, io_dict, unsignificant_mols)
    
    # Duplicate the model using the screening dictionary
    new_model = duplicate_model(model, reac_id_to_io_count_and_way)

    # Debug: Print out the first few reactions with non-zero lower bounds
    print(Fore.YELLOW + "🔄 Checking reactions with non-zero lower bounds..." + Style.RESET_ALL)
    for reac in new_model.reactions[:5]:  # Show only the first 5 for brevity
        if reac.lower_bound != 0:
            print(f"⚠️ Reaction: {reac.id} - Bounds: {reac.bounds}")

    # Correct the medium of the duplicated model based on the original model's medium
    new_model.medium = correct_duplicated_med(model.medium, new_model.medium)
    print(Fore.GREEN + "✅ Medium updated successfully!" + Style.RESET_ALL)

    # Print the elapsed time for duplication
    elapsed_time = time.time() - start_time
    print(Fore.CYAN + f"Model duplication completed in {elapsed_time:.2f} seconds." + Style.RESET_ALL)
    
    return new_model


def save_model(new_model: cobra.Model, original_path: str) -> str:
    """
    Save the duplicated metabolic model to a new SBML file.

    Args:
    - new_model (cobra.Model): The duplicated metabolic model.
    - original_path (str): Path of the original model file.

    Returns:
    - str: Path of the saved duplicated model.
    """
    pretty_print_subsection("Saving Duplicated Model")
    new_name = original_path[:-5] + "_duplicated.xml"
    cobra.io.write_sbml_model(new_model, new_name)
    print(Fore.GREEN + f"✅ Duplicated model saved at: {new_name}" + Style.RESET_ALL)
    return new_name

# =============================================================================
# Execution Flow (Main)
# =============================================================================

def main():
    """
    Main function to execute the entire pipeline for:
    1. Model Duplication
    2. Medium Processing
    3. Training Set Generation
    4. Model Training & Evaluation
    """

    # Step 1: Load and duplicate metabolic model
    pretty_print_section("STEP 1: MODEL DUPLICATION")
    sbml_path = "./Reservoir/iLC858.sbml"
    model = load_model(sbml_path)
    print(model)
    print_reactions(model)

    io_dict = {
        "_i": [(None, "e"), (None, "c"), ("e", "p"), ("p", "c"), ("e", "c"), ("c", "m"), ("p", "m")],
        "_o": [("c", None), ("e", None), ("p", "e"), ("c", "p"), ("c", "e"), ("m", "c"), ("m", "p")]
    }
    print("io_dict type:", type(io_dict))
    unsignificant_mols = ["h_p", "h_c", "pi_c", "pi_p", "adp_c", "h2o_c", "atp_c"]
    # Will print a dictionary counting the reactions in reversible, forward, backward
    reac_id_to_io_count_and_way = screen_out_in(model, io_dict, unsignificant_mols)
    new_model = duplicate_model_workflow(model, reac_id_to_io_count_and_way)
    save_model(new_model, sbml_path)

 
    for reac in new_model.reactions:
        if reac.lower_bound != 0:
            print('reaction with non-zero lower bound:', reac.id, reac.bounds)
    for el in new_model.medium:
        if new_model.reactions.get_by_id(el).lower_bound != 0:
            print('medium reaction with non-zero lower bound:',el)

    default_med = model.medium
    new_med = new_model.medium
    correct_med =  correct_duplicated_med(default_med, new_med)
    new_model.medium = correct_med
    print("New Medium Created")
    #print(new_model.medium)
    
    for i in range(10):
        s, new_s = change_medium(model, new_model, i*3)
        if s != None and new_s != None:
            print(s, new_s, "diff = ", abs(s-new_s))
        elif s != None:
            print("infeasible duplicated medium")
        elif new_s != None:
            print("infeasible default medium")
        elif s == None and new_s == None:
            print("Both medium are impossible")
    
    new_model.repair()

    model_path = "./Dataset_input/iLC858.sbml"

    print("Original model's location: " + model_path)
    new_name = model_path[:-5] + "_duplicated" + '.xml'
    cobra.io.write_sbml_model(new_model, new_name)
    print("Duplicated model's location: " + new_name)

   # Step 2: Generate Vibrio medium and save
    pretty_print_section("STEP 2: GENERATING VIBRIO MEDIUM")
    medium_csv_path = "./Dataset_input/e_coli_core.csv"
    medium_df = pd.read_csv(medium_csv_path)
    map = {
    'EX_co2_e_i': 'EX_cpd00011_e_i',
    'EX_glc__D_e_i': 'EX_cpd00027_e_i',
    'EX_h_e_i': 'EX_cpd00067_e_i',
    'EX_h2o_e_i': 'EX_cpd00001_e_i',
    'EX_nh4_e_i': 'EX_cpd00013_e_i',
    'EX_o2_e_i': 'EX_cpd00007_e_i',
    'EX_pi_e_i': 'EX_cpd00009_e_i',
    # 'EX_ac_e_i': 'EX_cpd00029_e_i',
    # 'EX_acald_e_i': '',   No equivalent exchange reaction
    # 'EX_akg_e_i': '',
    # 'EX_etoh_e_i': '',
    # 'EX_for_e_i': '',
    # 'EX_fru_e_i': '',
    'EX_fum_e_i': 'EX_cpd00106_e_i',
    # 'EX_gln__L_e_i': '',
    # 'EX_glu__L_e_i': '',
    # 'EX_lac__D_e_i': '',
    'EX_mal__L_e_i': 'EX_cpd00179_e_i',
    # ' EX_pyr_e_i': '',
    'EX_succ_e_i': 'EX_cpd00036_e_i',}

    vibrio_medium_df = medium_df.loc[:,medium_df.columns.isin(list(map.keys()))].rename(columns=map)
    vibrio_essential_inflow_rxns = [key for key in new_model.medium.keys() if new_model.medium[key] == 1000]
    missing_rxns = list(set(vibrio_essential_inflow_rxns).difference(set(vibrio_medium_df.columns)))
    vibrio_medium_df = pd.concat([vibrio_medium_df, pd.DataFrame([np.ones(len(missing_rxns)),
                   10 * np.ones(len(missing_rxns)),
                   [np.nan] * len(missing_rxns)],
                  columns=missing_rxns)], axis=1)
    # Add more essential reactions
    vibrio_essential_inflow_rxns = [key for key in new_model.medium.keys() if new_model.medium[key] == 1000]
    missing_rxns = list(set(vibrio_essential_inflow_rxns).difference(set(vibrio_medium_df.columns)))
    vibrio_medium_df = pd.concat([vibrio_medium_df, pd.DataFrame([np.ones(len(missing_rxns)),
                    10 * np.ones(len(missing_rxns)),
                    [np.nan] * len(missing_rxns)],
                    columns=missing_rxns)], axis=1)
    vibrio_medium_df.to_csv('/Users/mukulsherekar/pythonProject/hGSMM/Dataset_input/iLC858_core_with_varmed.csv', index=False)
    print(Fore.GREEN + "✅ Vibrio medium generated and saved!" + Style.RESET_ALL)
    fluxes = new_model.optimize().fluxes
    nonzero_fluxes = fluxes[(fluxes > 0) & (fluxes.index.str.startswith('EX'))]
    print(nonzero_fluxes)
    print(new_model.reactions.get_by_id('EX_cpd00048_e_o'))
    print(new_model.reactions.get_by_id('EX_cpd10516_e_o'))

    # Step 3: Training Set Generation
    pretty_print_section("STEP 3: TRAINING SET GENERATION")
    
    seed = 10
    np.random.seed(seed)
    DIRECTORY = './'
    cobraname = 'iLC858_duplicated'
    mediumname = 'iLC858_core_with_varmed'
    mediumbound = 'UB'
    method = 'FBA' # FBA or pFBA or EXP
    size = 1000
    reduce = False

    # Generate training set
    print(Fore.YELLOW + "🔄 Generating training set..." + Style.RESET_ALL)
    # Run cobra
    cobrafile  = DIRECTORY+'Dataset_input/'+cobraname
    mediumfile = DIRECTORY+'Dataset_input/'+mediumname
    parameter = TrainingSet(cobraname=cobrafile,
                            mediumname=mediumfile, mediumbound=mediumbound,
                            method=method,objective=[],
                            measure=[])
    #parameter.get(sample_size=size)
    parameter.get(sample_size=size, verbose=True)

    # Saving file
    print(Fore.YELLOW + "🔄 Saving training set..." + Style.RESET_ALL)
    trainingfile  = DIRECTORY+'Dataset_model/'+mediumname+'_'+parameter.mediumbound+'_'+str(size)
    parameter.save(trainingfile, reduce=reduce)

    # Verifying
    print(Fore.YELLOW + "🔄 Verifying training set..." + Style.RESET_ALL)
    parameter = TrainingSet()
    parameter.load(trainingfile)
    #parameter.printout()

    pretty_print_section("TRAINING SET HEATMAPS")
    sns.heatmap(pd.DataFrame(parameter.X))
    plt.show()
    sns.heatmap(pd.DataFrame(parameter.Y))
    plt.show()
    #plot_heatmaps(training_set)

    print(Fore.CYAN + "=" * 80)
    print(f"🎉🎉🎉  PIPELINE COMPLETED SUCCESSFULLY! 🎉🎉🎉".center(80))
    print("=" * 80 + Style.RESET_ALL)

    



if __name__ == '__main__':
    main()
