import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from snsynth import MWEMSynthesizer
from snsynth import QUAILSynthesizer
from snsynth.pytorch import PytorchDPSynthesizer
from snsynth.pytorch.nn import DPCTGAN, PATECTGAN

## DATA SYNTHESIS ##

def get_synthesizer(synthesizer, epsilon, embedding_dim=128, generator_dim=(256,256), discriminator_dim=(256,256),
                    batch_size=500, noise_multiplier=1e-3, sigma=5):

    # Instantiate an MWEM synthesizer
    if synthesizer == "MWEM":
        synth = MWEMSynthesizer(epsilon=epsilon, q_count=500, iterations=30, mult_weights_iterations=15,
        splits=[], split_factor=2, max_bin_count=400)

    elif synthesizer == "DPCTGAN":
        synth = PytorchDPSynthesizer(epsilon=epsilon, gan=DPCTGAN(embedding_dim=embedding_dim, generator_dim=generator_dim,
                                                                    discriminator_dim=discriminator_dim, batch_size=batch_size, sigma=sigma), preprocessor=None)

    elif synthesizer == "PATECTGAN":
        synth = PytorchDPSynthesizer(epsilon=epsilon, gan=PATECTGAN(embedding_dim=embedding_dim, generator_dim=generator_dim,
                                                                    discriminator_dim=discriminator_dim, batch_size=batch_size, noise_multiplier=noise_multiplier), preprocessor=None)

    return synth

def get_quail_synthesizer(synthesizer, classifier, epsilon, eps_split, target):

    # Instantiate a synthesizer
    if synthesizer == "MWEM":
        def QuailSynth(epsilon):
            return MWEMSynthesizer(epsilon=epsilon, q_count=500, iterations=30, mult_weights_iterations=15,
                                    splits=[], split_factor=2, max_bin_count=400)

    elif synthesizer == "DPCTGAN":
        def QuailSynth(epsilon):
            return PytorchDPSynthesizer(epsilon=epsilon, gan=DPCTGAN(), preprocessor=None)

    elif synthesizer == "PATECTGAN":
        def QuailSynth(epsilon):
            return PytorchDPSynthesizer(epsilon=epsilon, gan=PATECTGAN(), preprocessor=None)

    # Instantiate a DPLR classifier
    if classifier == "DPLR":
        def QuailClassifier(epsilon):
            return DPLR(epsilon=epsilon)

    # Create a QUAIL synthesizer with base synthesizer and DP classifier
    quail = QUAILSynthesizer(epsilon, QuailSynth, QuailClassifier, target, eps_split)

    return quail

def save_synthetic_data(epsilon_vals, train_df, synthesizer="MWEM", quail=False, classifier=None, eps_split=None,
                        n_reps=1, dataset="adult", embedding_dim=128, generator_dim=(256,256), discriminator_dim=(256,256),
                        batch_size=500, noise_multiplier=1e-3, sigma=5, results_dir=""):

    # Get a list of all features and target to be passed into categorical_columns
    if dataset == "adult":
        target = "label"
        all_columns = ["age", "workclass", "education", "education_num", "occupation",
                               "relationship", "race", "sex", "hours_week", "label"]

    elif dataset == "acs":
        target = "label"
        all_columns = ["age", "workclass", "education", "occupation", "marital",
                               "relationship", "race", "sex", "hours_week", "label"]

    elif dataset == "compas":
        target = "two_year_recid"
        all_columns = ['age','priors_count','sex','juv_fel_count', 'juv_misd_count',
                       'juv_other_count', 'c_charge_degree', 'length_of_stay','race','two_year_recid']

    elif dataset == "german":
        target = "credit_risk"
        all_columns = ['duration_in_year', 'credit_amount', 'installment_rate_in_percentage_of_disposable_income',
                       'age', 'credit_history', 'housing','status_of_existing_checking_account','present_employment_since', 'purpose', 'credit_risk']

    else:
        raise Exception(f"Dataset {dataset} not recognized.")

    # Loop through epsilon values and repetitions
    for epsilon in epsilon_vals:
        for i in range(n_reps):

            if quail:
                # Create a QUAIL-wrapped synthesizer
                synth = get_quail_synthesizer(synthesizer, classifier, epsilon, eps_split, target=target)

                # Fit synthesizer to the training data
                synth.fit(train_df, categorical_columns=[col for col in train_df.columns.tolist() if col != target])

            else:
                # Create a regular synthesizer
                synth = get_synthesizer(synthesizer, epsilon, embedding_dim=embedding_dim, generator_dim=generator_dim,
                                        discriminator_dim=discriminator_dim, batch_size=batch_size, noise_multiplier=noise_multiplier,
                                        sigma=sigma)

                if synthesizer == "MWEM":

                    # Fit synthesizer to the training data
                    synth.fit(train_df.to_numpy(), categorical_columns=train_df.columns.tolist())

                elif synthesizer in ["DPCTGAN", "PATECTGAN"]:

                  # Fit synthesizer to the training data
                  synth.fit(train_df, categorical_columns=train_df.columns.tolist())

            # Create and save private synthetic data
            train_synth = pd.DataFrame(synth.sample(int(train_df.shape[0])), columns=train_df.columns)
            if quail:
                train_synth.to_csv(results_dir+f"QUAIL_{synthesizer}_eps={epsilon}_rep={i}.csv")
            else:
                train_synth.to_csv(results_dir+f"{synthesizer}_eps={epsilon}_rep={i}.csv")

            print(f"Completed eps={epsilon}, rep={i+1}.")

def plot_distributions(df, title, dataset="adult", savefig=False):

    # Get name of target
    if dataset == "adult":
        target = "label"
    elif dataset == "acs":
        target = "label"
    elif dataset == "compas":
        target = "two_year_recid"
    elif dataset == "german":
        target = "credit_risk"
    else:
        raise Exception(f"Dataset {dataset} not recognized.")

    # Plot distributions of all features
    fig, ax = plt.subplots(2, 5, figsize=(20,8))
    ax = ax.ravel()
    for i, col in enumerate(df.columns):
            g = sns.histplot(data=df, x=col, hue=target, multiple="stack", ax=ax[i], discrete=True)
    fig.suptitle(title, size=20)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if savefig:
        plt.savefig(f"{dataset}_distributions.png", dpi=300, bbox_inches="tight")
