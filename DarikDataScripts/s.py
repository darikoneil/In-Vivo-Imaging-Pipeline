fig1 = plt.figure(figsize=(9, 9))
ax1 = fig1.add_subplot(111)
ax1.set_title("Receiver Operating Characteristic")
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")

plotROC(EM0121.PreExposure.decoder_dictionary.get(("CS+ Trial", "Shuffled Frames")).ModelPerformance[('ROC', 'training')][0],
    EM0121.PreExposure.decoder_dictionary.get(("CS+ Trial", "Shuffled Frames")).ModelPerformance[('ROC', 'training')][1], ax=ax1,
        color="#ff4e4b", ls="-", alpha=0.75)

plotROC(EM0121.PreExposure.decoder_dictionary.get(("CS+ Trial", "Shuffled Frames")).ModelPerformance[('ROC', 'testing')][0],
    EM0121.PreExposure.decoder_dictionary.get(("CS+ Trial", "Shuffled Frames")).ModelPerformance[('ROC', 'testing')][1], ax=ax1,
        color="#ff4e4b", ls=":", alpha=0.75)

plotROC(EM0121.PreExposure.decoder_dictionary.get(("CS+ Trial", "Shuffled Labels")).ModelPerformance[('ROC', 'training')][0],
    EM0121.PreExposure.decoder_dictionary.get(("CS+ Trial", "Shuffled Labels")).ModelPerformance[('ROC', 'training')][1], ax=ax1,
        color="#139fff", ls="-", alpha=0.75)

plotROC(EM0121.PreExposure.decoder_dictionary.get(("CS+ Trial", "Shuffled Labels")).ModelPerformance[('ROC', 'testing')][0],
    EM0121.PreExposure.decoder_dictionary.get(("CS+ Trial", "Shuffled Labels")).ModelPerformance[('ROC', 'testing')][1], ax=ax1,
        color="#139fff", ls=":", alpha=0.75)

plotROC(EM0121.PreExposure.decoder_dictionary.get(("CS+ Trial", "Shuffled Neurons")).ModelPerformance[('ROC', 'training')][0],
    EM0121.PreExposure.decoder_dictionary.get(("CS+ Trial", "Shuffled Neurons")).ModelPerformance[('ROC', 'training')][1], ax=ax1,
        color="#7840cc", ls="-", alpha=0.75)

plotROC(EM0121.PreExposure.decoder_dictionary.get(("CS+ Trial", "Shuffled Neurons")).ModelPerformance[('ROC', 'testing')][0],
    EM0121.PreExposure.decoder_dictionary.get(("CS+ Trial", "Shuffled Neurons")).ModelPerformance[('ROC', 'testing')][1], ax=ax1,
        color="#7840cc", ls=":", alpha=0.75)

ax1.plot([0, 1], [0, 1], color="black", lw=3, ls='--', alpha=0.95)
_legend = ["".join(["Shuffled Frames-Training: ",
                    str(np.around(EM0121.PreExposure.decoder_dictionary.get(("CS+ Trial",
                        "Shuffled Frames")).ModelPerformance[("AUC", "training")], decimals=2))]),
           "".join(["Shuffled Frames-Testing: ",
                    str(np.around(EM0121.PreExposure.decoder_dictionary.get(("CS+ Trial",
                        "Shuffled Frames")).ModelPerformance[("AUC", "testing")], decimals=2))]),
            "".join(["Shuffled Labels-Training: ",
                    str(np.around(EM0121.PreExposure.decoder_dictionary.get(("CS+ Trial",
                        "Shuffled Labels")).ModelPerformance[("AUC", "training")], decimals=2))]),
           "".join(["Shuffled Labels-Testing: ",
                    str(np.around(EM0121.PreExposure.decoder_dictionary.get(("CS+ Trial",
                        "Shuffled Labels")).ModelPerformance[("AUC", "testing")], decimals=2))]),
           "".join(["Shuffled Neurons-Training: ",
                    str(np.around(EM0121.PreExposure.decoder_dictionary.get(("CS+ Trial",
                                                                             "Shuffled Neurons")).ModelPerformance[
                                      ("AUC", "training")], decimals=2))]),
           "".join(["Shuffled Neurons-Testing: ",
                    str(np.around(EM0121.PreExposure.decoder_dictionary.get(("CS+ Trial",
                                                                             "Shuffled Neurons")).ModelPerformance[
                                      ("AUC", "testing")], decimals=2))]),
           "No-Skill: 0.50"
]

ax1.legend(_legend)


AUCs = "".join([
    "\n", "AUC",
    "\n", "Shuffled Frames-Training: ",
    str(EM0121.PreExposure.decoder_dictionary.get(("CS+ Trial",
                                                   "Shuffled Frames")).ModelPerformance[("AUC", "training")]),
    "\n", "Shuffled Frames-Testing: ",
    str(EM0121.PreExposure.decoder_dictionary.get(("CS+ Trial",
                                                   "Shuffled Frames")).ModelPerformance[("AUC", "testing")]),
    "\n", "Shuffled Labels-Training: ",
    str(EM0121.PreExposure.decoder_dictionary.get(("CS+ Trial",
                                                   "Shuffled Labels")).ModelPerformance[("AUC", "training")]),
    "\n", "Shuffled Labels-Testing: ",
    str(EM0121.PreExposure.decoder_dictionary.get(("CS+ Trial",
                                                   "Shuffled Labels")).ModelPerformance[("AUC", "testing")]),
    "\n", "Shuffled Neurons-Training: ",
    str(EM0121.PreExposure.decoder_dictionary.get(("CS+ Trial",
                                                   "Shuffled Neurons")).ModelPerformance[("AUC", "training")]),
    "\n", "Shuffled Neurons-Testing: ",
    str(EM0121.PreExposure.decoder_dictionary.get(("CS+ Trial",
                                                   "Shuffled Neurons")).ModelPerformance[("AUC", "testing")]),
])
# props = dict(boxstyle="round", facecolor="white", alpha=0.5)
# ax1.text(0.55, 0.25, AUCs, transform=ax1.transAxes, fontsize=10, verticalalignment="top", bbox=props)
fig1.tight_layout()


###

# Segregate trials
neural_plus = NeuralData_Matrix[:, np.where(FeatureData_Matrix[0, :]==0)[0]]
neural_minus = NeuralData_Matrix[:, np.where(FeatureData_Matrix[1, :]==1)[0]]

# reformat
neural_plus_trials = np.full((5, 663, 345), 0, dtype=np.float64)
neural_minus_trials = np.zeros_like(neural_plus_trials)

_start_idx = np.arange(0, 1725, 345)
_end_idx = np.append(np.arange(345, 1725, 345), 1725)

for i in range(5):
    neural_plus_trials[i, :, :] = neural_plus[:, _start_idx[i]:_end_idx[i]]
    neural_minus_trials[i, :, :] = neural_minus[:, _start_idx[i]:_end_idx[i]]

neural_trial_form = np.concatenate([neural_plus_trials, neural_minus_trials], axis=0)
features_trial_form = np.full((10, 1, 345), 0, dtype=np.float64)
features_trial_form[0:5, :, :] = 1

NTF = np.full((10, 663, 345), 0, dtype=np.float64)
_start_idx = np.arange(0, NeuralData_Matrix.shape[1], 345)
_end_idx = np.append(np.arange(345, NeuralData_Matrix.shape[1], 345), NeuralData_Matrix.shape[1])
for i in range(9):
    NTF[i, :, :] = NeuralData_Matrix[:, _start_idx[i]:_end_idx[i]]



