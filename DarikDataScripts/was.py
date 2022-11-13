
from ExperimentManagement.ExperimentHierarchy import ExperimentData
import numpy as np
EH = ExperimentData.loadHierarchy("D:\\EM0122")
from BehavioralAnalysis.BurrowFearConditioning import FearConditioning
EH.PreExposure = FearConditioning(EH.passMeta(), "PreExposure")
from ImagingAnalysis.PreprocessingImages import PreProcessing
RawVideoDirectory = EH.PreExposure.folder_dictionary.get("raw_imaging_data")
OutputDirectory = EH.PreExposure.folder_dictionary.get("compiled_imaging_data_folder").path
EH.PreExposure.loadBrukerMetaData()
PreProcessing.repackageBrukerTiffs(RawVideoDirectory, OutputDirectory)
EH.PreExposure.update_folder_dictionary()
images = PreProcessing.loadAllTiffs(OutputDirectory)
images = PreProcessing.blockwiseFastFilterTiff(images, Footprint=np.ones((7, 3, 3)))
images = PreProcessing.removeShuttleArtifact(images, chunk_size=7000, artifact_length=1000)
PreProcessing.saveRawBinary(images, OutputDirectory)
EH.PreExposure.update_folder_dictionary()
EH.recordMod("Repackaged, filtered, and exported images as raw binary. Made video even length this time")
EH.PreExposure.addImageSamplingFolder(30)
from ImagingAnalysis.Suite2PAnalysis import Suite2PModule
MotionCorrection = Suite2PModule(EH.PreExposure.folder_dictionary.get("compiled_imaging_data_folder").path, EH.PreExposure.folder_dictionary.get("imaging_30Hz").path, file_type="binary")
MotionCorrection.motionCorrect()
MotionCorrection.exportCroppedCorrection(MotionCorrection.ops)
del MotionCorrection # Clean Up
from ImagingAnalysis.Denoising import DenoisingModule
Denoiser = DenoisingModule("ModelForPyTorch", "binary_video",
                    model_path="C:\\ProgramData\\Anaconda3\\envs\\Calcium-Imaging-Analysis-Pipeline\\pth",
                    data_path="".join([EH.PreExposure.folder_dictionary.get("imaging_30Hz").path, "\\suite2p\\plane0"]),
                    output_path="".join([EH.PreExposure.folder_dictionary.get("imaging_30Hz").path, "\\denoised"]),
                    image_type="binary",
                    length="14000",
                    workers=4,
                    vram=24,
                    batch_size2=4)
Denoiser.runDenoising()
Denoiser = None
from ImagingAnalysis.Suite2PAnalysis import Suite2PModule
S2P = Suite2PModule("".join([EH.PreExposure.folder_dictionary.get("imaging_30Hz").path, "\\denoised"]), EH.PreExposure.folder_dictionary.get("imaging_30Hz").path, file_type="binary")
S2P.roiDetection()
S2P.extractTraces()
S2P.classifyROIs()
S2P.spikeExtraction() # Finalize (Required spks.npy to use GUI)
S2P.integrateMotionCorrectionDenoising()
S2P.iscell, S2P.stat = S2P.remove_small_neurons(S2P.iscell, S2P.stat)
S2P.save_files()
EH.PreExposure.recordMod()
EH.recordMod("S2P PreExposure")
EH.saveHierarchy()
del S2P
from ImagingAnalysis.FissaAnalysis import FissaModule
Fissa = FissaModule(data_folder=EH.PreExposure.folder_dictionary.get("imaging_30Hz").path, video_folder="".join([EH.PreExposure.folder_dictionary.get("imaging_30Hz").path, "\\denoised"]))
Fissa.initializeFissa()
Fissa.extractTraces() # simple, call to extract raw traces from videos
Fissa.saveFissaPrep()
from ImagingAnalysis.StaticProcessing import Processing
# let's smooth the data with edge-preserving to make it nicer
Fissa.ProcessedTraces.smoothed_raw = Processing.smoothTraces_TiffOrg(Fissa.preparation.raw, niter=50, kappa=150, gamma=0.15)[0]
Fissa.preparation.raw = Fissa.ProcessedTraces.smoothed_raw.copy()
#Let's use for separation, so replace the raws with smooths
Fissa.passPrepToFissa()
Fissa.separateTraces() # simple, call to separate the traces
Fissa.saveFissaSep()
# Calculate Fo/F
Fissa.ProcessedTraces.dFoF_result = Processing.calculate_dFoF(Fissa.experiment.result, Fissa.frame_rate, raw=Fissa.preparation.raw, merge_after=False)

# Condense the ROI Traces for each Trial into a Single Matrix
Fissa.ProcessedTraces.merged_dFoF_result = Processing.mergeTraces(Fissa.ProcessedTraces.dFoF_result)

# Detrend the Traces by fitting a 4th-order polynomial and subsequently subtracting
Fissa.ProcessedTraces.detrended_merged_dFoF_result = Processing.detrendTraces(Fissa.ProcessedTraces.merged_dFoF_result, order=4, plot=False)

# Save
Fissa.saveProcessedTraces()
EH.recordMod("PreExposure Source-Separation")
EH.saveHierarchy()

from ImagingAnalysis.CascadeAnalysis import CascadeModule

Cascade = CascadeModule(Fissa.ProcessedTraces.detrended_merged_dFoF_result, Fissa.frame_rate,
                        "".join([EH.PreExposure.folder_dictionary.get("imaging_30Hz").path, "\\cascade"]),
                        model_folder=
                        "C:\\ProgramData\\Anaconda3\\envs\\Calcium-Imaging-Analysis-Pipeline\\Pretrained_models")

# Pull Available Models
list_of_models = Cascade.pullModels(Cascade.model_folder)
# Select Model: If you know what model you want, you should use the string instead.
# This model is Global_EXC_10Hz_smoothing_100ms
# Cascade.model_name = list_of_models[21]
Cascade.model_name = "Global_EXC_30Hz_smoothing100ms"
Cascade.downloadModel(Cascade.model_name, "C:\\ProgramData\\Anaconda3\\envs\\Calcium-Imaging-Analysis-Pipeline\\Pretrained_models")
# Infer Spike Probability
Cascade.predictSpikeProb() # Simple, call to infer spike probability for each frame
# Calculate Firing Rates # Simple, firing rate = spike probability * imaging frequency
from ImagingAnalysis.StaticProcessing import Processing
Cascade.ProcessedInferences.firing_rates = Processing.calculateFiringRate(Cascade.spike_prob, Cascade.frame_rate)
Cascade.saveSpikeProb(Fissa.output_folder)
Cascade.saveProcessedInferences(Fissa.output_folder)
Cascade.inferDiscreteSpikes()
Cascade.saveSpikeInference(Fissa.output_folder)
EH.recordMod("PreExposure Cascade")
EH.saveHierarchy()
EH.PreExposure.recordMod()
EH.PreExposure.update_folder_dictionary()
EH.PreExposure.folder_dictionary.get("imaging_30Hz").current_stage = "Ready for Analysis"
EH.saveHierarchy()
del Fissa
del Cascade