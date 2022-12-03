# from ExperimentManagement.ExperimentHierarchy import ExperimentData
from IO.metadata import Metadata


class BrukerMeta:
    """
    Module for bruker meta data
    """
    def __init__(self, ImagingMetaFile, *args):

        self.imaging_meta_filepath = ImagingMetaFile
        self.voltage_recording_meta_filepath = None
        self.voltage_output_meta_filepath = None

        if args:
            self.voltage_recording_meta_filepath = args[0]
            if args.__len__() > 1:
                self.voltage_output_meta_filepath = args[1]

        # self.instance_date = ExperimentData.get_date()

        self.imaging_metadata = dict()
        self.frame_metadata = dict()
        self.voltage_recording_metadata = dict()
        self.voltage_output_metadata = dict()

        self.creation_date = None
        self.fov_coordinates = tuple([None, None, None]) # X, Y, Z
        self.num_frames = None
        self.pmt_red_drive = None
        self.pmt_green_drive = None
        self.frame_averaging = None
        self.bruker_power = None
        self.original_resolution = None
        self.objective = tuple([None, None, None]) # mag, NA, zoom
        self.microns_per_pixel = tuple([None, None])
        self.zoom = None
        self.frame_rate = None

        self.analog_channel_names = None
        self.acquisition_rate = None

    def import_meta_data(self):
        _metadata = BrukerMeta.load_meta_data(self.imaging_meta_filepath, self.voltage_recording_meta_filepath,
                                                    self.voltage_output_meta_filepath)


        self.imaging_metadata.update(_metadata.imaging_metadata[0])
        self.imaging_metadata.update(_metadata.imaging_metadata[1])
        self.frame_metadata = _metadata.imaging_metadata[2]
        self.voltage_recording_metadata = _metadata.full_voltage_recording_metadata
        self.voltage_output_metadata = "Pending Implementation"

        self.creation_date = _metadata.full_metadata.get("date")
        self.fov_coordinates = tuple([_metadata.params.get("PositionX"), _metadata.params.get("PositionY"),
                                      _metadata.params.get("PositionZphysical")])
        self.num_frames = _metadata.video_params.get("FrameNumber")

        self.pmt_red_drive = _metadata.imaging_metadata[0].get("PMTGainRed")
        self.pmt_green_drive = _metadata.imaging_metadata[0].get("PMTGainGreen")

        self.frame_averaging = _metadata.translated_imaging_metadata.get("FrameAveraging")
        self.bruker_power = _metadata.translated_imaging_metadata.get("PowerSetting")

        self.original_resolution = _metadata.translated_imaging_metadata.get("Resolution")

        self.objective = tuple([_metadata.translated_imaging_metadata.get("Objective"),
                                _metadata.imaging_metadata[0].get("ObjectiveMag"),
                                _metadata.imaging_metadata[0].get("ObjectiveNA")])

        self.microns_per_pixel = tuple([_metadata.imaging_metadata[0].get("MicronsPerPixelX"),
                                        _metadata.imaging_metadata[0].get("MicronsPerPixelY")])

        self.zoom = _metadata.imaging_metadata[0].get("OpticalZoom")

        self.frame_rate = 1/_metadata.imaging_metadata[0].get("framePeriod")

        self.analog_channel_names = _metadata.recorded_signals_csv

        self.acquisition_rate = \
            _metadata.full_voltage_recording_metadata.get("Experiment").get("Childs").get("Rate").get("Description")

    @staticmethod
    def load_meta_data(ImagingMetaFile, *args):
        _voltage_recording_meta_file = None
        _voltage_output_meta_file = None

        if args:
            _voltage_recording_meta_file = args[0]
            if args.__len__() > 1:
                _voltage_output_meta_file = args[1]

        return Metadata(aq_metadataPath=ImagingMetaFile, voltagerec_metadataPath=_voltage_recording_meta_file,
                            voltageoutput_metadataPath=_voltage_output_meta_file)


if __name__ == "__main__":
    BKM = BrukerMeta("D:\\EM0121\\PreExposure\\Imaging\\BrukerMetaData\\EM0121_PRE-001.xml")
    BKM.import_meta_data()
    print(BKM.zoom)
