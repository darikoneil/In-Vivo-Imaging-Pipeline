import os
from datetime import date, datetime
import pickle as pkl
import numpy as np



class ExperimentData:
    def __init__(self, **kwargs):
        self.directory = kwargs.get('Directory', None)
        self.mouse_id = kwargs.get('Mouse', None)
        self.study = kwargs.get('Study', None)
        self.study_mouse = kwargs.get('StudyMouse', None)

        self.instance_date = self.getDate()
        self.modifications = [(self.getDate(), self.getTime())]

    @classmethod
    def getDate(cls):
        return date.isoformat(date.today())

    @classmethod
    def getTime(cls):
        return datetime.now().strftime("%H:%M:%S")

    @classmethod
    def checkPath(cls, Path):
        return os.path.exists(Path)

    def passMeta(self):
        return self.directory, self.mouse_id, self.study, self.study_mouse

    def recordMod(self):
        self.modifications.append((self.getDate(), self.getTime()))

    def saveHierarchy(self):
        print("Saving Experimental Hierarchy..")
        _output_file = self.directory + "//" + "ExperimentalHierarchy"
        _output_pickle = open(_output_file, 'wb')

        # iterate through meta
        _metas = [self.directory, self.mouse_id, self.study, self.study_mouse, self.instance_date, self.modifications]
        for _meta in _metas:
            pkl.dump(_meta, _output_pickle)
        _output_pickle.close()
        print("Finished.")

    def loadHierarchy(self):
        return


class BehavioralStage:
    def __init__(self, Meta):
        self.mouse_directory = Meta[0]
        self.mouse_id = Meta[1]
        self.study = Meta[2]
        self.study_mouse = Meta[3]
        self.instance_date = ExperimentData.getDate()
        self.modifications = [(ExperimentData.getDate(), ExperimentData.getTime())]
        self.stage_directory = None
        self.computation_output_folder = None
        self.data_input_folder = None
        self.index_file = None
        self.features_file = None

    def recordMod(self):
        self.modifications.append((ExperimentData.getDate(), ExperimentData.getTime()))

    def setFolders(self):
        self.computation_output_folder = self.stage_directory + "\\Computation"
        self.data_input_folder = self.stage_directory + "\\Imaging"



