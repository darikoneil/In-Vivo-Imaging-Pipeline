# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:08:27 2021

@author: sp3660
"""
import glob
import os
import time
import numpy as np
import xml.etree.ElementTree as ET
import scipy as sp
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import logging
import json
import copy
from dateutil import parser

module_logger = logging.getLogger(__name__)

try:
    from .recursively_read_metadata import recursively_read_metadata
    from .select_values_gui import select_values_gui
    from .manually_get_some_metadata import manually_get_some_metadata
except:
    from recursively_read_metadata import recursively_read_metadata
    from select_values_gui import select_values_gui
    from manually_get_some_metadata import manually_get_some_metadata


# %%
class Metadata():

    def __init__(self, aq_metadataPath=None,
                 photostim_metadataPath=None,
                 voltagerec_metadataPath=None,
                 face_camera_metadata_path=False,
                 voltageoutput_metadataPath=False,
                 imaging_database_row=None,
                 temporary_path=None,
                 acquisition_directory_raw=None,
                 aquisition_object=None,
                 from_database=False):
        module_logger.info('Processing Metadata')

        self.temporary_path = temporary_path
        self.aquisition_object = aquisition_object
        self.full_mark_points_metadata = None
        self.acquisition_directory_raw = acquisition_directory_raw
        self.full_voltage_recording_metadata = None
        self.imaging_metadata_file = aq_metadataPath
        self.photostim_file = photostim_metadataPath
        self.voltage_file = voltagerec_metadataPath
        self.voltage_output = voltageoutput_metadataPath
        self.full_metadata = None
        self.full_voltage_recording_metadata = None
        self.imaging_metadata = None
        self.translated_imaging_metadata = None
        self.recorded_signals_csv = None
        self.recorded_signals = None
        self.all_frames = []
        self.all_volumes = []
        self.video_params = []
        self.timestamps = {}
        self.params = []

        self.check_metadata_in_folder()

        if self.aquisition_object:
            self.read_metadata_path = os.path.join(self.aquisition_object.slow_storage_all_paths['metadata'],
                                                   'imaging_metadata.json')
            self.timestamps_path = os.path.join(self.aquisition_object.slow_storage_all_paths['metadata'],
                                                'timestamps.json')
            self.read_voltage_metadata_path = os.path.join(self.aquisition_object.slow_storage_all_paths['metadata'],
                                                           'voltage_metadata.json')

            if from_database:
                self.get_all_metadata_from_database()
            else:
                module_logger.info('readig metdata from json')
                self.read_json_metadata()

        if self.imaging_metadata_file:
            if os.path.isfile(self.imaging_metadata_file):
                # module_logger.info('getting metadata')
                self.process_metadata()

        if self.voltage_file:
            if os.path.isfile(self.voltage_file):
                self.process_voltage_recording()

        if self.photostim_file:
            if os.path.isfile(self.photostim_file):
                self.process_mark_points_xml()

        if not from_database:
            self.translate_metadata()
            # else:
            #     self.add_metadata_manually()
            #     if self.photostim_metadataPath!=None:
            #         if os.path.isfile(self.photostim_file):
            #             self.photstim_metadata=self.process_photostim_metadata()
            #             self.photostim_extra_metadata=self.create_photostim_sequence(self.photostim_metadata, self.imaging_metadata)#,
            #             self.photostim_metadata['PhotoStimSeriesArtifact']=self.photostim_extra_metadata

            if self.temporary_path:
                self.transfer_metadata()
            if self.aquisition_object:
                self.save_metadata_as_json()

                # if self.video_params:
        #     self.plotting()
        module_logger.info('Finished  Metadata')

    def get_timestamps(self):

        if os.path.isfile(self.timestamps_path):
            if not self.timestamps:
                with open(self.timestamps_path, 'rb') as fout:
                    self.timestamps = json.load(fout)

        if not self.timestamps:
            if isinstance(self.video_params['relativeTimes'][0], list):
                self.timestamps = {'Plane' + str(i + 1): [vol[i] for vol in self.video_params['relativeTimes']] for i in
                                   range(len(self.video_params['relativeTimes'][0]))}
            elif isinstance(self.video_params['relativeTimes'][0], float):
                self.timestamps = {'Plane1': self.video_params['relativeTimes']}

        if not os.path.isfile(self.timestamps_path):
            if self.timestamps:
                with open(self.timestamps_path, 'w') as fout:
                    json.dump(self.timestamps, fout)

    def process_mark_points_xml(self):

        if not self.full_mark_points_metadata:
            tree = ET.parse(self.photostim_file)
            root = tree.getroot()
            self.full_mark_points_metadata = recursively_read_metadata(root)
            self.process_photostim_metadata()

    def process_voltage_recording(self):

        if not self.full_voltage_recording_metadata:
            tree = ET.parse(self.voltage_file)
            root = tree.getroot()
            self.full_voltage_recording_metadata = recursively_read_metadata(root)

        voltage_aq_time = self.full_voltage_recording_metadata['DateTime']['Description'][
                          self.full_voltage_recording_metadata['DateTime']['Description'].find('T') + 1:]
        # voltage_aq_time=root[3].text[root[3].text.find('T')+1:]
        self.voltage_aq_time = parser.parse(voltage_aq_time).time().strftime('%H:%M:%S')
        ExperimentInfo = self.full_voltage_recording_metadata['Experiment']
        # recorded_channels=[elem['Childs']['VisibleSignals']['VRecSignalPerPlotProperties']['Childs']['SignalId']['Value']['Description'] for elem in ExperimentInfo['Childs']["PlotConfigList"].values() if elem['Childs']['VisibleSignals']['VRecSignalPerPlotProperties']]
        recorded_channels = [
            elem['Childs']['VisibleSignals']['VRecSignalPerPlotProperties']['Childs']['SignalId']['Value'][
                'Description'] if elem['Childs']['VisibleSignals'] else '' for elem in
            ExperimentInfo['Childs']["PlotConfigList"].values()]

        recorded_channels2 = [chan[chan.find(')') - 1] if chan else '' for chan in recorded_channels]
        self.recorded_signals = [elem['Childs']['Name']['Description'] for elem in
                                 ExperimentInfo['Childs']["SignalList"].values() if
                                 elem['Childs']['Channel']['Description'] in recorded_channels2]
        self.recorded_signals_csv = []
        pth = ((os.path.splitext(self.voltage_file)[0]) + '.csv')
        if os.path.isfile(pth):
            voltage_rec_csv = pd.read_csv(pth, header=0)
            recorded_signals_csv = list(voltage_rec_csv.columns)
            recorded_signals_csv.pop(0)
            self.recorded_signals_csv = [i.strip() for i in recorded_signals_csv]

    def process_metadata(self):

        if not self.full_metadata:
            tree = ET.parse(self.imaging_metadata_file)
            root = tree.getroot()
            self.full_metadata = recursively_read_metadata(root)
        if not self.full_metadata:
            return []
        else:
            MicroscopeInfo = self.full_metadata['PVStateShard']
            seqinfo = self.full_metadata['Sequence']

            self.params = {'ImagingTime': parser.parse(self.full_metadata['date']).time().strftime('%H:%M:%S'),
                           'Date': self.full_metadata['date'],
                           'AquisitionName': os.path.splitext(os.path.basename(self.imaging_metadata_file))[0]}

            for element in MicroscopeInfo['Childs'].values():

                if element['key'] == 'activeMode':
                    self.params['ScanMode'] = element['value']

                if element['key'] == 'bitDepth':
                    self.params['BitDepth'] = int(element['value'])

                if element['key'] == 'dwellTime':
                    self.params['dwellTime'] = float(element['value'])

                if element['key'] == 'framePeriod':
                    self.params['framePeriod'] = float(element['value'])

                if element['key'] == 'laserPower':
                    self.params['ImagingLaserPower'] = float(element['IndexedValue']['value'])

                if element['key'] == 'laserPower':
                    self.params['UncagingLaserPower'] = float(element['IndexedValue_1']['value'])

                if element['key'] == 'linesPerFrame':
                    self.params['LinesPerFrame'] = int(element['value'])

                if element['key'] == 'micronsPerPixel':
                    self.params['MicronsPerPixelX'] = float(element['IndexedValue']['value'])

                if element['key'] == 'micronsPerPixel':
                    self.params['MicronsPerPixelY'] = float(element['IndexedValue_1']['value'])

                if element['key'] == 'objectiveLens':
                    self.params['Objective'] = element['value']

                if element['key'] == 'objectiveLensMag':
                    self.params['ObjectiveMag'] = int(element['value'])

                if element['key'] == 'objectiveLensNA':
                    self.params['ObjectiveNA'] = float(element['value'])

                if element['key'] == 'opticalZoom':
                    self.params['OpticalZoom'] = float(element['value'])

                if element['key'] == 'pixelsPerLine':
                    self.params['PixelsPerLine'] = int(element['value'])

                if element['key'] == 'pmtGain':
                    self.params['PMTGainRed'] = float(element['IndexedValue']['value'])

                if element['key'] == 'pmtGain':
                    self.params['PMTGainGreen'] = float(element['IndexedValue_1']['value'])

                if element['key'] == 'positionCurrent':
                    self.params['PositionX'] = float(element['SubindexedValues']['Childs']['SubindexedValue']['value'])

                if element['key'] == 'positionCurrent':
                    self.params['PositionY'] = float(
                        element['SubindexedValues_1']['Childs']['SubindexedValue']['value'])

                if element['key'] == 'positionCurrent':
                    self.params['PositionZphysical'] = float(
                        element['SubindexedValues_2']['Childs']['SubindexedValue']['value'])

                if element['key'] == 'positionCurrent':
                    self.params['PositionZETL'] = float(
                        element['SubindexedValues_2']['Childs']['SubindexedValue_1']['value'])

                if element['key'] == 'rastersPerFrame':
                    self.params['RasterAveraging'] = int(element['value'])

                if element['key'] == 'resonantSamplesPerPixel':
                    self.params['ResonantSampling'] = int(element['value'])

                if element['key'] == 'scanLinePeriod':
                    self.params['ScanLinePeriod'] = float(element['value'])

                if element['key'] == 'zDevice':
                    self.params['ZDevice'] = int(element['value'])

            self.video_params = {'MultiplanePrompt': seqinfo['type'],
                                 'ParameterSet': seqinfo['Childs']["Frame"]['parameterSet'],
                                 'RedChannelName': 'No Channel',
                                 'GreenChannelName': 'No Channel',
                                 'FrameNumber': int(len([x for x in seqinfo['Childs'] if 'Frame' in x])),
                                 'PlaneNumber': 'Single',
                                 'PlanePositionsOBJ': self.params['PositionZphysical'],
                                 'PlanePositionsETL': self.params['PositionZETL'],
                                 'Planepowers': self.params['ImagingLaserPower'],
                                 'XPositions': self.params['PositionX'],
                                 'YPositions': self.params['PositionY'],
                                 'pmtGains_Red': self.params['PMTGainRed'],
                                 'pmtGains_Green': self.params['PMTGainGreen'],

                                 }

            # MultiPlane=0
            SingleChannel = 0
            # %% here to create the full frame by frame volume by volume
            if len(list(self.full_metadata)) > 6 and (
                    self.video_params['MultiplanePrompt'] == "TSeries ZSeries Element" or self.video_params[
                'MultiplanePrompt'] == "AtlasVolume"):
                # FirstVolumeMetadat=root[2]
                FirstVolumeMetadat = self.full_metadata['Sequence']

                del self.video_params['FrameNumber']

                # self.video_params['VolumeNumber']=int(len(list(root.findall('Sequence'))))
                self.video_params['VolumeNumber'] = int(
                    len([key for key in self.full_metadata.keys() if 'Sequence' in key]))

                # MultiPlane=1
                self.all_volumes = []
                volumes = {key: volume for key, volume in self.full_metadata.items() if 'Sequence' in key}

                if self.video_params['MultiplanePrompt'] == "AtlasVolume":
                    self.video_params['StageGridYOverlap'] = volumes[list(volumes.keys())[0]]['xYStageGridYOverlap']
                    self.video_params['StageGridXOverlap'] = volumes[list(volumes.keys())[0]]['xYStageGridXOverlap']
                    self.video_params['StageGridOverlapPercentage'] = volumes[list(volumes.keys())[0]][
                        'xYStageGridOverlapPercentage']
                    self.video_params['StageGridNumYPositions'] = volumes[list(volumes.keys())[0]][
                        'xYStageGridNumYPositions']
                    self.video_params['StageGridNumXPositions'] = volumes[list(volumes.keys())[0]][
                        'xYStageGridNumXPositions']

                for i, volume in enumerate(volumes.values()):
                    if i == len(volumes.values()):
                        print('x')
                    all_planes = {}
                    planes = {key: plane for key, plane in volume['Childs'].items() if 'Frame' in key}
                    for i, plane in enumerate(planes.values()):
                        iplane = {}
                        iplane['scanLinePeriod'] = 'Default_' + str(self.params['ScanLinePeriod'])
                        iplane['absoluteTime'] = float(plane['absoluteTime'])
                        iplane['index'] = int(plane['index'])
                        iplane['relativeTime'] = float(plane['relativeTime'])

                        iplane['LastGoodFrame'] = int(plane['ExtraParameters']['lastGoodFrame'])
                        ExtraMetadata = plane['PVStateShard']
                        iplane['ImagingSlider'] = self.params['ImagingLaserPower']
                        iplane['UncagingSlider'] = self.params['UncagingLaserPower']
                        iplane['XAxis'] = 'Default_' + str(self.params['PositionX'])
                        iplane['YAxis'] = 'Default_' + str(self.params['PositionY'])
                        iplane['ObjectiveZ'] = 'Default_' + str(self.video_params['PlanePositionsOBJ'])
                        iplane['ETLZ'] = 'Default_' + str(self.video_params['PlanePositionsETL'])

                        # plane['File']['channelName']
                        # plane['File_1']['channelName']

                        for element in ExtraMetadata['Childs'].values():

                            if 'framePeriod' in element.values():
                                iplane['framePeriod'] = float(element['value'])

                            if 'scanLinePeriod' in element.values():
                                iplane['scanLinePeriod'] = float(element['value'])

                            if 'laserPower' in element.values():
                                for element2 in element.values():
                                    if isinstance(element2, dict):
                                        if 'Imaging' in element2.values():
                                            iplane['ImagingSlider'] = float(element2['value'])
                                        if 'Uncaging' in element2.values():
                                            iplane['UncagingSlider'] = float(element2['value'])

                            if 'positionCurrent' in element.values():
                                for element2 in element.values():
                                    if isinstance(element2, dict):
                                        if 'XAxis' in element2.values():
                                            iplane['XAxis'] = float(
                                                element2['Childs'][list(element2['Childs'].keys())[0]]['value'])
                                        if 'YAxis' in element2.values():
                                            iplane['YAxis'] = float(
                                                element2['Childs'][list(element2['Childs'].keys())[0]]['value'])

                                        if 'ZAxis' in element2.values():
                                            iplane['ObjectiveZ'] = float(
                                                element2['Childs'][list(element2['Childs'].keys())[0]]['value'])
                                            iplane['ETLZ'] = float(
                                                element2['Childs'][list(element2['Childs'].keys())[1]]['value'])

                            if 'pmtGain' in element.values():
                                for element2 in element.values():
                                    if isinstance(element2, dict):
                                        iplane['pmtGain_' + element2['description']] = float(element2['value'])

                        all_planes[i] = iplane
                    # this is because sometime sthe first xy positions were workng

                    if i != len(volumes.values()):
                        if len(all_planes.keys()) > 1:
                            all_planes[list(all_planes.keys())[0]]['XAxis'] = all_planes[list(all_planes.keys())[1]][
                                'XAxis']
                            all_planes[list(all_planes.keys())[0]]['YAxis'] = all_planes[list(all_planes.keys())[1]][
                                'YAxis']
                    self.all_volumes.append(all_planes)

                    # this is because i had metadata with 1 extra volume with 2 planes only
                if len(self.all_volumes) > 1:
                    if len(self.all_volumes[-1]) != len(self.all_volumes[-2]):
                        self.all_volumes.pop(-1)
                        self.video_params['VolumeNumber'] = self.video_params['VolumeNumber'] - 1

            elif len(list(self.full_metadata)) <= 6 and self.video_params[
                'MultiplanePrompt'] == "TSeries ZSeries Element":
                # FirstVolumeMetadat=root[2]
                FirstVolumeMetadat = self.full_metadata['Sequence']

                del self.video_params['FrameNumber']
                # self.video_params['VolumeNumber']=int(len(list(root.findall('Sequence'))))
                self.video_params['VolumeNumber'] = int(
                    len([key for key in self.full_metadata.keys() if 'Sequence' in key]))

                # MultiPlane=1
                self.all_volumes = []
                volumes = {key: volume for key, volume in self.full_metadata.items() if 'Sequence' in key}

                if self.video_params['MultiplanePrompt'] == "AtlasVolume":
                    self.video_params['StageGridYOverlap'] = volumes[list(volumes.keys())[0]]['xYStageGridYOverlap']
                    self.video_params['StageGridXOverlap'] = volumes[list(volumes.keys())[0]]['xYStageGridXOverlap']
                    self.video_params['StageGridOverlapPercentage'] = volumes[list(volumes.keys())[0]][
                        'xYStageGridOverlapPercentage']
                    self.video_params['StageGridNumYPositions'] = volumes[list(volumes.keys())[0]][
                        'xYStageGridNumYPositions']
                    self.video_params['StageGridNumXPositions'] = volumes[list(volumes.keys())[0]][
                        'xYStageGridNumXPositions']

                for i, volume in enumerate(volumes.values()):
                    all_planes = {}
                    planes = {key: plane for key, plane in volume['Childs'].items() if 'Frame' in key}
                    for i, plane in enumerate(planes.values()):
                        iplane = {}
                        iplane['absoluteTime'] = float(plane['absoluteTime'])
                        iplane['index'] = int(plane['index'])
                        iplane['relativeTime'] = float(plane['relativeTime'])
                        iplane['framePeriod'] = 'Default_' + str(self.params['framePeriod'])

                        iplane['LastGoodFrame'] = 'Default_0'
                        iplane['scanLinePeriod'] = 'Default_' + str(self.params['ScanLinePeriod'])

                        iplane['LastGoodFrame'] = int(plane['ExtraParameters']['lastGoodFrame'])
                        ExtraMetadata = plane['PVStateShard']
                        iplane['ImagingSlider'] = self.params['ImagingLaserPower']
                        iplane['UncagingSlider'] = self.params['UncagingLaserPower']
                        iplane['XAxis'] = 'Default_' + str(self.params['PositionX'])
                        iplane['YAxis'] = 'Default_' + str(self.params['PositionY'])
                        iplane['ObjectiveZ'] = 'Default_' + str(self.video_params['PlanePositionsOBJ'])
                        iplane['ETLZ'] = 'Default_' + str(self.video_params['PlanePositionsETL'])

                        # plane['File']['channelName']
                        # plane['File_1']['channelName']

                        for element in ExtraMetadata['Childs'].values():

                            if 'framePeriod' in element.values():
                                iplane['framePeriod'] = float(element['value'])

                            if 'scanLinePeriod' in element.values():
                                iplane['scanLinePeriod'] = float(element['value'])

                            if 'laserPower' in element.values():
                                for element2 in element.values():
                                    if isinstance(element2, dict):
                                        if 'Imaging' in element2.values():
                                            iplane['ImagingSlider'] = float(element2['value'])
                                        if 'Uncaging' in element2.values():
                                            iplane['UncagingSlider'] = float(element2['value'])

                            if 'positionCurrent' in element.values():
                                for element2 in element.values():
                                    if isinstance(element2, dict):
                                        if 'XAxis' in element2.values():
                                            iplane['XAxis'] = float(
                                                element2['Childs'][list(element2['Childs'].keys())[0]]['value'])
                                        if 'YAxis' in element2.values():
                                            iplane['YAxis'] = float(
                                                element2['Childs'][list(element2['Childs'].keys())[0]]['value'])

                                        if 'ZAxis' in element2.values():
                                            iplane['ObjectiveZ'] = float(
                                                element2['Childs'][list(element2['Childs'].keys())[0]]['value'])
                                            iplane['ETLZ'] = float(
                                                element2['Childs'][list(element2['Childs'].keys())[1]]['value'])

                            if 'pmtGain' in element.values():
                                for element2 in element.values():
                                    if isinstance(element2, dict):
                                        iplane['pmtGain_' + element2['description']] = float(element2['value'])

                        all_planes[i] = iplane
                    # this is because sometime sthe first xy positions were workng
                    all_planes[list(all_planes.keys())[0]]['XAxis'] = all_planes[list(all_planes.keys())[1]]['XAxis']
                    all_planes[list(all_planes.keys())[0]]['YAxis'] = all_planes[list(all_planes.keys())[1]]['YAxis']
                    self.all_volumes.append(all_planes)


            elif self.video_params['MultiplanePrompt'] == 'AtlasPreview' or self.video_params[
                'MultiplanePrompt'] == 'AtlasOverview':

                seqinfo = self.full_metadata['Sequence']
                self.all_frames = []
                self.video_params['StageGridYOverlap'] = seqinfo['xYStageGridYOverlap']
                self.video_params['StageGridXOverlap'] = seqinfo['xYStageGridXOverlap']
                self.video_params['StageGridOverlapPercentage'] = seqinfo['xYStageGridOverlapPercentage']
                self.video_params['StageGridNumYPositions'] = seqinfo['xYStageGridNumYPositions']
                self.video_params['StageGridNumXPositions'] = seqinfo['xYStageGridNumXPositions']

                for key, frame in seqinfo['Childs'].items():
                    if 'Frame' in key:
                        iframe = {}
                        iframe['LastGoodFrame'] = 'Default_0'
                        iframe['scanLinePeriod'] = 'Default_' + str(self.params['ScanLinePeriod'])
                        iframe['absoluteTime'] = float(frame['absoluteTime'])
                        iframe['index'] = int(frame['index'])
                        iframe['relativeTime'] = float(frame['relativeTime'])
                        if 'ExtraParameters' in frame:
                            iframe['LastGoodFrame'] = int(frame['ExtraParameters']['lastGoodFrame'])
                        ExtraMetadata = frame['PVStateShard']

                        for element in ExtraMetadata['Childs'].values():

                            if 'framePeriod' in element.values():
                                iframe['framePeriod'] = float(element['value'])

                            if 'scanLinePeriod' in element.values():
                                iframe['scanLinePeriod'] = float(element['value'])

                            if 'laserPower' in element.values():
                                for element2 in element.values():
                                    if isinstance(element2, dict):
                                        if 'Imaging' in element2.values():
                                            iframe['ImagingSlider'] = float(element2['value'])
                                        if 'Uncaging' in element2.values():
                                            iframe['UncagingSlider'] = float(element2['value'])

                            if 'positionCurrent' in element.values():
                                for element2 in element.values():
                                    if isinstance(element2, dict):
                                        if 'XAxis' in element2.values():
                                            iframe['XAxis'] = float(
                                                element2['Childs'][list(element2['Childs'].keys())[0]]['value'])
                                        if 'YAxis' in element2.values():
                                            iframe['YAxis'] = float(
                                                element2['Childs'][list(element2['Childs'].keys())[0]]['value'])

                                        if 'ZAxis' in element2.values():
                                            iframe['ObjectiveZ'] = float(
                                                element2['Childs'][list(element2['Childs'].keys())[0]]['value'])
                                            iframe['ETLZ'] = float(
                                                element2['Childs'][list(element2['Childs'].keys())[1]]['value'])

                        self.all_frames.append(iframe)


            else:
                seqinfo = self.full_metadata['Sequence']
                self.all_frames = []
                for key, frame in seqinfo['Childs'].items():
                    if 'Frame' in key:
                        iframe = {}
                        iframe['framePeriod'] = 'Default_' + str(self.params['framePeriod'])
                        iframe['LastGoodFrame'] = 'Default_0'
                        iframe['scanLinePeriod'] = 'Default_' + str(self.params['ScanLinePeriod'])
                        iframe['absoluteTime'] = float(frame['absoluteTime'])
                        iframe['index'] = int(frame['index'])
                        iframe['relativeTime'] = float(frame['relativeTime'])
                        if 'ExtraParameters' in frame:
                            iframe['LastGoodFrame'] = int(frame['ExtraParameters']['lastGoodFrame'])

                        if 'PVStateShard' in frame:
                            ExtraMetadata = frame['PVStateShard']
                            iframe['framePeriod'] = float(ExtraMetadata['Childs']['PVStateValue']['value'])
                            if len(ExtraMetadata['Childs']) > 1:
                                if 'value' in ExtraMetadata['Childs'][list(ExtraMetadata['Childs'].keys())[-1]]:
                                    iframe['scanLinePeriod'] = float(
                                        ExtraMetadata['Childs'][list(ExtraMetadata['Childs'].keys())[-1]]['value'])
                        self.all_frames.append(iframe)

            # %% doind video params

            if self.video_params['MultiplanePrompt'] == "AtlasVolume":
                self.video_params['XPositions'] = [position[list(position.keys())[0]]['XAxis'] for position in
                                                   self.all_volumes]
                self.video_params['YPositions'] = [position[list(position.keys())[0]]['YAxis'] for position in
                                                   self.all_volumes]

            elif self.video_params['MultiplanePrompt'] == "TSeries ZSeries Element":
                self.video_params['XPositions'] = self.all_volumes[0][list(self.all_volumes[0].keys())[0]]['XAxis']
                self.video_params['YPositions'] = self.all_volumes[0][list(self.all_volumes[0].keys())[0]]['YAxis']


            elif self.video_params['MultiplanePrompt'] == 'AtlasPreview' or self.video_params[
                'MultiplanePrompt'] == 'AtlasOverview':
                self.video_params['XPositions'] = [frame['XAxis'] for frame in self.all_frames]
                self.video_params['YPositions'] = [frame['YAxis'] for frame in self.all_frames]

            if self.video_params['MultiplanePrompt'] == "TSeries ZSeries Element" or self.video_params[
                'MultiplanePrompt'] == "AtlasVolume":

                self.video_params['FullAcquisitionTime'] = self.all_volumes[-1][list(self.all_volumes[-1].keys())[-1]][
                    'relativeTime']
                # self.video_params['PlaneNumber']=len(list(FirstVolumeMetadat.findall('Frame')))
                self.video_params['PlaneNumber'] = len(
                    [key for key in FirstVolumeMetadat['Childs'].keys() if 'Frame' in key])

                self.video_params['PlanePositionsOBJ'] = [self.all_volumes[0][key]['ObjectiveZ'] for key in
                                                          self.all_volumes[0].keys()]
                self.video_params['PlanePositionsETL'] = [self.all_volumes[0][key]['ETLZ'] for key in
                                                          self.all_volumes[0].keys()]
                self.video_params['Planepowers'] = [self.all_volumes[0][key]['ImagingSlider'] for key in
                                                    self.all_volumes[0].keys()]
                self.video_params['pmtGains_Red'] = [self.all_volumes[0][key]['pmtGain_Red'] for key in
                                                     self.all_volumes[0].keys() if
                                                     'pmtGain_Red' in self.all_volumes[0][key].keys()]
                self.video_params['pmtGains_Green'] = [self.all_volumes[0][key]['pmtGain_Green'] for key in
                                                       self.all_volumes[0].keys() if
                                                       'pmtGain_Green' in self.all_volumes[0][key].keys()]

                if not self.video_params['pmtGains_Red']:
                    self.video_params['pmtGains_Red'] = self.params['PMTGainRed']
                if not self.video_params['pmtGains_Green']:
                    self.video_params['pmtGains_Green'] = self.params['PMTGainGreen']

                self.video_params['relativeTimes'] = [
                    [position[key]['relativeTime'] for key in self.all_volumes[0].keys()] for position in
                    self.all_volumes]
                self.video_params['absoluteTimes'] = [
                    [position[key]['absoluteTime'] for key in self.all_volumes[0].keys()] for position in
                    self.all_volumes]
                self.video_params['scanLinePeriods'] = [
                    [position[key]['scanLinePeriod'] for key in self.all_volumes[0].keys()] for position in
                    self.all_volumes]
                self.video_params['framePeriods'] = [
                    [position[key]['framePeriod'] for key in self.all_volumes[0].keys()] for position in
                    self.all_volumes]

                self.video_params['lastGoodFrames'] = [
                    [position[key]['LastGoodFrame'] for key in self.all_volumes[0].keys()] for position in
                    self.all_volumes]

            else:

                self.video_params['FullAcquisitionTime'] = self.all_frames[-1]['relativeTime']
                self.video_params['relativeTimes'] = [position['relativeTime'] for position in self.all_frames]
                self.video_params['absoluteTimes'] = [position['absoluteTime'] for position in self.all_frames]
                self.video_params['scanLinePeriods'] = [position['scanLinePeriod'] for position in self.all_frames]
                self.video_params['framePeriods'] = [position['framePeriod'] for position in self.all_frames]
                self.video_params['lastGoodFrames'] = [position['LastGoodFrame'] for position in self.all_frames]

            if seqinfo['Childs']['PVStateShard'] and self.video_params['MultiplanePrompt'] == "AtlasVolume":
                SingleChannel = 0
                files = {key: val for key, val in seqinfo['Childs']['Frame'].items() if 'File' in key}
                for chan in files.values():
                    ChannelName = chan['filename']
                    if 'Green' in chan.values():
                        self.video_params['GreenChannelName'] = ChannelName
                    elif 'Red' in chan.values():
                        self.video_params['RedChannelName'] = ChannelName
                if not all(self.video_params['RedChannelName'] and self.video_params['GreenChannelName']):
                    SingleChannel = 1

            elif seqinfo['Childs']['PVStateShard'] and not self.video_params['MultiplanePrompt'] == "AtlasVolume":
                SingleChannel = 0
                files = {key: val for key, val in seqinfo['Childs']['Frame'].items() if 'File' in key}
                for chan in files.values():
                    ChannelName = chan['filename']
                    if 'Green' in chan.values():
                        self.video_params['GreenChannelName'] = ChannelName
                    elif 'Red' in chan.values():
                        self.video_params['RedChannelName'] = ChannelName
                if not all(self.video_params['RedChannelName'] and self.video_params['GreenChannelName']):
                    SingleChannel = 1

            else:
                SingleChannel = 0
                files = {key: val for key, val in seqinfo['Childs']['Frame'].items() if 'File' in key}
                for chan in files.values():
                    ChannelName = chan['filename']
                    if 'Green' in chan.values():
                        self.video_params['GreenChannelName'] = ChannelName
                    elif 'Red' in chan.values():
                        self.video_params['RedChannelName'] = ChannelName
                if not all(self.video_params['RedChannelName'] and self.video_params['GreenChannelName']):
                    SingleChannel = 1

        if self.all_frames:
            self.imaging_metadata = [self.params, self.video_params, self.all_frames]
        if self.all_volumes:
            self.imaging_metadata = [self.params, self.video_params, self.all_volumes]

    # %% Plotting
    def plotting(self):

        f, axs = plt.subplots(5, 1, sharex=True)
        axs[0].plot(np.array(self.video_params['relativeTimes']).flatten())
        axs[1].plot(np.array(self.video_params['absoluteTimes']).flatten())
        axs[2].plot(np.array(self.video_params['scanLinePeriods']).flatten())
        axs[3].plot(np.array(self.video_params['framePeriods']).flatten())
        axs[4].plot(np.array(self.video_params['lastGoodFrames']).flatten())

        f, axs = plt.subplots(2, 1, sharex=True)
        axs[0].plot(np.array(self.video_params['XPositions']).flatten(), 'x')
        axs[1].plot(np.array(self.video_params['YPositions']).flatten(), 'x')

        f, axs = plt.subplots(3, 1, sharex=True)
        axs[0].plot(np.array(self.video_params['Planepowers']).flatten(), 'x')
        axs[1].plot(np.array(self.video_params['pmtGains_Red']).flatten(), 'x')
        axs[2].plot(np.array(self.video_params['pmtGains_Green']).flatten(), 'x')

    # %% manual and photostim metadata
    def add_metadata_manually(self):
        # module_logger.info('getting manual meta1data')

        self.imaging_metadata = [{}, {}, []]
        aquisition_to_process = os.path.split(self.imaging_metadata_file)[0]
        temp_path = os.path.split(os.path.split(self.imaging_metadata_file)[0])[0]
        aquisition_date = aquisition_to_process[temp_path.find('\SP') - 8:temp_path.find('\SP') - 1]
        formated_aquisition_date = aquisition_date

        self.imaging_metadata[0]['Date'] = formated_aquisition_date
        self.imaging_metadata[0]['AquisitionName'] = os.path.split(aquisition_to_process)[1]

        self.imaging_metadata[1]['MultiplanePrompt'] = select_values_gui(
            ["TSeries ZSeries Element", "TSeries ImageSequence Element"])

        mtdata = manually_get_some_metadata()

        self.imaging_metadata[1]['Plane Number'] = int(mtdata[0])
        self.imaging_metadata[1]['Volume Number'] = int(mtdata[2])
        self.imaging_metadata[1]['Frame Number'] = int(mtdata[2])
        self.imaging_metadata[0]['Lines per Frame'] = mtdata[1]
        self.imaging_metadata[0]['Pixels Per Line'] = mtdata[1]

    def process_photostim_metadata(self):

        Experiment = {'Iterations': int(self.full_mark_points_metadata['Iterations']),
                      'Iteration Delay': self.full_mark_points_metadata['IterationDelay'],

                      'PhotoStimSeries': {}}
        experiment_count = 1
        for name, photstim_inf in self.full_mark_points_metadata.items():

            if 'PVMarkPointElement' in name and ('Indices' in photstim_inf['Childs']['PVGalvoPointElement'].keys()):

                Experiment['PhotoStimSeries']['PhotostimExperiment_' + str(experiment_count)] = {}
                sequence = {'TriggerFrequency': photstim_inf['TriggerFrequency'],
                            'TriggerSelection': photstim_inf['TriggerSelection'],
                            'Repetitions': float(photstim_inf['Repetitions']),
                            'Point Order': photstim_inf['Childs']['PVGalvoPointElement']['Indices'],
                            'StimDuration': float(photstim_inf['Childs']['PVGalvoPointElement']['Duration']),
                            'InterpointDuration': float(
                                photstim_inf['Childs']['PVGalvoPointElement']['InterPointDelay']),
                            'RelativeDelay': float(photstim_inf['Childs']['PVGalvoPointElement']['InitialDelay']),
                            'SpiralRevolutions': float(
                                photstim_inf['Childs']['PVGalvoPointElement']['SpiralRevolutions']),
                            'AllPointsAtOnce': photstim_inf['Childs']['PVGalvoPointElement']['AllPointsAtOnce'],
                            'RepTime': float(photstim_inf['Childs']['PVGalvoPointElement']['InterPointDelay']) + float(
                                photstim_inf['Childs']['PVGalvoPointElement']['Duration']),
                            'FullTrialDuration': (float(
                                photstim_inf['Childs']['PVGalvoPointElement']['InterPointDelay']) + float(
                                photstim_inf['Childs']['PVGalvoPointElement']['Duration'])) * float(
                                photstim_inf['Repetitions']),
                            'RepFrequency': 1 / (
                                        float(photstim_inf['Childs']['PVGalvoPointElement']['InterPointDelay']) + float(
                                    photstim_inf['Childs']['PVGalvoPointElement']['Duration']))
                            }
                Experiment['PhotoStimSeries']['PhotostimExperiment_' + str(experiment_count)]['sequence'] = sequence
                Experiment['PhotoStimSeries']['PhotostimExperiment_' + str(experiment_count)]['points'] = {}
                point_count = 1
                for name, point in photstim_inf['Childs']['PVGalvoPointElement'].items():
                    if isinstance(point, dict):
                        point = {'index': point['Index'],
                                 'x_pos': point['X'],
                                 'y_pos': point['Y'],
                                 'spiral': point['IsSpiral'],
                                 'spiral_width': point['SpiralWidth'],
                                 'spiral_height': point['SpiralHeight'],
                                 'spiral_size_microns': point['SpiralSizeInMicrons']}
                        Experiment['PhotoStimSeries']['PhotostimExperiment_' + str(experiment_count)]['points'][
                            'Point_' + str(point_count)] = point
                        point_count = point_count + 1
                experiment_count = experiment_count + 1
        self.mark_points_experiment = Experiment

    def create_photostim_sequence(photmet, aqumet, pokels_signal=[]):

        timestamps = [tuple(l) for l in aqumet[2]]

        dt = np.dtype('double,double,double')
        zz = np.array(timestamps, dtype=dt)
        zz.dtype.names = ['Frame', 'AbsTime', 'RelTime']
        dd = zz.view((float, len(zz.dtype.names)))

        ddd = dd[:, [0, 2]]
        ddd_miliseconds = ddd
        ddd_miliseconds[:, 1] = ddd_miliseconds[:, 1] * 1000
        stim_delays = {}
        for key in photmet['PhotoStimSeries']:
            stim_delays[key] = photmet['PhotoStimSeries'][key]['sequence']['RelativeDelay']
            stim_delays[key] = int(stim_delays[key])
            stim_delays[key] = stim_delays[key] / 1000
        exp_total_duration = {}
        for key in photmet['PhotoStimSeries']:
            exp_total_duration[key] = (int(photmet['PhotoStimSeries'][key]['sequence']['StimDuration']) +
                                       int(photmet['PhotoStimSeries'][key]['sequence']['InterpointDuration'])) * \
                                      len((photmet['PhotoStimSeries'][key]['points'])) * \
                                      int(photmet['PhotoStimSeries'][key]['sequence']['Repetitions'])

        def dict_zip(*dicts):
            all_keys = {k for d in dicts for k in d.keys()}
            return {k: [d[k] for d in dicts if k in d] for k in all_keys}

        zzz = dict_zip(stim_delays, exp_total_duration)

        sortedDict = dict(sorted(zzz.items(), key=lambda x: x[0].lower()))

        zzz_sorted_list = list(sortedDict.values())
        stim_table = []

        for i, j in enumerate(zzz_sorted_list):
            if i == 0:
                stim_info = []
                stim_info.append(0)
                stim_info.append(zzz_sorted_list[i][0])
                stim_info.append(stim_info[0] + stim_info[1])
                stim_info.append(round(zzz_sorted_list[i][1] / 1000, 2))
                stim_info.append(stim_info[2] + round(stim_info[3], 2))
                stim_table.append(stim_info)

            else:
                stim_info = []
                stim_info.append(round(stim_table[i - 1][4], 2))
                stim_info.append(zzz_sorted_list[i][0])
                stim_info.append(stim_info[0] + stim_info[1])
                stim_info.append(round(zzz_sorted_list[i][1] / 1000, 2))
                stim_info.append(round(stim_info[2] + stim_info[3], 2))
                stim_table.append(stim_info)

        # %% fill artifical pockels cell signal
        if not pokels_signal:
            time_s = round(int(aqumet[1]['Frame Number']) * float(aqumet[1]['FramePeriod']) * int(
                aqumet[0]['RasterAveraging']))  # s
            freq_s = 1000  # hz
            samples = time_s * freq_s
            high_freq_signals = np.linspace(0, samples, samples, dtype='int_')
            high_freq_signals
            stim_table
            stim_table_ms = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

            for i, j in enumerate(stim_table):
                for k, l in enumerate(stim_table[i]):
                    stim_table_ms[i][k] = stim_table[i][k] * 1000

            index_to_fill = []
            len(high_freq_signals)
            for m, n in enumerate(stim_table_ms):
                index_to_fill.append(np.arange(stim_table_ms[m][2], stim_table_ms[m][4]).astype(int))

            new_column = np.zeros(len(high_freq_signals)).astype(int)
            high_freq_signals_full = np.vstack((high_freq_signals, new_column)).transpose()
            cum_indx_fil = []
            for o, p in enumerate(index_to_fill):
                cum_indx_fil = index_to_fill[o][len(index_to_fill[o]) - 1]
                if cum_indx_fil > samples:
                    break
                else:
                    high_freq_signals_full[index_to_fill[o], 1] = 5

        # fig, ax = plt.subplots()  # Create a figure containing a single axes.
        # ax.plot(high_freq_signals_full[:,0], high_freq_signals_full[:,1])
        resampled_laser = sp.signal.resample(high_freq_signals_full[:, 1], len(ddd))
        # fig, ax = plt.subplots()  # Create a figure containing a single axes.
        # ax.plot(resampled_laser[:])
        filtered_resampled_laser = sp.signal.medfilt(resampled_laser, 5)
        # fig, ax = plt.subplots()  # Create a figure containing a single axes.
        # ax.plot(filtered_resampled_laser[:])
        artifact_idx = np.where(filtered_resampled_laser > 1)
        PhotoStimSeriesArtifact = {'artifact_idx': artifact_idx, 'stim_table_ms': stim_table_ms,
                                   'Processed_Pockels_Signal': filtered_resampled_laser}

        return PhotoStimSeriesArtifact

    # %%   new functions
    def transfer_metadata(self):

        self.metadata_raw_files_full_path = [file for file in
                                             glob.glob(self.acquisition_directory_raw + '\\**', recursive=False) if
                                             '.xml' in file]
        self.transfered_metadata_paths = []
        for file in self.metadata_raw_files_full_path:
            if not os.path.isfile(os.path.join(self.temporary_path, os.path.split(file)[1])):
                shutil.copy(file, self.temporary_path)
                self.transfered_metadata_paths.append(os.path.join(self.temporary_path, file))

    def check_metadata_in_folder(self):
        if self.acquisition_directory_raw:
            xmlfiles = glob.glob(self.acquisition_directory_raw + '\\**.xml')
            for xml in xmlfiles:
                if 'Cycle' not in xml:
                    self.imaging_metadata_file = xml
                if 'VoltageRecording' in xml:
                    self.voltage_file = xml
                if 'MarkPoints' in xml:
                    self.photostim_file = xml
                    # if 'VoltageRecording' in xml:
                #     face_camera_metadata_path=xml
                if 'VoltageOutput' in xml:
                    self.voltage_output = xml

    def save_metadata_as_json(self):
        if not os.path.isfile(self.read_voltage_metadata_path):
            if self.full_voltage_recording_metadata:
                with open(self.read_voltage_metadata_path, 'w') as fout:
                    json.dump(self.full_voltage_recording_metadata, fout)

        if not os.path.isfile(self.read_metadata_path):
            with open(self.read_metadata_path, 'w') as fout:
                json.dump(self.full_metadata, fout)

    def read_json_metadata(self):
        if os.path.isfile(self.read_metadata_path):
            with open(self.read_metadata_path) as json_file:
                self.full_metadata = json.load(json_file)
        elif self.imaging_metadata_file:
            if os.path.isfile(self.imaging_metadata_file):
                self.process_metadata()

        if os.path.isfile(self.read_voltage_metadata_path):
            with open(self.read_voltage_metadata_path) as json_file:
                self.full_voltage_recording_metadata = json.load(json_file)

    def get_all_metadata_from_database(self):

        self.aquisition_object.full_database_dictionary
        self.acquisition_metadata = self.aquisition_object.full_database_dictionary['Acq'].to_dict('records')[0]

        self.imaging_metadata_database = self.aquisition_object.full_database_dictionary['Imaging'].to_dict('records')
        if self.imaging_metadata_database:
            self.translated_imaging_metadata = copy.copy(self.imaging_metadata_database[0])
            self.translated_imaging_metadata['CorrectedObjectivePositions'] = self.translated_imaging_metadata[
                'ObjectivePositions']
            self.translated_imaging_metadata['CorrectedETLPositions'] = self.translated_imaging_metadata['ETLPositions']
            self.translated_imaging_metadata.pop('ID')
            self.translated_imaging_metadata.pop('AcquisitionID')
            self.translated_imaging_metadata.pop('ImagingFilename')
            self.translated_imaging_metadata.pop('ImagingFullFilePath')
            self.translated_imaging_metadata.pop('IsSlowStorage')
            self.translated_imaging_metadata.pop('IsWorkingStorage')
            self.translated_imaging_metadata.pop('SlowStoragePath')
            self.translated_imaging_metadata.pop('ToDoDeepCaiman')
            self.translated_imaging_metadata.pop('WorkingStoragePath')
            self.translated_imaging_metadata = dict(
                sorted(self.translated_imaging_metadata.items(), key=lambda x: x[0].lower()))

        self.facecam_metadata = self.aquisition_object.full_database_dictionary['FaceCam'].to_dict('records')
        if self.facecam_metadata:
            self.facecam_metadata = self.facecam_metadata[0]

        self.visstim_metadata = self.aquisition_object.full_database_dictionary['VisStim'].to_dict('records')
        if self.visstim_metadata:
            self.visstim_metadata = self.visstim_metadata[0]

    def translate_metadata(self):

        self.translated_imaging_metadata = {
            'PowerSetting': np.nan,
            'Objective': np.nan,
            'PMT1GainRed': np.nan,
            'PMT2GainGreen': np.nan,
            'FrameAveraging': np.nan,
            'ObjectivePositions': np.nan,
            'ETLPositions': np.nan,
            'PlaneNumber': np.nan,
            'TotalVolumes': np.nan,
            'IsETLStack': np.nan,
            'IsObjectiveStack': np.nan,
            'InterFramePeriod': np.nan,
            'FinalVolumePeriod': np.nan,
            'FinalFrequency': np.nan,
            'TotalFrames': np.nan,
            'FOVNumber': np.nan,
            'ExcitationWavelength': np.nan,
            'CoherentPower': np.nan,
            'CalculatedPower': np.nan,
            'Comments': np.nan,
            'IsChannel1Red': np.nan,
            'IsChannel2Green': np.nan,
            'IsGalvo': np.nan,
            'IsResonant': np.nan,
            'Resolution': np.nan,
            'DwellTime': np.nan,
            'Multisampling': np.nan,
            'BitDepth': np.nan,
            'LinePeriod': np.nan,
            'FramePeriod': np.nan,
            'FullAcquisitionTime': np.nan,
            'RedFilter': np.nan,
            'GreenFilter': np.nan,
            'DichroicBeamsplitter': np.nan,
            'IsBlockingDichroic': np.nan,
            'OverlapPercentage': np.nan,
            'AtlasOverlap': np.nan,
            'OverlapPercentageMetadata': np.nan,
            'AtlasDirection': np.nan,
            'AtlasZStructure': np.nan,
            'AtlasGridSize': np.nan,
            'CorrectedObjectivePositions': np.nan,
            'CorrectedETLPositions': np.nan,
            'ImagingTime': np.nan,
            'IsVoltagERecording': np.nan,
            'MicronsPerPixelX': np.nan,
            'MicronsPerPixelY': np.nan,
            'Xpositions': np.nan,
            'Ypositions': np.nan,
            'Zoom': np.nan,
            'VoltageRecordingChannels': np.nan,
            'VoltageRecordingFrequency': np.nan,
            'Is10MinRec': np.nan,
            'IsGoodObjective': np.nan,
        }

        self.translated_imaging_metadata['PowerSetting'] = self.imaging_metadata[1]['Planepowers']
        self.translated_imaging_metadata['Objective'] = self.imaging_metadata[0]['Objective']
        self.translated_imaging_metadata['PMT1GainRed'] = self.imaging_metadata[1]['pmtGains_Red']
        self.translated_imaging_metadata['PMT2GainGreen'] = self.imaging_metadata[1]['pmtGains_Green']
        self.translated_imaging_metadata['FrameAveraging'] = self.imaging_metadata[0]['RasterAveraging']
        self.translated_imaging_metadata['ObjectivePositions'] = self.imaging_metadata[1]['PlanePositionsOBJ']
        self.translated_imaging_metadata['ETLPositions'] = self.imaging_metadata[1]['PlanePositionsETL']
        self.translated_imaging_metadata['Xpositions'] = self.imaging_metadata[1]['XPositions']
        self.translated_imaging_metadata['Ypositions'] = self.imaging_metadata[1]['YPositions']
        self.translated_imaging_metadata['ImagingTime'] = self.imaging_metadata[0]['ImagingTime']
        self.translated_imaging_metadata['MicronsPerPixelX'] = self.imaging_metadata[0]['MicronsPerPixelX']
        self.translated_imaging_metadata['MicronsPerPixelY'] = self.imaging_metadata[0]['MicronsPerPixelY']
        self.translated_imaging_metadata['Zoom'] = self.imaging_metadata[0]['OpticalZoom']
        self.translated_imaging_metadata['CorrectedObjectivePositions'] = self.imaging_metadata[1]['PlanePositionsOBJ']
        self.translated_imaging_metadata['CorrectedETLPositions'] = self.imaging_metadata[1]['PlanePositionsETL']

        if self.imaging_metadata[1]['PlaneNumber'] == 'Single':
            self.translated_imaging_metadata['IsETLStack'] = 0
            self.translated_imaging_metadata['IsObjectiveStack'] = 0
            self.translated_imaging_metadata['PlaneNumber'] = 1
            self.translated_imaging_metadata['TotalFrames'] = self.imaging_metadata[1]['FrameNumber']
            self.translated_imaging_metadata['InterFramePeriod'] = self.imaging_metadata[0]['framePeriod'] * \
                                                                   self.translated_imaging_metadata['FrameAveraging']
            self.translated_imaging_metadata['FinalVolumePeriod'] = self.translated_imaging_metadata['InterFramePeriod']
            self.translated_imaging_metadata['FinalFrequency'] = 1 / self.translated_imaging_metadata[
                'InterFramePeriod']
            self.translated_imaging_metadata['TotalVolumes'] = self.translated_imaging_metadata['TotalFrames']
        else:
            self.translated_imaging_metadata['TotalVolumes'] = self.imaging_metadata[1]['VolumeNumber']
            self.translated_imaging_metadata['IsETLStack'] = 0
            self.translated_imaging_metadata['IsObjectiveStack'] = 0
            self.translated_imaging_metadata['PlaneNumber'] = self.imaging_metadata[1]['PlaneNumber']

            self.translated_imaging_metadata['InterFramePeriod'] = self.imaging_metadata[0]['framePeriod']
            if not isinstance(self.imaging_metadata[2][0][list(self.imaging_metadata[2][0].keys())[0]]['framePeriod'],
                              str):
                self.translated_imaging_metadata['FinalVolumePeriod'] = \
                self.imaging_metadata[2][0][list(self.imaging_metadata[2][0].keys())[0]]['framePeriod'] * \
                self.translated_imaging_metadata['PlaneNumber']
            else:
                self.translated_imaging_metadata['FinalVolumePeriod'] = self.imaging_metadata[0]['framePeriod'] * \
                                                                        self.translated_imaging_metadata['PlaneNumber']

            self.translated_imaging_metadata['FinalFrequency'] = 1 / self.translated_imaging_metadata[
                'FinalVolumePeriod']
            self.translated_imaging_metadata['TotalFrames'] = self.translated_imaging_metadata['TotalVolumes'] * \
                                                              self.translated_imaging_metadata['PlaneNumber']
            self.translated_imaging_metadata['PowerSetting'] = str(self.translated_imaging_metadata['PowerSetting'])
            self.translated_imaging_metadata['CorrectedObjectivePositions'] = [float(i[8:]) if isinstance(i, str) else i
                                                                               for i in
                                                                               self.translated_imaging_metadata[
                                                                                   'ObjectivePositions']]
            self.translated_imaging_metadata['CorrectedETLPositions'] = [float(i[8:]) if isinstance(i, str) else i for i
                                                                         in self.translated_imaging_metadata[
                                                                             'ETLPositions']]
            if not all(element == self.translated_imaging_metadata['CorrectedObjectivePositions'][0] for element in
                       self.translated_imaging_metadata['CorrectedObjectivePositions']):
                self.translated_imaging_metadata['IsObjectiveStack'] = 1
            if not all(element == self.translated_imaging_metadata['CorrectedETLPositions'][0] for element in
                       self.translated_imaging_metadata['CorrectedETLPositions']):
                self.translated_imaging_metadata['IsETLStack'] = 1

        self.translated_imaging_metadata['Xpositions'] = self.imaging_metadata[1]['XPositions']
        self.translated_imaging_metadata['Ypositions'] = self.imaging_metadata[1]['YPositions']
        self.translated_imaging_metadata['ImagingTime'] = self.imaging_metadata[0]['ImagingTime']
        self.translated_imaging_metadata['MicronsPerPixelX'] = self.imaging_metadata[0]['MicronsPerPixelX']
        self.translated_imaging_metadata['MicronsPerPixelY'] = self.imaging_metadata[0]['MicronsPerPixelY']
        self.translated_imaging_metadata['Zoom'] = self.imaging_metadata[0]['OpticalZoom']

        if 'Atlas' in self.imaging_metadata[1]['MultiplanePrompt']:
            self.translated_imaging_metadata['AtlasOverlap'] = str(
                (self.imaging_metadata[1]['StageGridXOverlap'], self.imaging_metadata[1]['StageGridYOverlap']))
            self.translated_imaging_metadata['OverlapPercentageMetadata'] = self.imaging_metadata[1][
                'StageGridOverlapPercentage']
            self.translated_imaging_metadata['AtlasGridSize'] = str((self.imaging_metadata[1]['StageGridNumXPositions'],
                                                                     self.imaging_metadata[1][
                                                                         'StageGridNumYPositions']))
            self.translated_imaging_metadata['Xpositions'] = str(tuple(self.translated_imaging_metadata['Xpositions']))
            self.translated_imaging_metadata['Ypositions'] = str(tuple(self.translated_imaging_metadata['Ypositions']))

        self.translated_imaging_metadata['IsChannel1Red'] = 0
        self.translated_imaging_metadata['IsChannel2Green'] = 0

        if not self.imaging_metadata[1]['RedChannelName'] == 'No Channel':
            self.translated_imaging_metadata['IsChannel1Red'] = 1
        if not self.imaging_metadata[1]['GreenChannelName'] == 'No Channel':
            self.translated_imaging_metadata['IsChannel2Green'] = 1
        self.translated_imaging_metadata['IsGalvo'] = 1
        self.translated_imaging_metadata['IsResonant'] = 0
        if 'Resonant' in self.imaging_metadata[0]['ScanMode']:
            self.translated_imaging_metadata['IsResonant'] = 1
            self.translated_imaging_metadata['IsGalvo'] = 0
            self.translated_imaging_metadata['Multisampling'] = self.imaging_metadata[0]['ResonantSampling']

        self.translated_imaging_metadata['Resolution'] = str(self.imaging_metadata[0]['LinesPerFrame']) + 'x' + str(
            self.imaging_metadata[0]['PixelsPerLine'])
        self.translated_imaging_metadata['DwellTime'] = self.imaging_metadata[0]['dwellTime']

        self.translated_imaging_metadata['BitDepth'] = self.imaging_metadata[0]['BitDepth']

        self.translated_imaging_metadata['LinePeriod'] = self.imaging_metadata[0]['ScanLinePeriod']
        self.translated_imaging_metadata['FramePeriod'] = self.imaging_metadata[0]['framePeriod']
        self.translated_imaging_metadata['FullAcquisitionTime'] = self.imaging_metadata[1]['FullAcquisitionTime']

        self.translated_imaging_metadata['IsVoltageRecording'] = 0
        if self.full_voltage_recording_metadata:
            self.translated_imaging_metadata['IsVoltageRecording'] = 1
            self.translated_imaging_metadata['VoltageRecordingChannels'] = str(
                (self.recorded_signals and self.recorded_signals_csv))
            self.translated_imaging_metadata['VoltageRecordingFrequency'] = int(
                self.full_voltage_recording_metadata['Experiment']['Childs']['Rate']['Description'])


# %%
if __name__ == "__main__":
    temporary_path1 = r'F:\Projects\LabNY\Imaging\2022\20220428\Mice\SPMT\FOV_1\Aq_1\220428_SPMT_FOV2_AllenA_25x_920_52570_570620_without-000'
    meta = Metadata(acquisition_directory_raw=temporary_path1)
    markpoints = meta.full_mark_points_metadata














