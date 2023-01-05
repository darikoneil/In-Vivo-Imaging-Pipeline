from Management.Organization import Mouse
from Management.Organization import ImagingExperiment, BehavioralExperiment, ImagingBehaviorExperiment


def Test():
    MyMouse = Mouse(Directory="D:\\Theodore", Mouse="Theodore")
    MyMouse.experimental_condition = "Boy Mouse"
    MyMouse.study = "Gender Studies"
    MyMouse.study_mouse = "Boy1"
    MyMouse.create()  # This makes a folder for any lab notebook / record files, the actual directory,
    # and the organization.json by default
    MyMouse.create_experiment("E", interactive=True)
    MyMouse.create_experiment("I", Type=ImagingExperiment, interactive=True)
    return MyMouse
