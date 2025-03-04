import matplotlib.pyplot as plt
import numpy as np
import os
from oct2py import Oct2Py 
import pydicom


octave = Oct2Py()
iviolin_path = os.path.join(os.getcwd(), "iviolin")
octave.addpath(iviolin_path)

orig_path = r'G:\Cristina\Thesis\analyzer\predictions\iviolin\original\dcm'
#prediction_path = 

#iviolin short
slicesi2 = ["prediction_0000.dcm","prediction_0002.dcm", "prediction_0004.dcm", "prediction_0006.dcm", ]
originalsi2 = ["iviolin_abdo_pat1_target.dcm","iviolin_thor_acc_pat1_target.dcm", "iviolin_thor_acc_pat3_target.dcm", "iviolin_thor_nacc_pat2_target.dcm"]
roisi1 = [[158, 382, 18, 18],[327, 336, 18, 18], [367, 297, 18, 18], [356, 321 , 18, 18]]
roisi2 = [[142, 369, 18, 18],[313, 338, 18, 18], [389, 286, 18, 18], [325, 363, 18, 18]]

file_path = os.path.join(orig_path, originalsi2[0])

# Call the Octave function
f_int, MTF_int, NPS1_int, fNPS2, NPS2, cont_ROI1, cont_ROI2, MTF_area, PS_area, NPS_area, IMGreturn, Manufacturer, Model, ExposureTime, TubeCurrent, WindowWidth, WindowCenter, PredictedLabel, DecisionValue = octave.funcImageQualityOctave(
    file_path,
    roisi1[0][2],
    [roisi1[0][0], roisi1[0][1]],
    [roisi2[0][0], roisi2[0][1]],
    nout=19
)

print(f_int, MTF_int, NPS1_int, fNPS2, NPS2, cont_ROI1, cont_ROI2, MTF_area, PS_area, NPS_area, IMGreturn, Manufacturer, Model, ExposureTime, TubeCurrent, WindowWidth, WindowCenter, PredictedLabel, DecisionValue)