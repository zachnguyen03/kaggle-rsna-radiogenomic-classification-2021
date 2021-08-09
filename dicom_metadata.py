import pydicom
import numpy as np


sample_path = './train/00000/FLAIR/Image-20.dcm'
dicom = pydicom.read_file(sample_path)
# print(dicom)

def get_dicom_value(x, cast=int):
    if type(x) in [pydicom.multival.MultiValue, tuple]:
        return cast(x[0])
    else:
        return cast(x)
    

def get_windowing(metadata):
    
    return 0


def get_dicom_meta(dicom):
    return {
        # 'PatientID': dicom.PatientID, # can be grouped (20-548)
        # 'StudyInstanceUID': dicom.StudyInstanceUID, # can be grouped (20-60)
        # 'SeriesInstanceUID': dicom.SeriesInstanceUID, # can be grouped (20-60)
        'WindowWidth': get_dicom_value(dicom.WindowWidth),
        'WindowCenter': get_dicom_value(dicom.WindowCenter),
        'RescaleIntercept': float(dicom.RescaleIntercept),
        'RescaleSlope': float(dicom.RescaleSlope), # all same (1.0)
    }
    
    
metadata = get_dicom_meta(dicom)
print(metadata)

# {
# 'PatientID': '00000',
# 'StudyInstanceUID': '1.2.826.0.1.3680043.8.498.21557373119637153130481842401167746353',
# 'SeriesInstanceUID': '1.2.826.0.1.3680043.8.498.75098256064787219771161442192873622098',
# 'WindowWidth': 945,
# 'WindowCenter': 472,
# 'RescaleIntercept': 0.0,
# 'RescaleSlope': 1.0
# }