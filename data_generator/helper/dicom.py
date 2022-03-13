import glob

import nibabel as nib
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

import config
from data_generator.helper import image


def read_patient(name, size, segment, is_whole):
    nifti_path = config.DATA_PATH + name + "/*.nii.gz"
    nifti_file = glob.glob(nifti_path)
    img = nib.load(nifti_file[0])
    data = img.get_fdata().transpose()

    if not is_whole:
        data[data != float(segment)] = 0.
    
    mid = image.find_mid(data)
    dcm = _read_dicom(name, size, mid)
    data = image.cut(data, mid, size)

    return dcm, data


def _read_dicom(name, size, mid):
    path = config.DATA_PATH + name + "/*.dcm"
    dcm_files = glob.glob(path)
    pixel_data = []
    data = []

    for dcm_file in dcm_files:
        dataset = pydicom.dcmread(dcm_file)
        data.append(dataset)
    
    slices = sorted(data, key=lambda s: s.ImagePositionPatient[2])

    for slice in slices:
        pixel_array = slice.pixel_array
        pixel_data.append(apply_modality_lut(pixel_array, slice))
    
    pixel_arr = np.asarray(pixel_data)
    dcm_data = image.cut(pixel_arr, mid, size).astype("float32")
    dcm_data = image.normalize(dcm_data).astype("float32")
    x = np.empty((size, size, size, 1))
    x[..., 0] = dcm_data

    return x
