MRI_FOLDER = './brains_low_res/'
MASK_FOLDER = './masks/'
MAT_FOLDER = './flirt_mats/'
FLIRT_MRI_FOLDER = './flirt_brains/'
FLIRT_MASK_FOLDER = './flirt_masks/'
CONTROL_FOLDER = './fcd_classification_bank/'
CONTROL_FLIRT_FOLDER = './control_flirt/'


#pirogov images
def get_registered_img_and_mask(input_img_name, template):
    number = input_img_name.split('_')[0]
    havefcd = input_img_name.split('_')[2].split('.')[0]
    input_mask_name = f'{havefcd}_{number}_1.nii.gz'
    have_mask = os.path.isfile(os.path.join(MASK_FOLDER, input_mask_name))
    if have_mask:
        orig_mask = nib.load(os.path.join(MASK_FOLDER, input_mask_name))
    else:
        orig_mask = None


    #flirt image registration
    flt = fsl.FLIRT()
    flt.inputs.in_file = os.path.join(MRI_FOLDER, input_img_name)
    flt.inputs.reference= template.get_filename()
    flt.inputs.output_type = "NIFTI_GZ"
    flt.inputs.out_file = os.path.join(FLIRT_MRI_FOLDER, f'{number}_1_{havefcd}_flirt.nii.gz')
    flt.inputs.out_matrix_file = os.path.join(MAT_FOLDER, f'{number}_1_{havefcd}_flirt.mat')
    res = flt.run()

    flirt_img = nib.load(res.outputs.out_file)

    #mask image registration
    if have_mask:
        flt = fsl.FLIRT()
        flt.inputs.in_file = os.path.join(MASK_FOLDER, input_mask_name)
        flt.inputs.reference= template.get_filename()
        flt.inputs.apply_xfm = True
        flt.inputs.in_matrix_file = os.path.join(MAT_FOLDER, f'{number}_1_{havefcd}_flirt.mat')
        flt.inputs.out_file = os.path.join(FLIRT_MASK_FOLDER, f'{havefcd}_{number}_1_flirt.nii.gz')
        res = flt.run()
        flirt_mask = nib.load(res.outputs.out_file)
    else:
        flirt_mask = None

    #bias field correction
    fastr = fsl.FAST(img_type=1, no_pve=True) # 1 stays for T1
    fastr.inputs.in_files = os.path.join(FLIRT_MRI_FOLDER, f'{number}_1_{havefcd}_flirt.nii.gz')
    fastr.inputs.output_biascorrected = True
    out = fastr.run()
    flirt_bias_img = nib.load(os.path.join(FLIRT_MRI_FOLDER, f'{number}_1_{havefcd}_flirt_restore.nii.gz'))

    return flirt_img, flirt_bias_img, flirt_mask

#hcp and la5 images registration
def get_registered_img(input_img_name, template):
    #flirt image registration
    flt = fsl.FLIRT()
    flt.inputs.in_file = os.path.join(CONTROL_FOLDER, input_img_name)
    flt.inputs.reference= template.get_filename()
    flt.inputs.output_type = "NIFTI_GZ"
    flt.inputs.out_file = os.path.join(CONTROL_FLIRT_FOLDER, 'flirt_'+input_img_name)
    res = flt.run()

    #bias field correction
    fastr = fsl.FAST(img_type=1, no_pve=True) # 1 stays for T1
    fastr.inputs.in_files = os.path.join(CONTROL_FLIRT_FOLDER, 'flirt_'+input_img_name)
    fastr.inputs.output_biascorrected = True
    out = fastr.run()

    flirt_img = nib.load(res.outputs.out_file)
    flirt_bias_img = nib.load(os.path.join(CONTROL_FLIRT_FOLDER, 'flirt_' + input_img_name.split('.')[0] + '_restore.nii.gz'))
    return flirt_img, flirt_bias_img
