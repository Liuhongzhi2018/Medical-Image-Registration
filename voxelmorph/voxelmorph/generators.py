import os
import sys
import glob
import numpy as np

from . import py


def volgen(
    vol_names,
    batch_size=1,
    segs=None,
    np_var='vol',
    pad_shape=None,
    resize_factor=1,
    add_feat_axis=True
):
    """
    Base generator for random volume loading. Volumes can be passed as a path to
    the parent directory, a glob pattern, a list of file paths, or a list of
    preloaded volumes. Corresponding segmentations are additionally loaded if
    `segs` is provided as a list (of file paths or preloaded segmentations) or set
    to True. If `segs` is True, npz files with variable names 'vol' and 'seg' are
    expected. Passing in preloaded volumes (with optional preloaded segmentations)
    allows volumes preloaded in memory to be passed to a generator.

    Parameters:
        vol_names: Path, glob pattern, list of volume files to load, or list of
            preloaded volumes.
        batch_size: Batch size. Default is 1.
        segs: Loads corresponding segmentations. Default is None.
        np_var: Name of the volume variable if loading npz files. Default is 'vol'.
        pad_shape: Zero-pads loaded volumes to a given shape. Default is None.
        resize_factor: Volume resize factor. Default is 1.
        add_feat_axis: Load volume arrays with added feature axis. Default is True.
    """

    # convert glob path to filenames
    if isinstance(vol_names, str):
        if os.path.isdir(vol_names):
            vol_names = os.path.join(vol_names, '*')
        vol_names = glob.glob(vol_names)
    # print(f"*** volgen vol_names: {vol_names}")

    if isinstance(segs, list) and len(segs) != len(vol_names):
        raise ValueError('Number of image files must match number of seg files.')

    while True:
        # generate [batchsize] random image indices
        indices = np.random.randint(len(vol_names), size=batch_size)

        # load volumes and concatenate
        load_params = dict(np_var=np_var, add_batch_axis=True, add_feat_axis=add_feat_axis,
                           pad_shape=pad_shape, resize_factor=resize_factor)
        imgs = [py.utils.load_volfile(vol_names[i], **load_params) for i in indices]
        vols = [np.concatenate(imgs, axis=0)]

        # optionally load segmentations and concatenate
        if segs is True:
            # assume inputs are npz files with 'seg' key
            load_params['np_var'] = 'seg'  # be sure to load seg
            s = [py.utils.load_volfile(vol_names[i], **load_params) for i in indices]
            vols.append(np.concatenate(s, axis=0))
        elif isinstance(segs, list):
            # assume segs is a corresponding list of files or preloaded volumes
            s = [py.utils.load_volfile(segs[i], **load_params) for i in indices]
            vols.append(np.concatenate(s, axis=0))

        yield tuple(vols)
        
        
def volgen_pairs(
    vol_names,
    batch_size=1,
    segs=None,
    np_var='vol',
    pad_shape=None,
    resize_factor=1,
    add_feat_axis=True
):
    """
    Base generator for random volume loading. Volumes can be passed as a path to
    the parent directory, a glob pattern, a list of file paths, or a list of
    preloaded volumes. Corresponding segmentations are additionally loaded if
    `segs` is provided as a list (of file paths or preloaded segmentations) or set
    to True. If `segs` is True, npz files with variable names 'vol' and 'seg' are
    expected. Passing in preloaded volumes (with optional preloaded segmentations)
    allows volumes preloaded in memory to be passed to a generator.

    Parameters:
        vol_names: Path, glob pattern, list of volume files to load, or list of
            preloaded volumes.
        batch_size: Batch size. Default is 1.
        segs: Loads corresponding segmentations. Default is None.
        np_var: Name of the volume variable if loading npz files. Default is 'vol'.
        pad_shape: Zero-pads loaded volumes to a given shape. Default is None.
        resize_factor: Volume resize factor. Default is 1.
        add_feat_axis: Load volume arrays with added feature axis. Default is True.
    """

    # convert glob path to filenames
    if isinstance(vol_names, str):
        if os.path.isdir(vol_names):
            vol_names = os.path.join(vol_names, '*')
        vol_names = glob.glob(vol_names)
    print(f"*** volgen vol_names: {vol_names}")

    if isinstance(segs, list) and len(segs) != len(vol_names):
        raise ValueError('Number of image files must match number of seg files.')

    while True:
        # generate [batchsize] random image indices
        indices = np.random.randint(len(vol_names), size=batch_size)

        # load volumes and concatenate
        load_params = dict(np_var=np_var, add_batch_axis=True, add_feat_axis=add_feat_axis,
                           pad_shape=pad_shape, resize_factor=resize_factor)
        # print(f"vol_names[i][0]: {vol_names[0][0]} vol_names[i][1]: {vol_names[0][1]}")
        # vol_names[i][0]: /home/liuhongzhi/Data/ACDC/training/patient075/patient075_frame01.nii.gz 
        # vol_names[i][1]: /home/liuhongzhi/Data/ACDC/training/patient075/patient075_frame06.nii.gz
        imgs1 = [py.utils.load_volfile(vol_names[i][0], **load_params) for i in indices]
        vols1 = [np.concatenate(imgs1, axis=0)]
        imgs2 = [py.utils.load_volfile(vol_names[i][1], **load_params) for i in indices]
        vols2 = [np.concatenate(imgs2, axis=0)]

        # optionally load segmentations and concatenate
        if segs is True:
            # print("volgen_pairs segs")
            # assume inputs are npz files with 'seg' key
            load_params['np_var'] = 'seg'  # be sure to load seg
            s = [py.utils.load_volfile(vol_names[i], **load_params) for i in indices]
            vols.append(np.concatenate(s, axis=0))
        elif isinstance(segs, list):
            # print("volgen_pairs segs list")
            # assume segs is a corresponding list of files or preloaded volumes
            s = [py.utils.load_volfile(segs[i], **load_params) for i in indices]
            vols.append(np.concatenate(s, axis=0))

        yield (tuple(vols1), tuple(vols2))


def volgen_pairs_with_seg(
    vol_names,
    seg_names, 
    batch_size=1,
    np_var='vol',
    segs=None,
    pad_shape=None,
    resize_factor=1,
    add_feat_axis=True
):
    """
    Base generator for random volume loading. Volumes can be passed as a path to
    the parent directory, a glob pattern, a list of file paths, or a list of
    preloaded volumes. Corresponding segmentations are additionally loaded if
    `segs` is provided as a list (of file paths or preloaded segmentations) or set
    to True. If `segs` is True, npz files with variable names 'vol' and 'seg' are
    expected. Passing in preloaded volumes (with optional preloaded segmentations)
    allows volumes preloaded in memory to be passed to a generator.

    Parameters:
        vol_names: Path, glob pattern, list of volume files to load, or list of
            preloaded volumes.
        batch_size: Batch size. Default is 1.
        segs: Loads corresponding segmentations. Default is None.
        np_var: Name of the volume variable if loading npz files. Default is 'vol'.
        pad_shape: Zero-pads loaded volumes to a given shape. Default is None.
        resize_factor: Volume resize factor. Default is 1.
        add_feat_axis: Load volume arrays with added feature axis. Default is True.
    """

    # convert glob path to filenames
    if isinstance(vol_names, str):
        if os.path.isdir(vol_names):
            vol_names = os.path.join(vol_names, '*')
        vol_names = glob.glob(vol_names)
    # print(f"volgen_pairs_with_seg vol_names: {vol_names}")
    # print(f"volgen_pairs_with_seg seg_names: {seg_names}")

    if isinstance(segs, list) and len(segs) != len(vol_names):
        raise ValueError('Number of image files must match number of seg files.')

    while True:
        # generate [batchsize] random image indices
        indices = np.random.randint(len(vol_names), size=batch_size)

        # load volumes and concatenate
        load_params = dict(np_var=np_var, add_batch_axis=True, add_feat_axis=add_feat_axis,
                           pad_shape=pad_shape, resize_factor=resize_factor)
        # print(f"vol_names[i][0]: {vol_names[0][0]} vol_names[i][1]: {vol_names[0][1]}")
        # vol_names[i][0]: /home/liuhongzhi/Data/ACDC/training/patient075/patient075_frame01.nii.gz 
        # vol_names[i][1]: /home/liuhongzhi/Data/ACDC/training/patient075/patient075_frame06.nii.gz
        imgs1 = [py.utils.load_volfile(vol_names[i][0], **load_params) for i in indices]
        vols1 = [np.concatenate(imgs1, axis=0)]
        imgs2 = [py.utils.load_volfile(vol_names[i][1], **load_params) for i in indices]
        vols2 = [np.concatenate(imgs2, axis=0)]
        imgs3 = [py.utils.load_volfile(seg_names[i][0], **load_params) for i in indices]
        segs1 = [np.concatenate(imgs3, axis=0)]
        imgs4 = [py.utils.load_volfile(seg_names[i][1], **load_params) for i in indices]
        segs2 = [np.concatenate(imgs4, axis=0)]

        yield (tuple(vols1), tuple(vols2), tuple(segs1), tuple(segs2))


def scan_to_scan(vol_names, bidir=False, batch_size=1, prob_same=0, no_warp=False, **kwargs):
    """
    Generator for scan-to-scan registration.

    Parameters:
        vol_names: List of volume files to load, or list of preloaded volumes.
        bidir: Yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        prob_same: Induced probability that source and target inputs are the same. Default is 0.
        no_warp: Excludes null warp in output list if set to True (for affine training). 
            Default if False.
        kwargs: Forwarded to the internal volgen generator.
    """
    zeros = None
    # print(f"scan_to_scan vol_names: {vol_names}")
    gen = volgen(vol_names, batch_size=batch_size, **kwargs)
    while True:
        scan1 = next(gen)[0]
        scan2 = next(gen)[0]

        # some induced chance of making source and target equal
        if prob_same > 0 and np.random.rand() < prob_same:
            if np.random.rand() > 0.5:
                scan1 = scan2
            else:
                scan2 = scan1

        # cache zeros
        if not no_warp and zeros is None:
            shape = scan1.shape[1:-1]
            zeros = np.zeros((batch_size, *shape, len(shape)))

        invols = [scan1, scan2]
        outvols = [scan2, scan1] if bidir else [scan2]
        if not no_warp:
            outvols.append(zeros)

        yield (invols, outvols)
        
        
def scan_to_scan_pairs(vol_names, bidir=False, batch_size=1, prob_same=0, no_warp=False, **kwargs):
    """
    Generator for scan-to-scan registration.

    Parameters:
        vol_names: List of volume files to load, or list of preloaded volumes.
        bidir: Yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        prob_same: Induced probability that source and target inputs are the same. Default is 0.
        no_warp: Excludes null warp in output list if set to True (for affine training). 
            Default if False.
        kwargs: Forwarded to the internal volgen generator.
    """
    zeros = None
    # print(f"scan_to_scan vol_names: {vol_names}")
    # scan_to_scan vol_names: [['/home/liuhongzhi/Data/ACDC/training/patient075/patient075_frame01.nii.gz', 
    # '/home/liuhongzhi/Data/ACDC/training/patient075/patient075_frame06.nii.gz'], 
    # ['/home/liuhongzhi/Data/ACDC/training/patient059/patient059_frame01.nii.gz', 
    # '/home/liuhongzhi/Data/ACDC/training/patient059/patient059_frame09.nii.gz'],]
    gen = volgen_pairs(vol_names, batch_size=batch_size, **kwargs)
    while True:
        # scan1 = next(gen)[0]
        # scan2 = next(gen)[0]
        scans = next(gen)
        scan1, scan2 = scans[0][0], scans[1][0]
        # print(f"* scan_to_scan_pairs scan1: {scan1.shape}")
        # print(f"* scan_to_scan_pairs scan2: {scan2.shape}")

        # # some induced chance of making source and target equal
        # if prob_same > 0 and np.random.rand() < prob_same:
        #     if np.random.rand() > 0.5:
        #         scan1 = scan2
        #     else:
        #         scan2 = scan1

        # cache zeros
        # if not no_warp and zeros is None:
        #     shape = scan1.shape[1:-1]
        #     zeros = np.zeros((batch_size, *shape, len(shape)))

        invols = [scan1, scan2]
        # outvols = [scan2, scan1] if bidir else [scan2]
        outvols = [scan2, scan1]
        # if not no_warp:
        #     outvols.append(zeros)

        # print(f"** scan_to_scan_pairs invols: {invols.shape}")
        # print(f"** scan_to_scan_pairs outvols: {outvols.shape}")
        # print(f"** scan_to_scan_pairs invols0: {invols[0].shape} invols1: {invols[1].shape}")
        # print(f"** scan_to_scan_pairs outvols0: {outvols[0].shape} outvols1: {outvols[1].shape}")
        yield (invols, outvols)


def scan_to_atlas(vol_names, atlas, bidir=False, batch_size=1, no_warp=False, segs=None, **kwargs):
    """
    Generator for scan-to-atlas registration.

    TODO: This could be merged into scan_to_scan() by adding an optional atlas
    argument like in semisupervised().

    Parameters:
        vol_names: List of volume files to load, or list of preloaded volumes.
        atlas: Atlas volume data.
        bidir: Yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        no_warp: Excludes null warp in output list if set to True (for affine training). 
            Default is False.
        segs: Load segmentations as output, for supervised training. Forwarded to the
            internal volgen generator. Default is None.
        kwargs: Forwarded to the internal volgen generator.
    """
    shape = atlas.shape[1:-1]
    zeros = np.zeros((batch_size, *shape, len(shape)))
    atlas = np.repeat(atlas, batch_size, axis=0)
    # /mnt/lhz/Github/Image_registration/voxelmorph/voxelmorph/generators.py 
    # def volgen
    gen = volgen(vol_names, batch_size=batch_size, segs=segs, **kwargs)
    # print(f"scan_to_atlas atlas shape: {atlas.shape} shape: {shape} zeros: {zeros.shape} atlas: {atlas.shape}")
    # scan_to_atlas atlas shape: (1, 160, 192, 160, 1) shape: (160, 192, 160) 
    # zeros: (1, 160, 192, 160, 3) atlas: (1, 160, 192, 160, 1)
    while True:
        res = next(gen)
        scan = res[0]
        invols = [scan, atlas]
        # print(f"scan_to_atlas scan shape: {scan.shape} atlas shape: {atlas.shape}")
        # scan_to_atlas scan shape: (1, 160, 192, 160, 1) atlas shape: (1, 160, 192, 160, 1)
        if not segs:
            # print(f"scan_to_atlas not segs")
            # scan_to_atlas not segs
            outvols = [atlas, scan] if bidir else [atlas]
        else:
            # print(f"scan_to_atlas have segs")
            seg = res[1]
            outvols = [seg, scan] if bidir else [seg]
        if not no_warp:
            # print(f"scan_to_atlas not no_warp")
            # scan_to_atlas not no_warp
            outvols.append(zeros)
        yield (invols, outvols)


def semisupervised(vol_names, seg_names, labels, atlas_file=None, downsize=2):
    """
    Generator for semi-supervised registration training using ground truth segmentations.
    Scan-to-atlas training can be enabled by providing the atlas_file argument. 

    Parameters:
        vol_names: List of volume files to load, or list of preloaded volumes.
        seg_names: List of corresponding seg files to load, or list of preloaded volumes.
        labels: Array of discrete label values to use in training.
        atlas_file: Atlas npz file for scan-to-atlas training. Default is None.
        downsize: Downsize factor for segmentations. Default is 2.
    """
    # configure base generator
    gen = volgen(vol_names, segs=seg_names, np_var='vol')
    zeros = None

    # internal utility to generate downsampled prob seg from discrete seg
    def split_seg(seg):
        prob_seg = np.zeros((*seg.shape[:4], len(labels)))
        for i, label in enumerate(labels):
            prob_seg[0, ..., i] = seg[0, ..., 0] == label
        return prob_seg[:, ::downsize, ::downsize, ::downsize, :]

    # cache target vols and segs if atlas is supplied
    if atlas_file:
        trg_vol = py.utils.load_volfile(atlas_file, np_var='vol',
                                        add_batch_axis=True, add_feat_axis=True)
        trg_seg = py.utils.load_volfile(atlas_file, np_var='seg',
                                        add_batch_axis=True, add_feat_axis=True)
        trg_seg = split_seg(trg_seg)

    while True:
        # load source vol and seg
        src_vol, src_seg = next(gen)
        src_seg = split_seg(src_seg)

        # load target vol and seg (if not provided by atlas)
        if not atlas_file:
            trg_vol, trg_seg = next(gen)
            trg_seg = split_seg(trg_seg)

        # cache zeros
        if zeros is None:
            shape = src_vol.shape[1:-1]
            zeros = np.zeros((1, *shape, len(shape)))

        invols = [src_vol, trg_vol, src_seg]
        outvols = [trg_vol, zeros, trg_seg]
        yield (invols, outvols)


def semisupervised_pairs(vol_names, seg_names, batch_size=1, atlas_file=None, downsize=2):
    """
    Generator for semi-supervised registration training using ground truth segmentations.
    Scan-to-atlas training can be enabled by providing the atlas_file argument. 

    Parameters:
        vol_names: List of volume files to load, or list of preloaded volumes.
        seg_names: List of corresponding seg files to load, or list of preloaded volumes.
        labels: Array of discrete label values to use in training.
        atlas_file: Atlas npz file for scan-to-atlas training. Default is None.
        downsize: Downsize factor for segmentations. Default is 2.
    """
    # configure base generator
    # gen = volgen(vol_names, segs=seg_names, np_var='vol')
    gen = volgen_pairs_with_seg(vol_names, seg_names, batch_size=batch_size)
    zeros = None

    # internal utility to generate downsampled prob seg from discrete seg
    def split_seg(seg, labels):
        prob_seg = np.zeros((*seg.shape[:4], len(labels)))
        for i, label in enumerate(labels):
            prob_seg[0, ..., i] = seg[0, ..., 0] == label
        # return prob_seg[:, ::downsize, ::downsize, ::downsize, :]
        return prob_seg

    # cache target vols and segs if atlas is supplied
    if atlas_file:
        trg_vol = py.utils.load_volfile(atlas_file, np_var='vol',
                                        add_batch_axis=True, add_feat_axis=True)
        trg_seg = py.utils.load_volfile(atlas_file, np_var='seg',
                                        add_batch_axis=True, add_feat_axis=True)
        trg_seg = split_seg(trg_seg)

    while True:
        # load source vol and seg
        # src_vol, trg_vol = next(gen)
        src_vol, trg_vol, src_seg, trg_seg = next(gen)
        src_vol = src_vol[0]
        trg_vol = trg_vol[0]
        # print(f"semisupervised_pairs src_vol: {src_vol.shape} trg_vol: {trg_vol.shape} ")
        # semisupervised_pairs src_vol: (1, 160, 192, 160, 1) trg_vol: (1, 160, 192, 160, 1) 

        # load target vol and seg (if not provided by atlas)
        if not atlas_file:
            # src_seg, trg_seg = next(gen_seg)
            # print(f"semisupervised_pairs src_seg: {src_seg[0].shape} trg_seg: {trg_seg[0].shape} ")
            # semisupervised_pairs src_seg: (1, 216, 256, 8, 1) trg_seg: (1, 216, 256, 8, 1) 
            labels = np.unique(trg_seg[0])
            # print(f"semisupervised_pairs labels: {labels}")
            # semisupervised_pairs labels: [0 1 2 3]
            src_seg = split_seg(src_seg[0], labels)
            trg_seg = split_seg(trg_seg[0], labels)
            # print(f"semisupervised_pairs src_seg: {src_seg.shape}")
            # print(f"semisupervised_pairs trg_seg: {trg_seg.shape}")
            # semisupervised_pairs src_seg: (1, 80, 96, 80, 57)
            # semisupervised_pairs trg_seg: (1, 80, 96, 80, 57)
            # semisupervised_pairs src_seg: (1, 160, 192, 160, 57)
            # semisupervised_pairs trg_seg: (1, 160, 192, 160, 57)

        # cache zeros
        # print(f"semisupervised_pairs zeros: {zero.shape}") if zeros else print(f"semisupervised_pairs zero None")
        # semisupervised_pairs zero None
        if zeros is None:
            shape = src_vol.shape[1:-1]
            zeros = np.zeros((1, *shape, len(shape)))

        invols = [src_vol, trg_vol, src_seg]
        outvols = [trg_vol, zeros, trg_seg]
        yield (invols, outvols)


def template_creation(vol_names, bidir=False, batch_size=1, **kwargs):
    """
    Generator for unconditional template creation.

    Parameters:
        vol_names: List of volume files to load, or list of preloaded volumes.
        bidir: Yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        kwargs: Forwarded to the internal volgen generator.
    """
    zeros = None
    gen = volgen(vol_names, batch_size=batch_size, **kwargs)
    while True:
        scan = next(gen)[0]

        # cache zeros
        if zeros is None:
            shape = scan.shape[1:-1]
            zeros = np.zeros((1, *shape, len(shape)))

        invols = [scan]
        outvols = [scan, zeros, zeros, zeros] if bidir else [scan, zeros, zeros]
        yield (invols, outvols)


def conditional_template_creation(vol_names, atlas, attributes,
                                  batch_size=1, np_var='vol', pad_shape=None, add_feat_axis=True):
    """
    Generator for conditional template creation.

    Parameters:
        vol_names: List of volume files to load, or list of preloaded volumes.
        atlas: Atlas input volume data.
        attributes: Dictionary of phenotype data for each vol name.
        batch_size: Batch size. Default is 1.
        np_var: Name of the volume variable if loading npz files. Default is 'vol'.
        pad_shape: Zero-pads loaded volumes to a given shape. Default is None.
        add_feat_axis: Load volume arrays with added feature axis. Default is True.
    """
    shape = atlas.shape[1:-1]
    zeros = np.zeros((batch_size, *shape, len(shape)))
    atlas = np.repeat(atlas, batch_size, axis=0)
    while True:
        indices = np.random.randint(len(vol_names), size=batch_size)

        # load pheno from attributes dictionary
        pheno = np.stack([attributes[vol_names[i]] for i in indices], axis=0)

        # load volumes and concatenate
        load_params = dict(np_var=np_var, add_batch_axis=True,
                           add_feat_axis=add_feat_axis, pad_shape=pad_shape)
        vols = [py.utils.load_volfile(vol_names[i], **load_params) for i in indices]
        vols = np.concatenate(vols, axis=0)

        invols = [pheno, atlas, vols]
        outvols = [vols, zeros, zeros, zeros]
        yield (invols, outvols)


def surf_semisupervised(
    vol_names,
    atlas_vol,
    atlas_seg,
    nb_surface_pts,
    labels=None,
    batch_size=1,
    surf_bidir=True,
    surface_pts_upsample_factor=2,
    smooth_seg_std=1,
    nb_labels_sample=None,
    sdt_vol_resize=1,
    align_segs=False,
    add_feat_axis=True
):
    """
    Scan-to-atlas generator for semi-supervised learning using surface point clouds 
    from segmentations.

    Parameters:
        vol_names: List of volume files to load.
        atlas_vol: Atlas volume array.
        atlas_seg: Atlas segmentation array.
        nb_surface_pts: Total number surface points for all structures.
        labels: Label list to include. If None, all labels in atlas_seg are used. Default is None.
        batch_size: Batch size. NOTE some features only implemented for 1. Default is 1.
        surf_bidir: Train with bidirectional surface distance. Default is True.
        surface_pts_upsample_factor: Upsample factor for surface pointcloud. Default is 2.
        smooth_seg_std: Segmentation smoothness sigma. Default is 1.
        nb_labels_sample: Number of labels to sample. Default is None.
        sdt_vol_resize: Resize factor for signed distance transform volumes. Default is 1.
        align_segs: Whether to pass in segmentation image instead. Default is False.
        add_feat_axis: Load volume arrays with added feature axis. Default is True.
    """

    # some input checks
    assert nb_surface_pts > 0, 'number of surface point should be greater than 0'

    # prepare some shapes
    vol_shape = atlas_seg.shape
    sdt_shape = [int(f * sdt_vol_resize) for f in vol_shape]

    # compute labels from atlas, and the number of labels to sample.
    if labels is not None:
        atlas_seg = py.utils.filter_labels(atlas_seg, labels)
    else:
        labels = np.sort(np.unique(atlas_seg))[1:]

    # use all labels by default
    if nb_labels_sample is None:
        nb_labels_sample = len(labels)

    # prepare keras format atlases
    atlas_vol_bs = np.repeat(atlas_vol[np.newaxis, ..., np.newaxis], batch_size, axis=0)
    atlas_seg_bs = np.repeat(atlas_seg[np.newaxis, ..., np.newaxis], batch_size, axis=0)

    # prepare surface extraction function
    std_to_surf = lambda x, y: py.utils.sdt_to_surface_pts(
        x, y,
        surface_pts_upsample_factor=surface_pts_upsample_factor,
        thr=(1 / surface_pts_upsample_factor + 1e-5))

    # prepare zeros, which will be used for outputs unused in cost functions
    zero_flow = np.zeros((batch_size, *vol_shape, len(vol_shape)))
    zero_surface_values = np.zeros((batch_size, nb_surface_pts, 1))

    # precompute label edge volumes
    atlas_sdt = [None] * len(labels)
    atlas_label_vols = [None] * len(labels)
    nb_edges = np.zeros(len(labels))
    for li, label in enumerate(labels):  # if only one label, get surface points here
        atlas_label_vols[li] = atlas_seg == label
        atlas_label_vols[li] = py.utils.clean_seg(atlas_label_vols[li], smooth_seg_std)
        atlas_sdt[li] = py.utils.vol_to_sdt(
            atlas_label_vols[li], sdt=True, sdt_vol_resize=sdt_vol_resize)
        nb_edges[li] = np.sum(np.abs(atlas_sdt[li]) < 1.01)
    layer_edge_ratios = nb_edges / np.sum(nb_edges)

    # if working with all the labels passed in (i.e. no label sampling per batch),
    # pre-compute the atlas surface points
    atlas_surface_pts = np.zeros((batch_size, nb_surface_pts, len(vol_shape) + 1))
    if nb_labels_sample == len(labels):
        nb_surface_pts_sel = py.utils.get_surface_pts_per_label(nb_surface_pts, layer_edge_ratios)
        for li, label in enumerate(labels):  # if only one label, get surface points here
            atlas_surface_pts_ = std_to_surf(atlas_sdt[li], nb_surface_pts_sel[li])[np.newaxis, ...]
            # get the surface point stack indexes for this element
            srf_idx = slice(int(np.sum(nb_surface_pts_sel[:li])), int(
                np.sum(nb_surface_pts_sel[:li + 1])))
            atlas_surface_pts[:, srf_idx, :-1] = np.repeat(atlas_surface_pts_, batch_size, 0)
            atlas_surface_pts[:, srf_idx, -1] = li

    # generator
    gen = volgen(vol_names, segs=True, batch_size=batch_size, add_feat_axis=add_feat_axis)

    assert batch_size == 1, 'only batch size 1 supported for now'

    while True:

        # prepare data
        X = next(gen)
        X_img = X[0]
        X_seg = py.utils.filter_labels(X[1], labels)

        # get random labels
        sel_label_idxs = range(len(labels))  # all labels
        if nb_labels_sample != len(labels):
            sel_label_idxs = np.sort(np.random.choice(
                range(len(labels)), size=nb_labels_sample, replace=False))
            sel_layer_edge_ratios = [layer_edge_ratios[li] for li in sel_label_idxs]
            nb_surface_pts_sel = py.utils.get_surface_pts_per_label(
                nb_surface_pts, sel_layer_edge_ratios)

        # prepare signed distance transforms and surface point arrays
        X_sdt_k = np.zeros((batch_size, *sdt_shape, nb_labels_sample))
        atl_dt_k = np.zeros((batch_size, *sdt_shape, nb_labels_sample))
        subj_surface_pts = np.zeros((batch_size, nb_surface_pts, len(vol_shape) + 1))
        if nb_labels_sample != len(labels):
            atlas_surface_pts = np.zeros((batch_size, nb_surface_pts, len(vol_shape) + 1))

        for li, sli in enumerate(sel_label_idxs):
            # get the surface point stack indexes for this element
            srf_idx = slice(int(np.sum(nb_surface_pts_sel[:li])), int(
                np.sum(nb_surface_pts_sel[:li + 1])))

            # get atlas surface points for this label
            if nb_labels_sample != len(labels):
                atlas_surface_pts_ = std_to_surf(atlas_sdt[sli], nb_surface_pts_sel[li])[
                    np.newaxis, ...]
                atlas_surface_pts[:, srf_idx, :-1] = np.repeat(atlas_surface_pts_, batch_size, 0)
                atlas_surface_pts[:, srf_idx, -1] = sli

            # compute X distance from surface
            X_label = X_seg == labels[sli]
            X_label = py.utils.clean_seg_batch(X_label, smooth_seg_std)
            X_sdt_k[..., li] = py.utils.vol_to_sdt_batch(
                X_label, sdt=True, sdt_vol_resize=sdt_vol_resize)[..., 0]

            if surf_bidir:
                atl_dt = atlas_sdt[li][np.newaxis, ...]
                atl_dt_k[..., li] = np.repeat(atl_dt, batch_size, 0)
                ssp_lst = [std_to_surf(f[...], nb_surface_pts_sel[li]) for f in X_sdt_k[..., li]]
                subj_surface_pts[:, srf_idx, :-1] = np.stack(ssp_lst, 0)
                subj_surface_pts[:, srf_idx, -1] = li

        # check if returning segmentations instead of images
        # this is a bit hacky for basically building a segmentation-only network (no images)
        X_ret = X_img
        atlas_ret = atlas_vol_bs

        if align_segs:
            assert len(labels) == 1, 'align_seg generator is only implemented for single label'
            X_ret = X_seg == labels[0]
            atlas_ret = atlas_seg_bs == labels[0]

        # finally, output
        if surf_bidir:
            inputs = [X_ret, atlas_ret, X_sdt_k, atl_dt_k, subj_surface_pts, atlas_surface_pts]
            outputs = [atlas_ret, X_ret, zero_flow, zero_surface_values, zero_surface_values]
        else:
            inputs = [X_ret, atlas_ret, X_sdt_k, atlas_surface_pts]
            outputs = [atlas_ret, X_ret, zero_flow, zero_surface_values]

        yield (inputs, outputs)


def synthmorph(label_maps, batch_size=1, same_subj=False, flip=True):
    """
    Generator for SynthMorph registration.

    Parameters:
        labels_maps: List of pre-loaded ND label maps, each as a NumPy array.
        batch_size: Batch size. Default is 1.
        same_subj: Whether the same label map is returned as the source and target for further
            augmentation. Default is False.
        flip: Whether axes are flipped randomly. Default is True.
    """
    in_shape = label_maps[0].shape
    num_dim = len(in_shape)

    # "True" moved image and warp, that will be ignored by SynthMorph losses.
    void = np.zeros((batch_size, *in_shape, num_dim), dtype='float32')

    rand = np.random.default_rng()
    prop = dict(replace=False, shuffle=False)
    while True:
        ind = rand.integers(len(label_maps), size=2 * batch_size)
        x = [label_maps[i] for i in ind]

        if same_subj:
            x = x[:batch_size] * 2
        x = np.stack(x)[..., None]

        if flip:
            axes = rand.choice(num_dim, size=rand.integers(num_dim + 1), **prop)
            x = np.flip(x, axis=axes + 1)

        src = x[:batch_size, ...]
        trg = x[batch_size:, ...]
        yield [src, trg], [void] * 2
