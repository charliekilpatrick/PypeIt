"""
Implements the flat-field class.

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst

"""
from pathlib import Path
from copy import deepcopy
import numpy as np

from IPython import embed

from pypeit import msgs
from pypeit import calibframe
from pypeit import edgetrace
from pypeit import io
from pypeit.images import buildimage
from pypeit.core import parse
from pypeit import dataPaths


class SlitlessFlat(calibframe.CalibFrame):
    """
    Class to generate a slitless pixel flat-field calibration image.

    Args:
        fitstbl (:class:`~pypeit.metadata.PypeItMetaData`):
                The class holding the metadata for all the frames.
        slitless_rows (`numpy.ndarray`_):
                Boolean array selecting the rows in the fitstbl that
                correspond to the slitless frames.
        spectrograph (:class:`~pypeit.spectrographs.spectrograph.Spectrograph`):
                The spectrograph object.
        par (:class:`~pypeit.par.pypeitpar.CalibrationsPar`):
            Parameter set defining optional parameters of PypeIt's algorithms
            for Calibrations
        qa_path (`Path`_):
            Path for the QA diagnostics.

    """
    version = '1.0.0'

    # Calibration frame attributes
    calib_type = 'SlitlessFlat'
    calib_file_format = 'fits'

    # Datamodel already includes PYP_SPEC, so no need to combine it with the
    # CalibFrame base datamodel.
    datamodel = {'PYP_SPEC': dict(otype=str, descr='PypeIt spectrograph name'),
                 'pixelflat_raw': dict(otype=np.ndarray, atype=np.floating,
                                       descr='Processed, combined pixel flats'),
                 'pixelflat_norm': dict(otype=np.ndarray, atype=np.floating,
                                        descr='Normalized pixel flat'),
                 'pixelflat_model': dict(otype=np.ndarray, atype=np.floating, descr='Model flat'),
    }
    def __init__(self, fitstbl, slitless_rows, spectrograph, par, qa_path=None):

        self.fitstbl = fitstbl
        # Boolean array selecting the rows in the fitstbl that correspond to the slitless frames.
        self.slitless_rows = slitless_rows
        self.spectrograph = spectrograph
        self.par = par
        self.qa_path = qa_path

    def slitless_pixflat_fname(self):
        """
        Generate the name of the slitless pixel flat file.

        Returns:
            :obj:`str`: The name of the slitless pixel flat

        """
        if len(self.slitless_rows) == 0:
            msgs.error('No slitless_pixflat frames found. Cannot generate the slitless pixel flat file name.')

        # generate the slitless pixel flat file name
        spec_name = self.fitstbl.spectrograph.name
        date = self.fitstbl.construct_obstime(self.slitless_rows[0]).iso.split(' ')[0].replace('-', '') if \
            self.fitstbl[self.slitless_rows][0]['mjd'] is not None else '00000000'
        # setup info to add to the filename
        dispname = '' if 'dispname' not in self.spectrograph.configuration_keys() else \
            f"_{self.fitstbl[self.slitless_rows[0]]['dispname'].replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '').replace(':', '_').replace('+', '_')}"
        dichroic = '' if 'dichroic' not in self.spectrograph.configuration_keys() else \
            f"_d{self.fitstbl[self.slitless_rows[0]]['dichroic']}"
        binning = self.fitstbl[self.slitless_rows[0]]['binning'].replace(',', 'x')
        # file name
        return f'pixelflat_{spec_name}{dispname}{dichroic}_{binning}_{date}.fits'

    def make_slitless_pixflat(self, msbias=None, msdark=None, calib_dir=None, write_qa=False, show=False):
        """
        Generate and save to disc a slitless pixel flat-field calibration images.
        The pixel flat file will have one extension per detector, even in the case of a mosaic.
        Contrary to the regular calibration flow, the slitless pixel flat is created for all detectors
        of the current spectrograph at once, and not only the one for the current detector.
        Since the slitless pixel flat images are saved to disc, this approach helps with the I/O
        This is a method is used in `~pypeit.calibrations.get_flats()`.

        Note: par['flatfield']['pixelflat_file'] is updated in this method.

        Args:
            msbias (:class:`~pypeit.images.buildimage.BiasImage`, optional):
                Bias image for bias subtraction; passed to
                :func:`~pypeit.images.buildimage.buildimage_fromlist()`
            msdark (:class:`~pypeit.images.buildimage.DarkImage`, optional):
                Dark-current image; passed to
                :func:`~pypeit.images.buildimage.buildimage_fromlist()`
            calib_dir (`Path`_):
                Path for the processed calibration files.
            write_qa (:obj:`bool`, optional):
                Write QA plots to disk?
            show (:obj:`bool`, optional):
                Show the diagnostic plots?

        Returns:
            :obj:`str`: The name of the slitless pixel flat file that was generated.

        """

        # First thing first, check if the user has provided slitless_pixflat frames
        if len(self.slitless_rows) == 0:
            # return unchanged self.par['flatfield']['pixelflat_file']
            return self.par['flatfield']['pixelflat_file']

        # all detectors of this spectrograph
        _detectors = np.array(self.spectrograph.select_detectors())

        # Check if a user-provided slitless pixelflat already exists for the current detectors
        if self.par['flatfield']['pixelflat_file'] is not None:
            _pixel_flat_file = dataPaths.pixelflat.get_file_path(self.par['flatfield']['pixelflat_file'],
                                                                 return_none=True)

            if _pixel_flat_file is not None:
                # get detector names
                detnames = np.array([self.spectrograph.get_det_name(_det) for _det in _detectors])
                # open the file
                with io.fits_open(_pixel_flat_file) as hdu:
                    # list of available detectors in the pixel flat file
                    file_detnames = [h.name.split('-')[0] for h in hdu]
                    # check if the current detnames are in the list
                    in_file = np.array([d in file_detnames for d in detnames])
                    # if all detectors are in the file, return
                    if np.all(in_file):
                        msgs.info(f"Both slitless_pixflat frames and user-defined file found. "
                                  f"The user-defined file will be used: {self.par['flatfield']['pixelflat_file']}")
                        # return unchanged self.par['flatfield']['pixelflat_file']
                        return self.par['flatfield']['pixelflat_file']
                    else:
                        # get the detectors that are not in the file
                        _detectors = _detectors[np.logical_not(in_file)]
                        detnames = detnames[np.logical_not(in_file)]
                        msgs.info(f'Both slitless_pixflat frames and user-defined file found, but the '
                                  f'following detectors are not in the file: {detnames}. Using the '
                                  f'slitless_pixflat frames to generate the missing detectors.')

        # make the slitless pixel flat
        pixflat_norm_list = []
        detname_list = []
        for _det in _detectors:
            # Parse the raw slitless pixelflat frames. Note that this is spectrograph dependent.
            # If the method does not exist in the specific spectrograph class, nothing will happen
            this_raw_idx = self.spectrograph.parse_raw_files(self.fitstbl[self.slitless_rows], det=_det,
                                                             ftype='slitless_pixflat')
            if len(this_raw_idx) == 0:
                msgs.warn(f'No raw slitless_pixflat frames found for {self.spectrograph.get_det_name(_det)}. '
                          f'Continuing...')
                continue
            this_raw_files = self.fitstbl.frame_paths(self.slitless_rows[this_raw_idx])
            msgs.info(f'Creating slitless pixel-flat calibration frame '
                      f'for {self.spectrograph.get_det_name(_det)} using files: ')
            for f in this_raw_files:
                msgs.prindent(f'{Path(f).name}')

            # Reset the BPM
            msbpm = self.spectrograph.bpm(this_raw_files[0], _det, msbias=msbias if self.par['bpm_usebias'] else None)

            # trace image
            traceimg = buildimage.buildimage_fromlist(self.spectrograph, _det, self.par['traceframe'],
                                                      [this_raw_files[0]], dark=msdark, bias=msbias, bpm=msbpm)
            # slit edges
            # we need to change some parameters for the slit edge tracing
            edges_par = deepcopy(self.par['slitedges'])
            # lower the threshold for edge detection
            edges_par['edge_thresh'] = 50.
            # this is used for longslit (i.e., no pca)
            edges_par['sync_predict'] = 'nearest'
            # remove spurious edges by setting a large minimum slit gap (20% of the detector size
            platescale = parse.parse_binning(traceimg.detector.binning)[1] * traceimg.detector['platescale']
            edges_par['minimum_slit_gap'] = 0.2 * traceimg.image.shape[1] * platescale
            # if no slits are found the bound_detector parameter add 2 traces at the detector edges
            edges_par['bound_detector'] = True
            # set the buffer to 0
            edges_par['det_buffer'] = 0
            _spectrograph = deepcopy(self.spectrograph)
            # need to treat this as a MultiSlit spectrograph (no echelle parameters used)
            _spectrograph.pypeline = 'MultiSlit'
            edges = edgetrace.EdgeTraceSet(traceimg, _spectrograph, edges_par, auto=True)
            slits = edges.get_slits()
            if show:
                edges.show(title='Slitless flat edge tracing')
            #
            # flat image
            slitless_pixel_flat = buildimage.buildimage_fromlist(self.spectrograph, _det, self.par['slitless_pixflatframe'],
                                                                 this_raw_files, dark=msdark, bias=msbias, bpm=msbpm)

            # increase saturation threshold (some hires slitless flats are very bright)
            slitless_pixel_flat.detector.saturation *= 1.5
            # Initialise the pixel flat
            flatpar = deepcopy(self.par['flatfield'])
            # do not tweak the slits
            flatpar['tweak_slits'] = False
            pixelFlatField = FlatField(slitless_pixel_flat, self.spectrograph, flatpar, slits, wavetilts=None,
                                       wv_calib=None, slitless=True, qa_path=self.qa_path)

            # Generate
            pixelflatImages = pixelFlatField.run(doqa=write_qa, show=show)
            pixflat_norm_list.append(pixelflatImages.pixelflat_norm)
            detname_list.append(self.spectrograph.get_det_name(_det))

        embed()
        exit()

        if len(detname_list) > 0:
            # get the pixel flat file name
            if self.par['flatfield']['pixelflat_file'] is not None and _pixel_flat_file is not None:
                fname = self.par['flatfield']['pixelflat_file']
            else:
                fname = self.slitless_pixflat_fname()
                # file will be saved in the reduction directory, but also cached in the data/pixelflats folder
                # therefore we update self.par['flatfield']['pixelflat_file'] to the new file,
                # so that it can be used for the rest of the reduction and for the other files in the same run
                self.par['flatfield']['pixelflat_file'] = fname

            # Save the result
            write_pixflat_to_fits(pixflat_norm_list, detname_list, self.spectrograph.name,
                                  calib_dir.parent if calib_dir is not None else Path('.').absolute(),
                                  fname, to_cache=True)

        return self.par['flatfield']['pixelflat_file']


def write_pixflat_to_fits(pixflat_norm_list, detname_list, spec_name, outdir, pixelflat_name, to_cache=True):
    """
    Write the pixel-to-pixel flat-field images to a FITS file.
    The FITS file will have an extension for each detector (never a mosaic).
    The `load_pixflat()` method read this file and transform it into a mosaic if needed.
    This image is generally used as a user-provided pixel flat-field image and ingested
    in the reduction using the `pixelflat_file` parameter in the PypeIt file.

    Args:
        pixflat_norm_list (:obj:`list`):
            List of 2D `numpy.ndarray`_ arrays containing the pixel-to-pixel flat-field images.
        detname_list (:obj:`list`):
            List of detector names.
        spec_name (:obj:`str`):
            Name of the spectrograph.
        outdir (:obj:`pathlib.Path`):
            Path to the output directory.
        pixelflat_name (:obj:`str`):
            Name of the output file to be written.
        to_cache (:obj:`bool`, optional):
            If True, the file will be written to the cache directory pypeit/data/pixflats.

    """

    msgs.info("Writing the pixel-to-pixel flat-field images to a FITS file.")

    # Check that the number of detectors matches the number of pixelflat_norm arrays
    if len(pixflat_norm_list) != len(detname_list):
        msgs.error("The number of detectors does not match the number of pixelflat_norm arrays. "
                   "The pixelflat file cannot be written.")

    # local output (reduction directory)
    pixelflat_file = outdir / pixelflat_name

    # Check if the file already exists
    old_hdus = []
    old_detnames = []
    old_hdr = None
    if pixelflat_file.exists():
        msgs.warn("The pixelflat file already exists. It will be overwritten/updated.")
        old_hdus = fits.open(pixelflat_file)
        old_detnames = [h.name.split('-')[0] for h in old_hdus]  # this has also 'PRIMARY'
        old_hdr = old_hdus[0].header

    # load spectrograph
    spec = load_spectrograph(spec_name)

    # Create the new HDUList
    _hdr = io.initialize_header(hdr=old_hdr)
    prihdu = fits.PrimaryHDU(header=_hdr)
    prihdu.header['CALIBTYP'] = (FlatImages.calib_type, 'PypeIt: Calibration frame type')
    new_hdus = [prihdu]

    extnum = 1
    for d in spec.select_detectors():
        detname = spec.get_det_name(d)
        extname = f'{detname}-PIXELFLAT_NORM'
        # update or add the detectors that we want to save
        if detname in detname_list:
            det_idx = detname_list.index(detname)
            pixflat_norm = pixflat_norm_list[det_idx]
            hdu = fits.ImageHDU(data=pixflat_norm, name=extname)
            prihdu.header[f'EXT{extnum:04d}'] = hdu.name
            new_hdus.append(hdu)
        # keep the old detectors that were not updated
        elif detname in old_detnames:
            old_det_idx = old_detnames.index(detname)
            hdu = old_hdus[old_det_idx]
            prihdu.header[f'EXT{extnum:04d}'] = hdu.name
            new_hdus.append(hdu)
        extnum += 1

    # Write the new HDUList
    new_hdulist = fits.HDUList(new_hdus)
    # Check if the directory exists
    if not pixelflat_file.parent.is_dir():
        pixelflat_file.parent.mkdir(parents=True)
    new_hdulist.writeto(pixelflat_file, overwrite=True)
    msgs.info(f'A slitless Pixel Flat file for detectors {detname_list} has been saved to {msgs.newline()}'
              f'{pixelflat_file}')

    # common msg
    add_msgs = f"add the following to your PypeIt Reduction File:{msgs.newline()}"  \
               f" [calibrations]{msgs.newline()}"  \
               f"   [[flatfield]]{msgs.newline()}"  \
               f"     pixelflat_file = {pixelflat_name}{msgs.newline()}{msgs.newline()}{msgs.newline()}"  \
               f"Please consider sharing your Pixel Flat file with the PypeIt Developers.{msgs.newline()}"  \


    if to_cache:
        # NOTE that the file saved in the cache is gzipped, while the one saved in the outdir is not
        # This prevents `dataPaths.pixelflat.get_file_path()` from returning the file saved in the outdir
        cache.write_file_to_cache(pixelflat_file, pixelflat_name+'.gz', f"pixelflats")
        msgs.info(f"The slitless Pixel Flat file has also been saved to the PypeIt cache directory {msgs.newline()}"
                  f"{str(dataPaths.pixelflat)} {msgs.newline()}"
                  f"It will be automatically used in this run. "
                  f"If you want to use this file in future runs, {add_msgs}")
    else:
        msgs.info(f"To use this file, move it to the PypeIt data directory {msgs.newline()}"
                  f"{str(dataPaths.pixelflat)} {msgs.newline()} and {add_msgs}")


def load_pixflat(pixel_flat_file, spectrograph, det, flatimages, calib_dir=None, chk_version=False):
    """
    Load a pixel flat from a file and add it to the flatimages object.
    The pixel flat file has one detector per extension, even in the case of a mosaic.
    Therefore, if this is a mosaic reduction, this script will construct a pixel flat
    mosaic. The Edges file needs to exist in the Calibration Folder, since the mosaic
    parameters are pulled from it.
    This is used in `~pypeit.calibrations.get_flats()`.

    Args:
        pixel_flat_file (:obj:`str`):
            Name of the pixel flat file.
        spectrograph (:class:`~pypeit.spectrographs.spectrograph.Spectrograph`):
            The spectrograph object.
        det (:obj:`int`, :obj:`tuple`):
            The single detector or set of detectors in a mosaic to process.
        flatimages (:class:`~pypeit.flatfield.FlatImages`):
            The flat field images object.
        calib_dir (:obj:`str`, optional):
            The path to the calibration directory.
        chk_version (:obj:`bool`, optional):
            Check the version of the file.

    Returns:
        :class:`~pypeit.flatfield.FlatImages`: The flat images object with the pixel flat added.

    """
    # Check if the pixel flat file exists
    if pixel_flat_file is None:
        msgs.error('No pixel flat file defined. Cannot load the pixel flat!')

    # get the path
    _pixel_flat_file = dataPaths.pixelflat.get_file_path(pixel_flat_file, return_none=True)
    if _pixel_flat_file is None:
        msgs.error(f'Cannot load the pixel flat file, {pixel_flat_file}. It is not a direct path, '
                   f'a cached file, or a file that can be downloaded from a PypeIt repository.')

    # If this is a mosaic, we need to construct the pixel flat mosaic
    if isinstance(det, tuple):
        # We need to grab mosaic info from another existing calibration frame.
        # We use EdgeTraceSet image to get `tform` and `msc_ord`. Check if EdgeTraceSet file exists.
        edges_file = Path(edgetrace.EdgeTraceSet.construct_file_name(flatimages.calib_key,
                                                                     calib_dir=calib_dir)).absolute()
        if not edges_file.exists():
            msgs.error('Edges file not found in the Calibrations folder. '
                       'It is needed to grab the mosaic parameters to load and mosaic the input pixel flat!')

        # Load detector info from EdgeTraceSet file
        traceimg = edgetrace.EdgeTraceSet.from_file(edges_file, chk_version=chk_version).traceimg
        det_info = traceimg.detector
        # check that the mosaic parameters are defined
        if not np.all(np.in1d(['tform', 'msc_ord'], list(det_info.keys()))) or  \
                det_info.tform is None or det_info.msc_ord is None:
            msgs.error('Mosaic parameters are not defined in the Edges frame. Cannot load the pixel flat!')

        # read the file
        with io.fits_open(_pixel_flat_file) as hdu:
            # list of available detectors in the pixel flat file
            file_dets = [int(h.name.split('-')[0].split('DET')[1]) for h in hdu[1:]]
            # check if all detectors required for the mosaic are in the list
            if not np.all(np.in1d(list(det), file_dets)):
                msgs.error(f'Not all detectors in the mosaic are in the pixel flat file: '
                           f'{pixel_flat_file}. Cannot load the pixel flat!')

            # get the pixel flat images of only the detectors in the mosaic
            pixflat_images = np.concatenate([hdu[f'DET{d:02d}-PIXELFLAT_NORM'].data[None,:,:] for d in det])
            # construct the pixel flat mosaic
            pixflat_msc, _,_,_ = build_image_mosaic(pixflat_images, det_info.tform, order=det_info.msc_ord)
            # check that the mosaic has the correct shape
            if pixflat_msc.shape != traceimg.image.shape:
                msgs.error('The constructed pixel flat mosaic does not have the correct shape. '
                           'Cannot load this pixel flat as a mosaic!')
            msgs.info(f'Using pixelflat file: {pixel_flat_file} '
                      f'for {spectrograph.get_det_name(det)}.')
            nrm_image = FlatImages(pixelflat_norm=pixflat_msc)

    # If this is not a mosaic, we can simply read the pixel flat for the current detector
    else:
        # current detector name
        detname = spectrograph.get_det_name(det)
        # read the file
        with io.fits_open(_pixel_flat_file) as hdu:
            # list of available detectors in the pixel flat file
            file_detnames = [h.name.split('-')[0] for h in hdu] # this list has also the 'PRIMARY' extension
            # check if the current detector is in the list
            if detname in file_detnames:
                # get the index of the current detector
                idx = file_detnames.index(detname)
                # get the pixel flat image
                msgs.info(f'Using pixelflat file: {pixel_flat_file} for {detname}.')
                nrm_image = FlatImages(pixelflat_norm=hdu[idx].data)
            else:
                msgs.error(f'{detname} not found in the pixel flat file: '
                           f'{pixel_flat_file}. Cannot load the pixel flat!')
                nrm_image = None

    return merge(flatimages, nrm_image)

