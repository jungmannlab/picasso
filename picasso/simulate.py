"""
    picasso.simulate
    ~~~~~~~~~~~~~~~~

    Simulate single molcule fluorescence data

    :author: Maximilian Thomas Strauss, 2016-2018
    :copyright: Copyright (c) 2016-2018 Jungmann Lab, MPI of Biochemistry
"""
import numpy as _np
from . import io as _io
from numba import njit

magfac = 0.79


@njit
def calculate_zpsf(
    z: _np.ndarray | float, 
    cx: _np.ndarray, 
    cy: _np.ndarray,
) -> tuple[_np.ndarray, _np.ndarray] | tuple[float, float]:
    """Calculates the astigmatic PSF size at a given z position.
    
    Parameters
    ----------
    z : _np.ndarray or number
        The z position(s) at which to calculate the PSF.
    cx, cy : _np.ndarray
        Coefficients for the x/y dimension of the PSF.
    
    Returns
    -------
    wx, wy : _np.ndarray
        The calculated PSF sizes in the x and y dimensions.
    """

    z = z / magfac
    z2 = z * z
    z3 = z * z2
    z4 = z * z3
    z5 = z * z4
    z6 = z * z5
    wx = (
        cx[0] * z6
        + cx[1] * z5
        + cx[2] * z4
        + cx[3] * z3
        + cx[4] * z2
        + cx[5] * z
        + cx[6]
    )
    wy = (
        cy[0] * z6
        + cy[1] * z5
        + cy[2] * z4
        + cy[3] * z3
        + cy[4] * z2
        + cy[5] * z
        + cy[6]
    )
    return (wx, wy)


def test_calculate_zpsf() -> _np.ndarray:
    """Test function for calculate_zpsf."""

    cx = _np.array([1, 2, 3, 4, 5, 6, 7])
    cy = _np.array([1, 2, 3, 4, 5, 6, 7])
    z = _np.array([1, 2, 3, 4, 5, 6, 7])
    wx, wy = calculate_zpsf(z, cx, cy)

    result = [
        4.90350522e01,
        7.13644987e02,
        5.52316597e03,
        2.61621620e04,
        9.06621337e04,
        2.54548124e05,
        6.14947219e05,
    ]

    delta = wx - result
    assert sum(delta**2) < 0.001


def saveInfo(filename: str, info: dict) -> None:
    """Saves metadata to a YAML file."""

    _io.save_info(filename, [info], default_flow_style=True)


def noisy(image: _np.ndarray, mu: float, sigma: float) -> _np.ndarray:
    """Adds gaussian noise to an image.

    Parameters
    ----------
    image : _np.ndarray
        The input image to which noise will be added.
    mu : float
        The mean of the Gaussian noise.
    sigma : float
        The standard deviation of the Gaussian noise.

    Returns
    -------
    noisy : _np.ndarray
        The noisy image with Gaussian noise added.
    """

    row, col = image.shape  # Variance for _np.random is 1
    gauss = sigma * _np.random.normal(0, 1, (row, col)) + mu
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    noisy[noisy < 0] = 0
    return noisy


def noisy_p(image: _np.ndarray, mu: float) -> _np.ndarray:
    """Adds Poisson noise to an image or movie.
    
    Parameters
    ----------
    image : _np.ndarray
        The input image to which Poisson noise will be added.
    mu : float
        The mean of the Poisson noise.  

    Returns
    -------
    noisy : _np.ndarray
        The noisy image with Poisson noise added.
    """
    
    poiss = _np.random.poisson(mu, image.shape).astype(float)
    noisy = image + poiss
    return noisy


def check_type(movie: _np.ndarray) -> _np.ndarray:
    """Checks the type of the movie and converts it to a 16-bit unsigned 
    integer if necessary.

    Parameters
    ----------
    movie : _np.ndarray
        The input movie to be checked and converted.    

    Returns
    -------
    movie : _np.ndarray
        The movie converted to a 16-bit unsigned integer type.
    """

    movie[movie >= (2**16) - 1] = (2**16) - 1
    movie = movie.astype("<u2")  # little-endian 16-bit unsigned int
    return movie


def paintgen(
    meandark: int, 
    meanbright: int, 
    frames: int, 
    time: float, 
    photonrate: float, 
    photonratestd: float, 
    photonbudget: float,
) -> tuple[_np.ndarray, _np.ndarray, list]:
    """Paint-Generator: generates on and off-traces for given parameters
    and calculates the number of photons in each frame for a binding 
    site.

    Parameters
    ----------
    meandark : float
        Mean dark time (in ms) for the binding site.
    meanbright : float
        Mean bright time (in ms) for the binding site.
    frames : int
        Number of frames in the movie.
    time : float
        Time per frame (in ms).
    photonrate : float
        Mean photon rate (in photons per frame).
    photonratestd : float
        Standard deviation of the photon rate (in photons per frame).
    photonbudget : float
        Maximum number of photons that can be emitted in one on-event.  

    Returns
    -------
    photonsinframe : _np.ndarray
        Array containing the number of photons emitted in each frame.
    timetrace : _np.ndarray
        Array containing the time trace of the binding events.
    spotkinetics : list
        List containing the number of on-events, total bright events,
        simulated mean dark time, and simulated mean bright time.
    """

    meanlocs = 4 * int(
        _np.ceil(frames * time / (meandark + meanbright))
    )  # This is an estimate for the total number of binding events
    if meanlocs < 10:
        meanlocs = meanlocs * 10

    dark_times = _np.random.exponential(meandark, meanlocs)
    bright_times = _np.random.exponential(meanbright, meanlocs)

    events = _np.vstack((dark_times, bright_times)).reshape(
        (-1,), order="F"
    )  # Interweave dark_times and bright_times [dt,bt,dt,bt..]
    eventsum = _np.cumsum(events)
    maxloc = _np.argmax(
        eventsum > (frames * time)
    )  # Find the first event that exceeds the total integration time
    simulatedmeandark = _np.mean(events[:maxloc:2])

    simulatedmeanbright = _np.mean(events[1:maxloc:2])

    # check trace
    if _np.mod(maxloc, 2):  # uneven -> ends with an OFF-event
        onevents = int(_np.floor(maxloc / 2))
    else:  # even -> ends with bright event
        onevents = int(maxloc / 2)
    bright_events = _np.floor(maxloc / 2)  # number of bright_events

    photonsinframe = _np.zeros(
        int(frames + _np.ceil(meanbright / time * 20))
    )  # an on-event might be longer than the movie, so allocate more memory

    # calculate photon numbers
    for i in range(1, maxloc, 2):
        if photonratestd == 0:
            photons = _np.round(photonrate * time)
        else:
            photons = _np.round(
                _np.random.normal(photonrate, photonratestd) * time
            )  # Number of Photons that are emitted in one frame

        if photons < 0:
            photons = 0

        tempFrame = int(
            _np.floor(eventsum[i - 1] / time)
        )  # Get the first frame in which something happens in on-event
        onFrames = int(
            _np.ceil((eventsum[i] - tempFrame * time) / time)
        )  # Number of frames in which photon emittance happens

        if photons * onFrames > photonbudget:
            onFrames = int(
                _np.ceil(photonbudget / (photons * onFrames) * onFrames)
            )  # Reduce the number of on-frames if the photonbudget is reached

        for j in range(0, (onFrames)):
            if onFrames == 1:  # CASE 1: all photons are emitted in one frame
                photonsinframe[1 + tempFrame] = int(
                    _np.random.poisson(
                        ((tempFrame + 1) * time - eventsum[i - 1]) / time * photons
                    )
                )
            elif onFrames == 2:  # CASE 2: all photons are emitted in two frames
                emittedphotons = (
                    ((tempFrame + 1) * time - eventsum[i - 1]) / time * photons
                )
                if j == 1:  # photons in first onframe
                    photonsinframe[1 + tempFrame] = int(
                        _np.random.poisson(
                            ((tempFrame + 1) * time - eventsum[i - 1]) / time * photons
                        )
                    )
                else:  # photons in second onframe
                    photonsinframe[2 + tempFrame] = int(
                        _np.random.poisson(
                            (eventsum[i] - (tempFrame + 1) * time) / time * photons
                        )
                    )
            else:  # CASE 3: all photons are mitted in three or more frames
                if j == 1:
                    photonsinframe[1 + tempFrame] = int(
                        _np.random.poisson(
                            ((tempFrame + 1) * time - eventsum[i - 1]) / time * photons
                        )
                    )  # Indexing starts with 0
                elif j == onFrames:
                    photonsinframe[onFrames + tempFrame] = int(
                        _np.random.poisson(
                            (eventsum(i) - (tempFrame + onFrames - 1) * time)
                            / time
                            * photons
                        )
                    )
                else:
                    photonsinframe[tempFrame + j] = int(_np.random.poisson(photons))

        totalphotons = _np.sum(photonsinframe[1 + tempFrame : tempFrame + 1 + onFrames])
        if totalphotons > photonbudget:
            photonsinframe[onFrames + tempFrame] = int(
                photonsinframe[onFrames + tempFrame] - (totalphotons - photonbudget)
            )

    photonsinframe = photonsinframe[0:frames]
    timetrace = events[0:maxloc]

    if onevents > 0:
        spotkinetics = [
            onevents,
            sum(photonsinframe > 0),
            simulatedmeandark,
            simulatedmeanbright,
        ]
    else:
        spotkinetics = [0, sum(photonsinframe > 0), 0, 0]
    return photonsinframe, timetrace, spotkinetics


def distphotons(
    structures: _np.ndarray,
    itime: float,
    frames: int,
    taud: float,
    taub: float,
    photonrate: float,
    photonratestd: float,
    photonbudget: float,
) -> tuple[_np.ndarray, _np.ndarray, list]:
    """Distribute photons and binding kinetics for the given simulated 
    structures.

    Parameters
    ----------
    structures : _np.ndarray
        Array containing the binding sites' coordinates and exchange 
        information.
    itime : float
        Integration time for the simulation (in ms).
    frames : int
        Number of frames in the movie.
    taud : float
        Mean dark time for the binding sites (in ms).
    taub : float
        Mean bright time for the binding sites (in ms).
    photonrate : float
        Mean photon rate for the binding sites (in photons per frame).
    photonratestd : float
        Standard deviation of the photon rate (in photons per frame).
    photonbudget : float
        Maximum number of photons that can be emitted in one on-event.

    Returns
    -------
    photonsinframe : _np.ndarray
        Array containing the number of photons emitted in each frame.
    timetrace : _np.ndarray
        Array containing the time trace of the binding events.
    spotkinetics : list
        List containing the number of on-events, total bright events,
        simulated mean dark time, and simulated mean bright time.
    """
    
    time = itime
    meandark = int(taud)
    meanbright = int(taub)

    bindingsitesx = structures[0, :]
    bindingsitesy = structures[1, :]
    nosites = len(bindingsitesx)

    photonposall = _np.zeros((2, 0))
    photonposall = [1, 1]

    photonsinframe, timetrace, spotkinetics = paintgen(
        meandark,
        meanbright,
        frames,
        time,
        photonrate,
        photonratestd,
        photonbudget,
    )

    return photonsinframe, timetrace, spotkinetics


def distphotonsxy(
    runner: int, 
    photondist: _np.ndarray, 
    structures: _np.ndarray, 
    psf: float, 
    mode3Dstate: bool, 
    cx: list, 
    cy: list,
) -> _np.ndarray:
    """Distributes photons in a PSF, with an option for astigmatic PSF
    in 3D.
    
    Parameters
    ----------
    runner : int
        The index of the current binding site.
    photondist : _np.ndarray
        Array containing the number of photons emitted in each frame for
        each binding site.
    structures : _np.ndarray
        Array containing the binding sites' coordinates and exchange
        information.
    psf : float
        The point spread function (PSF) size (only for 2D).
    mode3Dstate : bool
        If True, uses a 3D astigmatic PSF; if False, uses a 2D Gaussian 
        PSF.
    cx, cy : list
        Calibration coefficients for the x/y dimension of the PSF, used 
        if mode3Dstate is True.
        
    Returns
    -------
    photonposframe : _np.ndarray
        Array containing the positions of the photons emitted in the
        current frame for the specified binding site.
    """

    bindingsitesx = structures[0, :]
    bindingsitesy = structures[1, :]
    bindingsitesz = structures[4, :]
    nosites = len(bindingsitesx)  # number of binding sites in image

    tempphotons = _np.array(photondist[:, runner]).astype(int)
    n_photons = _np.sum(tempphotons)
    n_photons_step = _np.cumsum(tempphotons)
    n_photons_step = _np.insert(n_photons_step, 0, 0)

    # Allocate memory
    photonposframe = _np.zeros((n_photons, 2))
    for i in range(0, nosites):
        photoncount = int(photondist[i, runner])
        if mode3Dstate:
            wx, wy = calculate_zpsf(bindingsitesz[i], cx, cy)
            cov = [[wx * wx, 0], [0, wy * wy]]
        else:
            cov = [[psf * psf, 0], [0, psf * psf]]

        if photoncount > 0:
            mu = [bindingsitesx[i], bindingsitesy[i]]
            photonpos = _np.random.multivariate_normal(mu, cov, photoncount)
            photonposframe[n_photons_step[i] : n_photons_step[i + 1], :] = photonpos

    return photonposframe


def convertMovie(
    runner: int,
    photondist: _np.ndarray,
    structures: _np.ndarray,
    imagesize: int,
    frames: int,
    psf: float,
    photonrate: float,
    background: float,
    noise: float,
    mode3Dstate: bool,
    cx: list,
    cy: list,
):
    """Converts the photon distribution into a simulated movie frame.
    
    Parameters
    ----------
    runner : int
        The index of the current binding site.
    photondist : _np.ndarray
        Array containing the number of photons emitted in each frame for
        each binding site.
    structures : _np.ndarray
        Array containing the binding sites' coordinates and exchange
        information.
    imagesize : int
        Size of the image (in pixels).
    frames : int
        Number of frames in the movie.
    psf : float
        The point spread function (PSF) size (sx/sy) for the binding
        sites.
    photonrate : float
        Mean photon rate for the binding sites (in photons per frame).
    background : float
        Background intensity (in photons per frame).
    noise : float
        Standard deviation of the noise to be added to the image.
    mode3Dstate : bool
        If True, uses a 3D astigmatic PSF; if False, uses a 2D Gaussian
        PSF.
    cx, cy : list
        Calibration coefficients for the x/y dimension of the PSF, used
        if mode3Dstate is True. 
        
    Returns
    ------- 
    simframe : _np.ndarray
        The simulated movie frame with the photon distribution and noise
        added.
    """

    edges = range(0, imagesize + 1)

    photonposframe = distphotonsxy(
        runner, photondist, structures, psf, mode3Dstate, cx, cy
    )

    if len(photonposframe) == 0:
        simframe = _np.zeros((imagesize, imagesize))
    else:
        x = photonposframe[:, 0]
        y = photonposframe[:, 1]
        simframe, xedges, yedges = _np.histogram2d(y, x, bins=(edges, edges))
        simframe = _np.flipud(simframe)  # to be consistent with render

    return simframe


def saveMovie(filename: str, movie: _np.ndarray, info: dict) -> None:
    """Saves the simulated movie to a file."""

    _io.save_raw(filename, movie, [info])


# Function to store the coordinates of a structure in a container.
# The coordinates wil be adjustet so that the center of mass is the origin
def defineStructure(
    structurexxpx: _np.ndarray,
    structureyypx: _np.ndarray,
    structureex: _np.ndarray,
    structure3d: _np.ndarray,
    pixelsize: float,
    mean: bool = True,
) -> _np.ndarray:
    """Define a structure with given coordinates and exchange 
    information.
    
    Parameters
    ----------
    structurexxpx : _np.ndarray
        Array containing the x-coordinates of the structure in pixels.
    structureyypx : _np.ndarray
        Array containing the y-coordinates of the structure in pixels.
    structureex : _np.ndarray
        Array containing the exchange information for the structure.
    structure3d : _np.ndarray
        Array containing the 3D coordinates of the structure.
    pixelsize : float
        The pixel size in nanometers.
    mean : bool, optional
        If True, centers the structure by subtracting the mean of the
        coordinates. Default is True.
        
    Returns
    -------
    structure : _np.ndarray
        Array containing the structure's x-positions, y-positions,
        exchange information, and 3D coordinates.
    """

    if mean:
        structurexxpx = structurexxpx - _np.mean(structurexxpx)
        structureyypx = structureyypx - _np.mean(structureyypx)
    # from px to nm
    structurexx = []
    for x in structurexxpx:
        structurexx.append(x / pixelsize)
    structureyy = []
    for x in structureyypx:
        structureyy.append(x / pixelsize)

    structure = _np.array(
        [structurexx, structureyy, structureex, structure3d]
    )  # FORMAT: x-pos,y-pos,exchange information

    return structure


def generatePositions(
    number: int, 
    imagesize: int, 
    frame: int, 
    arrangement: int,
) -> _np.ndarray:
    """Generate a set of positions where structures will be placed.

    Parameters
    ----------
    number : int
        Number of positions to generate.
    imagesize : int
        Size of the image (in pixels).
    frame : int
        Frame size to leave around the edges of the image.
    arrangement : int
        Arrangement type for the positions:
        - 0: Grid arrangement
        - 1: Random arrangement
    
    Returns
    -------
    gridpos : _np.ndarray
        Array containing the generated positions in the format
        [[x1, y1], [x2, y2], ...].
    """

    if arrangement == 0:
        spacing = int(_np.ceil((number**0.5)))
        linpos = _np.linspace(frame, imagesize - frame, spacing)
        [xxgridpos, yygridpos] = _np.meshgrid(linpos, linpos)
        xxgridpos = _np.ravel(xxgridpos)
        yygridpos = _np.ravel(yygridpos)
        xxpos = xxgridpos[0:number]
        yypos = yygridpos[0:number]
        gridpos = _np.vstack((xxpos, yypos))
        gridpos = _np.transpose(gridpos)
    else:
        gridpos = (imagesize - 2 * frame) * _np.random.rand(number, 2) + frame

    return gridpos


def rotateStructure(structure: _np.ndarray) -> _np.ndarray:
    """Rotate a structure randomly.
    
    Parameters
    ----------
    structure : _np.ndarray
        Array containing the structure's coordinates and exchange
        information.
        
    Returns
    -------
    newstructure : _np.ndarray
        Array containing the rotated structure's coordinates and
        exchange information.
    """

    angle_rad = _np.random.rand(1) * 2 * _np.pi
    newstructure = _np.array(
        [
            (structure[0, :]) * _np.cos(angle_rad)
            - (structure[1, :]) * _np.sin(angle_rad),
            (structure[0, :]) * _np.sin(angle_rad)
            + (structure[1, :]) * _np.cos(angle_rad),
            structure[2, :],
            structure[3, :],
        ]
    )
    return newstructure


def incorporateStructure(
    structure: _np.ndarray, 
    incorporation: float
) -> _np.ndarray:
    """Returns a subset of the structure to reflect incorporation of 
    staples.

    Parameters
    ----------
    structure : _np.ndarray
        Array containing the structure's coordinates and exchange
        information.
    incorporation : float
        Probability of incorporation for each staple in the structure. 
    
    Returns
    -------
    newstructure : _np.ndarray
        Array containing the subset of the structure after applying the
        incorporation probability.
    """
    newstructure = structure[:, (_np.random.rand(structure.shape[1]) < incorporation)]
    return newstructure


def randomExchange(pos: _np.ndarray) -> _np.ndarray:
    """Randomly shuffle exchange parameters for random labeling.

    Parameters
    ----------
    pos : _np.ndarray
        Array containing the positions and exchange information of the
        structures.
    
    Returns
    -------
    newpos : _np.ndarray
        Array containing the positions with shuffled exchange
        parameters.
    """

    arraytoShuffle = pos[2, :]
    _np.random.shuffle(arraytoShuffle)
    newpos = _np.array([pos[0, :], pos[1, :], arraytoShuffle, pos[3, :]])
    return newpos


def prepareStructures(
    structure: _np.ndarray, 
    gridpos: _np.ndarray, 
    orientation: int, 
    number: int, 
    incorporation: float, 
    exchange: int,
) -> _np.ndarray:
    """Prepares input positions, the structure definition considering 
    rotation etc.

    Parameters
    ----------
    structure : _np.ndarray
        Array containing the structure's coordinates and exchange
        information.
    gridpos : _np.ndarray
        Array containing the positions where structures will be placed.
    orientation : int
        Orientation of the structure:
        - 0: No rotation
        - 1: Random rotation
    number : int
        Number of structures to generate.
    incorporation : float
        Probability of incorporation for each staple in the structure.
    exchange : int
        If 1, randomizes the exchange parameters; if 0, keeps them as 
        is.

    Returns
    -------
    newpos : _np.ndarray
        Array containing the new positions of the structures after
        applying the specified transformations.
    """
    
    newpos = []
    oldstructure = _np.array(
        [structure[0, :], structure[1, :], structure[2, :], structure[3, :]]
    )

    for i in range(0, len(gridpos)):
        if orientation == 0:
            structure = oldstructure
        else:
            structure = rotateStructure(oldstructure)

        if incorporation == 1:
            pass
        else:
            structure = incorporateStructure(structure, incorporation)

        newx = structure[0, :] + gridpos[i, 0]
        newy = structure[1, :] + gridpos[i, 1]
        newstruct = _np.array(
            [
                newx,
                newy,
                structure[2, :],
                structure[2, :] * 0 + i,
                structure[3, :],
            ]
        )
        if i == 0:
            newpos = newstruct
        else:
            newpos = _np.concatenate((newpos, newstruct), axis=1)

    if exchange == 1:
        newpos = randomExchange(newpos)
    return newpos
