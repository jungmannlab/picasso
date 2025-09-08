"""
    picasso.simulate
    ~~~~~~~~~~~~~~~~

    Simulate single molecule fluorescence data.

    :author: Maximilian Thomas Strauss, 2016-2018
    :copyright: Copyright (c) 2016-2018 Jungmann Lab, MPI of Biochemistry
"""

import numpy as np
from . import io
from numba import njit

magfac = 0.79


@njit
def calculate_zpsf(
    z: np.ndarray | float,
    cx: np.ndarray,
    cy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | tuple[float, float]:
    """Calculate the astigmatic PSF size at a given z position.

    Parameters
    ----------
    z : np.ndarray or number
        The z position(s) at which to calculate the PSF.
    cx, cy : np.ndarray
        Coefficients for the x/y dimension of the PSF.

    Returns
    -------
    wx, wy : np.ndarray
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


def test_calculate_zpsf() -> np.ndarray:
    """Test function for calculate_zpsf."""
    cx = np.array([1, 2, 3, 4, 5, 6, 7])
    cy = np.array([1, 2, 3, 4, 5, 6, 7])
    z = np.array([1, 2, 3, 4, 5, 6, 7])
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
    """Save metadata to a YAML file."""
    io.save_info(filename, [info], default_flow_style=True)


def noisy(image: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Add gaussian noise to an image.

    Parameters
    ----------
    image : np.ndarray
        The input image to which noise will be added.
    mu : float
        The mean of the Gaussian noise.
    sigma : float
        The standard deviation of the Gaussian noise.

    Returns
    -------
    noisy : np.ndarray
        The noisy image with Gaussian noise added.
    """
    row, col = image.shape  # Variance for np.random is 1
    gauss = sigma * np.random.normal(0, 1, (row, col)) + mu
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    noisy[noisy < 0] = 0
    return noisy


def noisy_p(image: np.ndarray, mu: float) -> np.ndarray:
    """Add Poisson noise to an image or movie.

    Parameters
    ----------
    image : np.ndarray
        The input image to which Poisson noise will be added.
    mu : float
        The mean of the Poisson noise.

    Returns
    -------
    noisy : np.ndarray
        The noisy image with Poisson noise added.
    """
    poiss = np.random.poisson(mu, image.shape).astype(float)
    noisy = image + poiss
    return noisy


def check_type(movie: np.ndarray) -> np.ndarray:
    """Check the type of the movie and convert it to a 16-bit unsigned
    integer, if necessary.

    Parameters
    ----------
    movie : np.ndarray
        The input movie to be checked and converted.

    Returns
    -------
    movie : np.ndarray
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
) -> tuple[np.ndarray, np.ndarray, list]:
    """Paint-Generator: generate on and off-traces for given parameters
    and calculate the number of photons in each frame for a binding
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
        Maximum number of photons that can be emitted by one emitter.

    Returns
    -------
    photonsinframe : np.ndarray
        Array containing the number of photons emitted in each frame.
    timetrace : np.ndarray
        Array containing the time trace of the binding events.
    spotkinetics : list
        List containing the number of on-events, total bright events,
        simulated mean dark time, and simulated mean bright time.
    """
    meanlocs = 4 * int(
        np.ceil(frames * time / (meandark + meanbright))
    )  # This is an estimate for the total number of binding events
    if meanlocs < 10:
        meanlocs = meanlocs * 10

    dark_times = np.random.exponential(meandark, meanlocs)
    bright_times = np.random.exponential(meanbright, meanlocs)

    events = np.vstack((dark_times, bright_times)).reshape(
        (-1,), order="F"
    )  # Interweave dark_times and bright_times [dt,bt,dt,bt..]
    eventsum = np.cumsum(events)
    maxloc = np.argmax(
        eventsum > (frames * time)
    )  # Find the first event that exceeds the total integration time
    simulatedmeandark = np.mean(events[:maxloc:2])

    simulatedmeanbright = np.mean(events[1:maxloc:2])

    # check trace
    if np.mod(maxloc, 2):  # uneven -> ends with an OFF-event
        onevents = int(np.floor(maxloc / 2))
    else:  # even -> ends with bright event
        onevents = int(maxloc / 2)

    photonsinframe = np.zeros(
        int(frames + np.ceil(meanbright / time * 20))
    )  # an on-event might be longer than the movie, so allocate more memory

    # calculate photon numbers
    for i in range(1, maxloc, 2):
        if photonratestd == 0:
            photons = np.round(photonrate * time)
        else:
            photons = np.round(
                np.random.normal(photonrate, photonratestd) * time
            )  # Number of Photons that are emitted in one frame

        if photons < 0:
            photons = 0

        tempFrame = int(
            np.floor(eventsum[i - 1] / time)
        )  # Get the first frame in which something happens in on-event
        onFrames = int(
            np.ceil((eventsum[i] - tempFrame * time) / time)
        )  # Number of frames in which photon emittance happens

        if photons * onFrames > photonbudget:
            onFrames = int(
                np.ceil(photonbudget / (photons * onFrames) * onFrames)
            )  # Reduce the number of on-frames if the photonbudget is reached

        for j in range(0, (onFrames)):
            if onFrames == 1:  # CASE 1: all photons are emitted in one frame
                photonsinframe[1 + tempFrame] = int(
                    np.random.poisson(
                        ((tempFrame + 1) * time - eventsum[i - 1])
                        / time * photons
                    )
                )
            # CASE 2: all photons are emitted in two frames
            elif onFrames == 2:
                if j == 1:  # photons in first onframe
                    photonsinframe[1 + tempFrame] = int(
                        np.random.poisson(
                            ((tempFrame + 1) * time - eventsum[i - 1])
                            / time * photons
                        )
                    )
                else:  # photons in second onframe
                    photonsinframe[2 + tempFrame] = int(
                        np.random.poisson(
                            (eventsum[i] - (tempFrame + 1) * time)
                            / time * photons
                        )
                    )
            else:  # CASE 3: all photons are mitted in three or more frames
                if j == 1:
                    photonsinframe[1 + tempFrame] = int(
                        np.random.poisson(
                            ((tempFrame + 1) * time - eventsum[i - 1])
                            / time * photons
                        )
                    )  # Indexing starts with 0
                elif j == onFrames:
                    photonsinframe[onFrames + tempFrame] = int(
                        np.random.poisson(
                            (eventsum(i) - (tempFrame + onFrames - 1) * time)
                            / time
                            * photons
                        )
                    )
                else:
                    photonsinframe[tempFrame + j] = (
                        int(np.random.poisson(photons))
                    )

        totalphotons = (
            np.sum(photonsinframe[1 + tempFrame:tempFrame + 1 + onFrames])
        )
        if totalphotons > photonbudget:
            photonsinframe[onFrames + tempFrame] = int(
                photonsinframe[onFrames + tempFrame]
                - (totalphotons - photonbudget)
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
    structures: np.ndarray,
    itime: float,
    frames: int,
    taud: float,
    taub: float,
    photonrate: float,
    photonratestd: float,
    photonbudget: float,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Distribute photons and binding kinetics for the given simulated
    structures.

    Parameters
    ----------
    structures : np.ndarray
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
        Maximum number of photons that can be emitted by one emitter.

    Returns
    -------
    photonsinframe : np.ndarray
        Array containing the number of photons emitted in each frame.
    timetrace : np.ndarray
        Array containing the time trace of the binding events.
    spotkinetics : list
        List containing the number of on-events, total bright events,
        simulated mean dark time, and simulated mean bright time.
    """
    time = itime
    meandark = int(taud)
    meanbright = int(taub)

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
    photondist: np.ndarray,
    structures: np.ndarray,
    psf: float,
    mode3Dstate: bool,
    cx: list,
    cy: list,
) -> np.ndarray:
    """Distribute photons in a PSF, with an option for astigmatic PSF
    in 3D.

    Parameters
    ----------
    runner : int
        The index of the current binding site.
    photondist : np.ndarray
        Array containing the number of photons emitted in each frame for
        each binding site.
    structures : np.ndarray
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
    photonposframe : np.ndarray
        Array containing the positions of the photons emitted in the
        current frame for the specified binding site.
    """
    bindingsitesx = structures[0, :]
    bindingsitesy = structures[1, :]
    bindingsitesz = structures[4, :]
    nosites = len(bindingsitesx)  # number of binding sites in image

    tempphotons = np.array(photondist[:, runner]).astype(int)
    n_photons = np.sum(tempphotons)
    n_photons_step = np.cumsum(tempphotons)
    n_photons_step = np.insert(n_photons_step, 0, 0)

    # Allocate memory
    photonposframe = np.zeros((n_photons, 2))
    for i in range(0, nosites):
        photoncount = int(photondist[i, runner])
        if mode3Dstate:
            wx, wy = calculate_zpsf(bindingsitesz[i], cx, cy)
            cov = [[wx * wx, 0], [0, wy * wy]]
        else:
            cov = [[psf * psf, 0], [0, psf * psf]]

        if photoncount > 0:
            mu = [bindingsitesx[i], bindingsitesy[i]]
            photonpos = np.random.multivariate_normal(mu, cov, photoncount)
            photonposframe[
                n_photons_step[i]:n_photons_step[i + 1], :
            ] = photonpos

    return photonposframe


def convertMovie(
    runner: int,
    photondist: np.ndarray,
    structures: np.ndarray,
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
    """Convert the photon distribution into a simulated movie frame.

    Parameters
    ----------
    runner : int
        The index of the current binding site.
    photondist : np.ndarray
        Array containing the number of photons emitted in each frame for
        each binding site.
    structures : np.ndarray
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
    simframe : np.ndarray
        The simulated movie frame with the photon distribution and noise
        added.
    """
    edges = range(0, imagesize + 1)

    photonposframe = distphotonsxy(
        runner, photondist, structures, psf, mode3Dstate, cx, cy
    )

    if len(photonposframe) == 0:
        simframe = np.zeros((imagesize, imagesize))
    else:
        x = photonposframe[:, 0]
        y = photonposframe[:, 1]
        simframe, xedges, yedges = np.histogram2d(y, x, bins=(edges, edges))
        simframe = np.flipud(simframe)  # to be consistent with render

    return simframe


def saveMovie(filename: str, movie: np.ndarray, info: dict) -> None:
    """Save the simulated movie to a file."""
    io.save_raw(filename, movie, [info])


# Function to store the coordinates of a structure in a container.
# The coordinates wil be adjustet so that the center of mass is the origin
def defineStructure(
    structurexxpx: np.ndarray,
    structureyypx: np.ndarray,
    structureex: np.ndarray,
    structure3d: np.ndarray,
    pixelsize: float,
    mean: bool = True,
) -> np.ndarray:
    """Define a structure with given coordinates and exchange
    information.

    Parameters
    ----------
    structurexxpx : np.ndarray
        Array containing the x-coordinates of the structure in pixels.
    structureyypx : np.ndarray
        Array containing the y-coordinates of the structure in pixels.
    structureex : np.ndarray
        Array containing the exchange information for the structure.
    structure3d : np.ndarray
        Array containing the 3D coordinates of the structure.
    pixelsize : float
        The pixel size in nanometers.
    mean : bool, optional
        If True, centers the structure by subtracting the mean of the
        coordinates. Default is True.

    Returns
    -------
    structure : np.ndarray
        Array containing the structure's x-positions, y-positions,
        exchange information, and 3D coordinates.
    """
    if mean:
        structurexxpx = structurexxpx - np.mean(structurexxpx)
        structureyypx = structureyypx - np.mean(structureyypx)
    # from px to nm
    structurexx = []
    for x in structurexxpx:
        structurexx.append(x / pixelsize)
    structureyy = []
    for x in structureyypx:
        structureyy.append(x / pixelsize)

    structure = np.array(
        [structurexx, structureyy, structureex, structure3d]
    )  # FORMAT: x-pos,y-pos,exchange information

    return structure


def generatePositions(
    number: int,
    imagesize: int,
    frame: int,
    arrangement: int,
) -> np.ndarray:
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
    gridpos : np.ndarray
        Array containing the generated positions in the format
        [[x1, y1], [x2, y2], ...].
    """
    if arrangement == 0:
        spacing = int(np.ceil((number**0.5)))
        linpos = np.linspace(frame, imagesize - frame, spacing)
        [xxgridpos, yygridpos] = np.meshgrid(linpos, linpos)
        xxgridpos = np.ravel(xxgridpos)
        yygridpos = np.ravel(yygridpos)
        xxpos = xxgridpos[0:number]
        yypos = yygridpos[0:number]
        gridpos = np.vstack((xxpos, yypos))
        gridpos = np.transpose(gridpos)
    else:
        gridpos = (imagesize - 2 * frame) * np.random.rand(number, 2) + frame

    return gridpos


def rotateStructure(structure: np.ndarray) -> np.ndarray:
    """Rotate a structure randomly.

    Parameters
    ----------
    structure : np.ndarray
        Array containing the structure's coordinates and exchange
        information.

    Returns
    -------
    newstructure : np.ndarray
        Array containing the rotated structure's coordinates and
        exchange information.
    """
    angle_rad = np.random.rand(1) * 2 * np.pi
    newstructure = np.array(
        [
            (structure[0, :]) * np.cos(angle_rad)
            - (structure[1, :]) * np.sin(angle_rad),
            (structure[0, :]) * np.sin(angle_rad)
            + (structure[1, :]) * np.cos(angle_rad),
            structure[2, :],
            structure[3, :],
        ]
    )
    return newstructure


def incorporateStructure(
    structure: np.ndarray,
    incorporation: float
) -> np.ndarray:
    """Return a subset of the structure to reflect incorporation of
    staples.

    Parameters
    ----------
    structure : np.ndarray
        Array containing the structure's coordinates and exchange
        information.
    incorporation : float
        Probability of incorporation for each staple in the structure.

    Returns
    -------
    newstructure : np.ndarray
        Array containing the subset of the structure after applying the
        incorporation probability.
    """
    newstructure = (
        structure[:, (np.random.rand(structure.shape[1]) < incorporation)]
    )
    return newstructure


def randomExchange(pos: np.ndarray) -> np.ndarray:
    """Randomly shuffle exchange parameters for random labeling.

    Parameters
    ----------
    pos : np.ndarray
        Array containing the positions and exchange information of the
        structures.

    Returns
    -------
    newpos : np.ndarray
        Array containing the positions with shuffled exchange
        parameters.
    """
    arraytoShuffle = pos[2, :]
    np.random.shuffle(arraytoShuffle)
    newpos = np.array([pos[0, :], pos[1, :], arraytoShuffle, pos[3, :]])
    return newpos


def prepareStructures(
    structure: np.ndarray,
    gridpos: np.ndarray,
    orientation: int,
    number: int,
    incorporation: float,
    exchange: int,
) -> np.ndarray:
    """Prepare input positions, the structure definition considering
    rotation etc.

    Parameters
    ----------
    structure : np.ndarray
        Array containing the structure's coordinates and exchange
        information.
    gridpos : np.ndarray
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
    newpos : np.ndarray
        Array containing the new positions of the structures after
        applying the specified transformations.
    """
    newpos = []
    oldstructure = np.array(
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
        newstruct = np.array(
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
            newpos = np.concatenate((newpos, newstruct), axis=1)

    if exchange == 1:
        newpos = randomExchange(newpos)
    return newpos
