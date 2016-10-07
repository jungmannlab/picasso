"""
    picasso.simulate
    ~~~~~~~~~~~~~~~~

    Simulate single molcule fluorescence data

    :author: Maximilian Thomas Strauss, 2016
    :copyright: Copyright (c) 2016 Jungmann Lab, Max Planck Institute of Biochemistry
"""
import numpy as _np
from . import io as _io

def saveInfo(filename,info):
    _io.save_info(filename, [info], default_flow_style=True)

def noisy(image,mu,sigma):        #Add gaussian noise to an image.
    row,col= image.shape  #Variance for _np.random is 1
    gauss = sigma*_np.random.normal(0,1,(row,col)) + mu
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    noisy[noisy<0]=0
    return noisy

def paintgen( meandark,meanbright,frames,time,photonrate,photonratestd,photonbudget ): #Paint-Generator: Generates on and off-traces for given parameters. Calculates the number of Photons in each frame for a binding site
    meanlocs = int(_np.ceil(frames*time/(meandark+meanbright))) #This is an estimate for the total number of binding events
    if meanlocs < 10:
        meanlocs = meanlocs*5
    else:
        meanlocs = meanlocs*2

    ## GENERATE ON AND OFF-EVENTS
    dark_times = _np.random.exponential(meandark,meanlocs)
    bright_times = _np.random.exponential(meanbright,meanlocs)

    events = _np.vstack((dark_times,bright_times)).reshape((-1,),order='F') # Interweave dark_times and bright_times [dt,bt,dt,bt..]
    simulatedmeandark = _np.mean(events[::2])
    simulatedmeanbright = _np.mean(events[1::2])
    eventsum = _np.cumsum(events)

    maxloc = next(x[0] for x in enumerate(eventsum) if x[1] > frames*time) #Find the first event that exceeds the total integration time

    ## CHECK Trace
    if _np.mod(maxloc,2): #uneven -> ends with an OFF-event
        onevents = int(_np.floor(maxloc/2));
    else: #even -> ends with bright event
        onevents = int(maxloc/2);
    bright_events = _np.floor(maxloc/2); #number of bright_events

    #AN ON-EVENT MIGHT BE LONGER THAN THE MOVIE, SO ALLOCATE MORE MEMORY, AS AN ESTIMATE: MEANBRIGHT/time*10
    photonsinframe = _np.zeros(int(frames+_np.ceil(meanbright/time*20)))

    ## CALCULATE PHOTON NUMBERS
    for i in range(1,maxloc,2):
        photons = _np.round(_np.random.normal(photonrate,photonratestd)*time) #Number of Photons that are emitted in one frame
        if photons < 0:
            photons = 0

        tempFrame = int(_np.floor(eventsum[i-1]/time)); #Get the first frame in which something happens in on-event
        onFrames = int(_np.ceil((eventsum[i]-tempFrame*time)/time)); #Number of frames in which photon emittance happens

        if photons*onFrames > photonbudget:
            onFrames = int(_np.ceil(photonbudget/(photons*onFrames)*onFrames)) #Reduce the number of on-frames once the photonbudget is reached

        for j in range(0,(onFrames)): #LOOP THROUGH ALL ONFRAMES

            if onFrames == 1: #CASE 1: ALL PHOTONS ARE EMITTED IN ONE FRAME
                photonsinframe[1+tempFrame]=int(_np.random.poisson(((tempFrame+1)*time-eventsum[i-1])/time*photons))
            elif onFrames == 2: #CASE 2: ALL PHOTONS ARE EMITTED IN TWO FRAMES
                emittedphotons = (((tempFrame+1)*time-eventsum[i-1])/time*photons)
                if j == 1: # PHOTONS IN FIRST ONFRAME
                    photonsinframe[1+tempFrame]=int(_np.random.poisson(((tempFrame+1)*time-eventsum[i-1])/time*photons))
                else: # PHOTONS IN SECOND ONFRAME
                    photonsinframe[2+tempFrame]=int(_np.random.poisson((eventsum[i]-(tempFrame+1)*time)/time*photons))
            else: # CASE 3: ALL PHOTONS ARE EMITTED IN THREE OR MORE FRAMES
                if j == 1:
                    photonsinframe[1+tempFrame]=int(_np.random.poisson(((tempFrame+1)*time-eventsum[i-1])/time*photons))  #Indexing starts with 0
                elif j == onFrames:
                    photonsinframe[onFrames+tempFrame]=int(_np.random.poisson((eventsum(i)-(tempFrame+onFrames-1)*time)/time*photons))
                else:
                    photonsinframe[tempFrame+j]=int(_np.random.poisson(photons))

        totalphotons = _np.sum(photonsinframe[1+tempFrame:tempFrame+1+onFrames])
        if totalphotons > photonbudget:
            photonsinframe[onFrames+tempFrame]=int(photonsinframe[onFrames+tempFrame]-(totalphotons-photonbudget))

    photonsinframe = photonsinframe[0:frames] #Cach exception if a trace should be longer than the movie
    timetrace = events[0:maxloc]

    if onevents > 0:
        spotkinetics = [onevents,sum(photonsinframe>0),simulatedmeandark,simulatedmeanbright]
    else:
        spotkinetics = [0,sum(photonsinframe>0),0,0]
    #spotkinetics is an output variable, that gives out the number of on-events, the number of localizations, the mean of the dark and bright times
    return photonsinframe,timetrace,spotkinetics

def distphotons(structures,itime,frames,taud,taub,photonrate,photonratestd,photonbudget):

    time = itime
    meandark = int(taud)
    meanbright = int(taub)

    bindingsitesx = structures[0,:]
    bindingsitesy = structures[1,:]
    nosites  = len(bindingsitesx) # number of binding sites in image

    #PHOTONDIST: DISTRIBUTE PHOTONS FOR ALL BINDING SITES

    photonposall = _np.zeros((2,0))
    photonposall = [1,1]

    photonsinframe,timetrace,spotkinetics = paintgen(meandark,meanbright,frames,time,photonrate,photonratestd,photonbudget)

    return photonsinframe

def convertMovie(runner, photondist,structures,imagesize,frames,psf,photonrate,background, noise):

    pixels = imagesize

    bindingsitesx = structures[0,:]
    bindingsitesy = structures[1,:]
    nosites  = len(bindingsitesx) # number of binding sites in image


    #FRAMEWISE SIMULATION OF PSF
    #ALL PHOTONS FOR 1 STRUCTURE IN ALL FRAMES
    edges = range(0,pixels+1)
    #ALLCOATE MEMORY
    movie = _np.zeros(shape=(frames,pixels,pixels), dtype='<u2')

    flag = 0
    photonposframe = _np.zeros((2,0))

    for i in range(0,nosites):
        tempphotons = photondist[i,:]
        photoncount = int(tempphotons[runner])

        if photoncount > 0:
            flag = flag+1
            mu = [bindingsitesx[i],bindingsitesy[i]]
            cov = [[psf*psf, 0], [0, psf*psf]]
            photonpos = _np.random.multivariate_normal(mu, cov, photoncount)
            if flag == 1:
                photonposframe = photonpos
            else:
                photonposframe = _np.concatenate((photonposframe,photonpos),axis=0)

        #HANDLE CASE FOR NO PHOTONS DETECTED AT ALL IN FRAME
    if photonposframe.size == 0:
        simframe = _np.zeros((pixels,pixels))
    else:
        xx = photonposframe[:,0]
        yy = photonposframe[:,1]
        simframe, xedges, yedges = _np.histogram2d(yy,xx,bins=(edges,edges))
        simframe = _np.flipud(simframe) # to be consistent with render
    simframenoise = noisy(simframe,background,noise)
    simframeout=_np.round(simframenoise).astype('<u2')

    return simframeout



def saveMovie(filename,movie,info):
    _io.save_raw(filename, movie, [info])


def defineStructure(structurexxpx,structureyypx,structureex,pixelsize): #Function to store the coordinates of a structure in a container. The coordinates wil be adjustet so that the center of mass is the origin
    structurexxpx = structurexxpx-_np.mean(structurexxpx)
    structureyypx = structureyypx-_np.mean(structureyypx)
    #from px to nm
    structurexx = []
    for x in structurexxpx:
        structurexx.append(x/pixelsize)
    structureyy = []
    for x in structureyypx:
        structureyy.append(x/pixelsize)

    structure = _np.array([structurexx, structureyy,structureex]) #FORMAT: x-pos,y-pos,exchange information

    return structure

def generatePositions(number,imagesize,frame,arrangement): #GENERATE A SET OF POSITIONS WHERE STRUCTURES WILL BE PLACED

    if arrangement==0:
        spacing = _np.ceil((number**0.5))
        linpos = _np.linspace(frame,imagesize-frame,spacing)
        [xxgridpos,yygridpos]=_np.meshgrid(linpos,linpos)
        xxgridpos = _np.ravel(xxgridpos)
        yygridpos = _np.ravel(yygridpos)
        xxpos = xxgridpos[0:number]
        yypos = yygridpos[0:number]
        gridpos =_np.vstack((xxpos,yypos))
        gridpos = _np.transpose(gridpos)
    else:
        gridpos = (imagesize-2*frame)*_np.random.rand(number,2)+frame

    return gridpos

def rotateStructure(structure): #ROTATE A STRUCTURE RANDOMLY
    angle_rad = _np.random.rand(1)*2*3.141592
    newstructure = _np.array([(structure[0,:])*_np.cos(angle_rad)-(structure[1,:])*_np.sin(angle_rad),
                (structure[0,:])*_np.sin(angle_rad)+(structure[1,:])*_np.cos(angle_rad),
                structure[2,:]])
    return newstructure

def incorporateStructure(structure,incorporation): #CONSIDER STAPLE INCORPORATION
    newstructure = structure[:,(_np.random.rand(structure.shape[1])<incorporation)]
    return newstructure

def randomExchange(pos): # RANDOMLY SHUFFLE EXCHANGE PARAMETERS ('RANDOM LABELING')
    arraytoShuffle = pos[2,:]
    _np.random.shuffle(arraytoShuffle)
    newpos = _np.array([pos[0,:],pos[1,:],arraytoShuffle,pos[3,:]])
    return newpos

def prepareStructures(structure,gridpos,orientation,number,incorporation,exchange): #prepareStructures: Input positions, the structure definition, consider rotation etc.
    newpos = []
    oldstructure = structure
    for i in range(0,len(gridpos)):#LOOP THROUGH ALL POSITIONS
        if orientation == 0:
            pass
        else:
            structure = rotateStructure(oldstructure)
        if incorporation == 1:
            pass
        else:
            structure = incorporateStructure(structure,incorporation)
        newx = structure[0,:]+gridpos[i,0]
        newy = structure[1,:]+gridpos[i,1]
        newstruct = _np.array([newx,newy,structure[2,:],structure[2,:]*0+i])
        if i == 0:
            newpos = newstruct
        else:
            newpos = _np.concatenate((newpos,newstruct),axis=1)

    if exchange == 1:
        newpos = randomExchange(newpos)

    return newpos
