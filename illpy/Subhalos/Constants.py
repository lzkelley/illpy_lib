

class SNAPSHOT():
    IDS                  = "ParticleIDs"
    POS                  = "Coordinates"

    POT                  = "Potential"
    DENS                 = "Denity"
    SFR                  = "StarFormationRate"
    VEL                  = "Velocities"
    EINT                 = "InternalEnergy"
    MASS                 = "Masses"

    HSML                 = "SmoothingLength"      # 2x max triangle radius

    SUBF_HSML            = "SubfindHsml"
    SUBF_VDISP           = "SubfindVelDisp"


    '''
    BH_MASS                  = "BH_Mass"
    BH_HSML                  = "BH_Hsml"
    BH_MDOT                  = "BH_Mdot"
    STELLAR_PHOTOS           = "GFM_StellarPhotometrics"
    FORM_TIME                = "GFM_StellarFormationTime"

    PARENT                   = "ParentID"
    NPART                    = "npart_loaded"
    '''



class SUBHALO():

    POS                  = "SubhaloPos"
    # SubhaloBHMdot
    VMAX                 = "SubhaloVmax"
    # SubhaloWindMass
    # SubhaloGasMetallicityMaxRad
    VDISP                = "SubhaloVelDisp"
    SFR                  = "SubhaloSFR"
    # SubhaloStarMetallicityMaxRad
    NUM_PARTS            = "SubhaloLen"
    SFR_HALF_RAD         = "SubhaloSFRinHalfRad"
    PHOTOS               = "SubhaloStellarPhotometrics"
    METZ                 = "SubhaloGasMetallicity"
    BH_MASS              = "SubhaloBHMass"
    MOST_BOUND           = "SubhaloIDMostbound"
    MASS_TYPE            = "SubhaloMassType"
    # SubhaloStellarPhotometricsMassInRad
    RAD_HALF_MASS        = "SubhaloHalfmassRad"
    # SubhaloParent
    # SubhaloSpin
    # SubhaloStarMetallicityHalfRad
    VEL                  = "SubhaloVel"
    NUM_PARTS_TYPE       = "SubhaloLenType"
    # SubhaloGasMetallicitySfrWeighted
    # SubhaloGasMetallicityHalfRad
    # SubhaloMassInRad
    NUM_GROUP            = "SubhaloGrNr"
    # SubhaloMassInHalfRad
    # SubhaloSFRinRad
    # SubhaloMassInMaxRad
    # SubhaloHalfmassRadType
    # SubhaloMassInMaxRadType
    COM                  = "SubhaloCM"
    # SubhaloStarMetallicity
    # SubhaloMassInHalfRadType
    MASS                 = "SubhaloMass"
    # SubhaloMassInRadType
    RAD_VMAX             = "SubhaloVmaxRad"
    # SubhaloSFRinMaxRad
    # SubhaloStellarPhotometricsRad
    # SubhaloGasMetallicitySfr

    @staticmethod
    def PROPERTIES(): 
        return [getattr(SUBHALO,it) for it in vars(SUBHALO) 
                if not it.startswith('_') and not callable(getattr(SUBHALO,it)) ]

# } class SUBHALO
