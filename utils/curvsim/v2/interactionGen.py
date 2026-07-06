def setMultitypeInteractions(labels, allowedattractions, general_unattractive, flanks_attractive, mid_attractive):
    allowedattractions_functional_copy = allowedattractions.copy()
    inputcontents = ""
    epsilon = general_unattractive['epsilon']
    sigma = general_unattractive['sigma']
    shift = general_unattractive['shift']
    wcacut = general_unattractive['wcacut']

    epsilon_flanks = flanks_attractive['epsilon_flanks']
    sigma_flanks = flanks_attractive['sigma_flanks']
    shift_flanks = flanks_attractive['shift_flanks']
    ljcut_flanks = flanks_attractive['ljcut_flanks']

    epsilon_mid = mid_attractive['epsilon_mid']
    sigma_mid = mid_attractive['sigma_mid']
    shift_mid = mid_attractive['shift_mid']
    ljcut_mid = mid_attractive['ljcut_mid']

    def associateValues(inputseries):
        newassociativedictionary = {}
        for i in range(len(inputseries)):
            newassociativedictionary[inputseries[i]] = i
        return newassociativedictionary

    def skew(basic, locus):
        return basic + 7 * locus

    numericassociation = associateValues(labels)

    for focusitem in labels:
        focusnum = numericassociation[focusitem]

        for skewedreferencenum in range(len(labels) - focusnum): # assigning repulsive interactions
            referencenum = skewedreferencenum + focusnum
            referenceitem = labels[referencenum]
            if referenceitem == focusitem:
                inputcontents += f"""
# upside down bonding 1 {focusitem} <-> {referenceitem}
pair_coeff {skew(1, focusnum)} {skew(5, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(3, focusnum)} {skew(3, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(1, focusnum)} {skew(3, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(1, focusnum)} {skew(1, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(3, focusnum)} {skew(5, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(5, focusnum)} {skew(5, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}

# repulsive ends
pair_coeff {skew(1, focusnum)} {skew(7, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(2, focusnum)} {skew(7, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(3, focusnum)} {skew(7, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(4, focusnum)} {skew(7, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(5, focusnum)} {skew(7, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(6, focusnum)} {skew(7, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(7, focusnum)} {skew(7, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}

# upside down bonding 2 {focusitem} <-> {referenceitem}
pair_coeff {skew(2, focusnum)} {skew(6, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(4, focusnum)} {skew(6, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(6, focusnum)} {skew(6, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(2, focusnum)} {skew(4, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(4, focusnum)} {skew(4, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(2, focusnum)} {skew(2, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
"""
            else:
                inputcontents += f"""
# upside down bonding 1 {focusitem} <-> {referenceitem}
pair_coeff {skew(1, focusnum)} {skew(5, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(5, focusnum)} {skew(1, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(3, focusnum)} {skew(3, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(1, focusnum)} {skew(3, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(3, focusnum)} {skew(1, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(1, focusnum)} {skew(1, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(3, focusnum)} {skew(5, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(5, focusnum)} {skew(3, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(5, focusnum)} {skew(5, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}

# repulsive ends
pair_coeff {skew(7, focusnum)} {skew(1, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(7, focusnum)} {skew(2, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(7, focusnum)} {skew(3, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(7, focusnum)} {skew(4, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(7, focusnum)} {skew(5, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(7, focusnum)} {skew(6, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(7, focusnum)} {skew(7, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(1, focusnum)} {skew(7, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(2, focusnum)} {skew(7, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(3, focusnum)} {skew(7, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(4, focusnum)} {skew(7, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(5, focusnum)} {skew(7, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(6, focusnum)} {skew(7, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}

# upside down bonding 2 {focusitem} <-> {referenceitem}
pair_coeff {skew(2, focusnum)} {skew(6, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(6, focusnum)} {skew(2, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(4, focusnum)} {skew(6, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(6, focusnum)} {skew(4, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(6, focusnum)} {skew(6, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(2, focusnum)} {skew(4, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(4, focusnum)} {skew(2, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(4, focusnum)} {skew(4, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(2, focusnum)} {skew(2, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
"""
        for skewedreferencenum in range(len(labels) - focusnum): # assign attractive interactions. if no attractive interaction is specified, repulsive interactions are implemented
            referencenum = skewedreferencenum + focusnum
            referenceitem = labels[referencenum]
            isrepulsive = True
            for attraction in allowedattractions_functional_copy:
                if focusitem in attraction:
                    print(focusitem)
                    objectofattractionlist = attraction.split('-')
                    print(objectofattractionlist)
                    objectofattractionlist.remove(str(focusitem))
                    objectofattraction = objectofattractionlist[0]
                    print(objectofattraction)
                    objectofattractionnum = numericassociation[objectofattraction]

                    if objectofattraction == referenceitem:
                        isrepulsive = False
                        allowedattractions_functional_copy.remove(attraction)
                        if objectofattraction == focusitem:
                            inputcontents += f"""
# correct attractive bonding {focusitem} <-> {objectofattraction}
pair_coeff {skew(1, focusnum)} {skew(2, objectofattractionnum)} lj/expand {epsilon_flanks} {sigma_flanks} {shift_flanks} {ljcut_flanks-shift_flanks} 
pair_coeff {skew(3, focusnum)} {skew(4, objectofattractionnum)} lj/expand {epsilon_mid} {sigma_mid} {shift_mid} {ljcut_mid-shift_mid} 
pair_coeff {skew(5, focusnum)} {skew(6, objectofattractionnum)} lj/expand {epsilon_flanks} {sigma_flanks} {shift_flanks} {ljcut_flanks-shift_flanks}
pair_coeff {skew(1, focusnum)} {skew(4, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(1, focusnum)} {skew(6, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(2, focusnum)} {skew(3, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(2, focusnum)} {skew(5, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(3, focusnum)} {skew(6, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(4, focusnum)} {skew(5, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
"""
                        else:
                            inputcontents += f"""
# correct attractive bonding {focusitem} <-> {objectofattraction}
pair_coeff {skew(1, focusnum)} {skew(2, objectofattractionnum)} lj/expand {epsilon_flanks} {sigma_flanks} {shift_flanks} {ljcut_flanks-shift_flanks} 
pair_coeff {skew(3, focusnum)} {skew(4, objectofattractionnum)} lj/expand {epsilon_mid} {sigma_mid} {shift_mid} {ljcut_mid-shift_mid} 
pair_coeff {skew(5, focusnum)} {skew(6, objectofattractionnum)} lj/expand {epsilon_flanks} {sigma_flanks} {shift_flanks} {ljcut_flanks-shift_flanks}
pair_coeff {skew(2, focusnum)} {skew(1, objectofattractionnum)} lj/expand {epsilon_flanks} {sigma_flanks} {shift_flanks} {ljcut_flanks-shift_flanks}
pair_coeff {skew(4, focusnum)} {skew(3, objectofattractionnum)} lj/expand {epsilon_mid} {sigma_mid} {shift_mid} {ljcut_mid-shift_mid} 
pair_coeff {skew(6, focusnum)} {skew(5, objectofattractionnum)} lj/expand {epsilon_flanks} {sigma_flanks} {shift_flanks} {ljcut_flanks-shift_flanks}
pair_coeff {skew(1, focusnum)} {skew(4, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(4, focusnum)} {skew(1, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(1, focusnum)} {skew(6, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(6, focusnum)} {skew(1, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(2, focusnum)} {skew(3, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(3, focusnum)} {skew(2, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(2, focusnum)} {skew(5, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(5, focusnum)} {skew(2, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(3, focusnum)} {skew(6, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(6, focusnum)} {skew(3, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(4, focusnum)} {skew(5, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(5, focusnum)} {skew(4, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
#pair_modify shift yes
"""
                    if isrepulsive:
                        if objectofattraction == focusitem:
                            inputcontents += f"""
# correct repulsive bonding {focusitem} <-> {referenceitem}
pair_coeff {skew(1, focusnum)} {skew(2, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift} 
pair_coeff {skew(3, focusnum)} {skew(4, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift} 
pair_coeff {skew(5, focusnum)} {skew(6, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(1, focusnum)} {skew(4, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(1, focusnum)} {skew(6, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(2, focusnum)} {skew(3, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(2, focusnum)} {skew(5, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(3, focusnum)} {skew(6, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(4, focusnum)} {skew(5, referencenum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
#pair_modify shift yes
"""
                        else:
                            inputcontents += f"""
# correct repulsive bonding {focusitem} <-> {referenceitem}
pair_coeff {skew(1, focusnum)} {skew(2, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(3, focusnum)} {skew(4, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(5, focusnum)} {skew(6, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(2, focusnum)} {skew(1, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(4, focusnum)} {skew(3, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(6, focusnum)} {skew(5, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(1, focusnum)} {skew(4, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(4, focusnum)} {skew(1, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(1, focusnum)} {skew(6, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(6, focusnum)} {skew(1, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(2, focusnum)} {skew(3, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(3, focusnum)} {skew(2, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(2, focusnum)} {skew(5, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(5, focusnum)} {skew(2, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(3, focusnum)} {skew(6, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(6, focusnum)} {skew(3, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(4, focusnum)} {skew(5, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff {skew(5, focusnum)} {skew(4, objectofattractionnum)} lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
#pair_modify shift yes

"""
    
    return inputcontents