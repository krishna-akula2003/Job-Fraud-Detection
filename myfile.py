
def getFieldPlaces(fieldName,placeName):
    myd=['Account Executive','Marketing','Sales','Management','Other','Administrative','Business Development','Design','Business','chain','data entry']
    places=["hyderabad","chennai","bangalore","kolkata","pune","mumbai","noida","california","malasiya","US","New York","UK","London"]
    fn=fieldName.casefold()
    pn=placeName.casefold()
    res=list()
    if fn in myd or fieldName in myd:
        if pn in places or placeName in places:
            return 1
        return 0
    return 0


