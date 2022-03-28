import numpy as np
class MapMinMaxApplier(object):
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept
    def __call__(self, x):
        return x * self.slope + self.intercept
    def reverse(self, y):
        return (y-self.intercept) / self.slope
 
def mapminmax2(x, ymin=-1, ymax=+1):
	x = np.asanyarray(x).reshape((-1,1))
	xmax = x.max(axis=0)
	xmin = x.min(axis=0)
	if (xmax==xmin).any():
		raise ValueError("some rows have no variation")
	slope = ((ymax-ymin) / (xmax - xmin))[:,np.newaxis]
	intercept = (-xmin*(ymax-ymin)/(xmax-xmin))[:,np.newaxis] + ymin
	ps = MapMinMaxApplier(slope, intercept)
	return ps(x), ps