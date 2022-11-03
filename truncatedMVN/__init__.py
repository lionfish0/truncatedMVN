from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

class NoValidStartPointFoundError(ValueError):
    '''raise this when the method can't find a good start location'''
    
class InvalidCovariance(ValueError):
    '''raise this when the covariance is singular'''    

class TruncMVN():
    def __init__(self,mean,cov,planes,thinning=1,burnin=None,verbose=False):
        """
        Create the truncated MVN. You need to specify the mean and covariance of the multivariate normal (MVN), and
        the normals to the planes that will truncate this MVN. We assume for this version that the planes pass through
        the origin (as this was our use case).
        
        mean = (D) vector: location of the MVN
        cov = (DxD) matrix: covariance of the MVN
        planes = (NxD) matrix, each row a normal to a plane (vector points to 'valid' side of plane)
        
        thinning = how much thinning to do during sampling
        burnin = how many iterations of burnin to have (defaults to number of iterations used to generate samples)
        """
        self.mean = mean.astype(float)
        self.cov = cov.astype(float)
        self.Phi = planes.astype(float)
        self.thinning = thinning
        self.burnin = burnin
        self.verbose = verbose
        if np.linalg.det(cov)<1e-9: raise InvalidCovariance("Covariance singular")
        
    def getboundaries(self,x,axis):
        """
        For a location 'x', and a given axis direction (specified by 'axis'),
             find the points of these planes nearest to x along the axis passing through x.
             
        This algorithm assumes x is in a valid location, so doesn't check the 'direction' of each normal. Instead it
             just looks for the nearest plane-axis-crossing point on each side of x along the axis.
             
        Example: Four planes cross the axes, in this case two on each side of x. We want the locations of the nearest
             points to x, marked as 'a' and 'b':
            \   \                           /     |
             \   \                         /      |
              \   \                       /       |
            ---\---a------------x--------b--------|-------axis
                \   \                   /         |
                 \   \                 /          |
        Returns: (a list) of the two scalar values of these two points (a) and (b) along the axis.
        """
        
        #avoids division by zero warnings
        P = self.Phi.T[axis]
        P[P==0]=1e-9
        vs = -((self.Phi[:,:axis] @ x[:axis]) + (self.Phi[:,(axis+1):] @ x[(axis+1):]))/P
        
        lowv = vs[vs<x[axis]]
        highv = vs[vs>x[axis]]
        if len(lowv)==0: 
            lowv = -1000000
        else:
            lowv = max(lowv)
        if len(highv)==0: 
            highv = 1000000
        else:
            highv = min(highv)
            
        return [lowv,highv]   
     
    def getnearestpoint(self,w,x,margin=1e-4):
        """Find nearest point on plane (passing through origin) defined by normal 'w', to point x.
        Adds an additional little margin, to get it to lie on the positive side."""
        w = w / np.sqrt(np.sum(w**2))
        d = w.T@x
        d-=margin
        p = x - d * w
        return p

    def findstartpoint(self,margin=1e-8):
        startx = self.mean.copy()
        if self.verbose: print("Finding Start Point")
        for it in range(10000):
            if self.verbose: print(".",end="")
            v = self.Phi @ startx
            idx = np.where(v<margin)[0] #get planes that we are on the wrong side of
            if len(idx)==0: 
                return startx
            startx = self.getnearestpoint(self.Phi.T[:,idx[0]],startx)
        if self.verbose: print("")            
        raise NoValidStartPointFoundError("No valid start location has been found. Specify one, using 'initx' and/or check the domain contains non-zero space (i.e. that the truncations don't completely occlude the space.")
        
    def sample_truncnorm(self,a,b,loc,scale,size):
        """
        Sample from a one dimensional (univariate) truncated normal, that has mean at 'loc', standard deviation 'scale',
        and is truncated from 'a' to 'b'. Returns 'size' samples.
        """
        
        #if we have a singular covariance, then we're going have problems with a and b being invalid, etc...
        #this isn't going to mix, etc... but we'll leave this code here for now.
        if scale<=1e-9:
            if loc<a: return a
            if loc>b: return b
            return loc
            
        a, b = (a - loc) / scale, (b - loc) / scale
        return truncnorm.rvs(a,b,loc=loc,scale=scale,size=size)

    def oldsample(self,initx=None,samples=10):
        """
        Sample from the truncated MVN.
        Parameters:
            initx = the start location for sampling, if not set, then the tool tries to find a valid (non-zero/truncated) location
                        close to the MVN's mean.
            samples = number of samples (default 10).
        """
        if initx is None: 
            initx = self.findstartpoint()
            
        x = initx
        xs = []
        #remcovinv = []
        remcovmat = []
        remcovinvtimesrmmat = []        
        if self.verbose: print("Computing inverses etc, for conditional distributions.")
        for axis in range(len(self.mean)):
            #remcovinv.append(np.linalg.inv(np.delete(np.delete(self.cov,axis,0),axis,1)))
            remcovmat.append(np.delete(self.cov[axis,:],axis))
            remcovinvtimesrmmat.append(np.delete(self.cov[axis,:],axis) @ np.linalg.inv(np.delete(np.delete(self.cov,axis,0),axis,1)))

        if self.burnin == None:
            burnin = samples * self.thinning
        else:
            burnin = self.burnin
        if self.verbose: print("Sampling")
        for it in range(samples*self.thinning+burnin):
            if self.verbose: print("%5d/%5d [%s]\r" % (it,samples*self.thinning+burnin,"burn-in" if it<burnin else "samples"), end="")
            for axis in range(len(self.mean)):
                #remcov = np.delete(np.delete(self.cov,axis,0),axis,1)                
                #cond_var = self.cov[axis,axis] - remcovmat[axis] @ remcovinv[axis] @ remcovmat[axis].T
                #cond_mean = (self.mean[axis] + remcovmat[axis] @ remcovinv[axis] @ (np.delete(x,axis) - np.delete(self.mean,axis)))
                
                cond_var = self.cov[axis,axis] - remcovinvtimesrmmat[axis] @ remcovmat[axis].T
                cond_mean = (self.mean[axis] + remcovinvtimesrmmat[axis] @ (np.delete(x,axis) - np.delete(self.mean,axis)))
                
                
                bs = self.getboundaries(x,axis)
                
                s = self.sample_truncnorm(bs[0],bs[1],cond_mean,np.sqrt(cond_var),1)
                x[axis] = s
            if it>=burnin:
                if it%self.thinning == 0:
                    xs.append(x.copy())
        if self.verbose: print("")                    
        xs = np.array(xs)
        return xs
        
    def sample(self,initx=None,samples=10):
        """
        Sample from the truncated MVN.
        Parameters:
            initx = the start location for sampling, if not set, then the tool tries to find a valid (non-zero/truncated) location
                        close to the MVN's mean.
            samples = number of samples (default 10).
        """
        if initx is None: 
            initx = self.findstartpoint()
            
        x = initx
        xs = []
        
        if self.verbose: print("Computing inverse, for conditional distributions.")
        invcov = np.linalg.inv(self.cov)
        
        if self.burnin == None:
            burnin = samples * self.thinning
        else:
            burnin = self.burnin
            
        if self.verbose: print("Sampling")            
        for it in range(samples*self.thinning+burnin):
            if self.verbose: print("%5d/%5d [%s]\r" % (it,samples*self.thinning+burnin,"burn-in" if it<burnin else "samples"), end="")
            for axis in range(len(self.mean)):
                cond_var = 1/invcov[axis,axis] #self.cov[axis,axis] - remcovinvtimesrmmat[axis] @ remcovmat[axis].T
                cond_mean = x[axis] - (invcov@(x-self.mean))[axis]/invcov[axis,axis] #(self.mean[axis] + remcovinvtimesrmmat[axis] @ (np.delete(x,axis) - np.delete(self.mean,axis)))
                bs = self.getboundaries(x,axis)
                s = self.sample_truncnorm(bs[0],bs[1],cond_mean,np.sqrt(cond_var),1)
                x[axis] = s
            if it>=burnin:
                if it%self.thinning == 0:
                    xs.append(x.copy())
        if self.verbose: print("")                    
        xs = np.array(xs)
        return xs        
    
    def plot(self,samples=100):
        """
        For the two dimensional problem we can plot the planes, and the ellipse representing the MVN, and samples.
        """
        for phi in self.Phi:
            
            d = (phi / np.sqrt(np.sum(phi**2)))/10
            for i in range(10):
                plt.plot(np.array([-phi[1],phi[1]])-d[0]*i,np.array([phi[0],-phi[0]])-d[1]*i,'k-',alpha=0.4)
            plt.plot(np.array([-phi[1],phi[1]]),np.array([phi[0],-phi[0]]),'k-')
            
        confidence_ellipse(self.mean,self.cov,n_std=1)
        confidence_ellipse(self.mean,self.cov,n_std=0.2)

        samps = self.sample(samples=samples)
        plt.scatter(samps[:,0],samps[:,1],1)
            
       
    def compute_gelman_rubin(self,Nsamples=100,Nchains=10):
        """
        Computes the Gelman Rubin statistic. Note that we use the same start location, so this might
        cause an overestimate of the convergence.
        
        Returns a vector of GR statistics (one for each axis).
        
        These should all be about 1 (e.g. no more than 1.2).
        """
        chainmeans = []
        chainvars = []
        for chains in range(Nchains):
            samps = self.sample(samples=Nsamples)
            chainmeans.append(np.mean(samps,0))
            chainvars.append(np.var(samps,0,ddof=1))
        chainmeans = np.array(chainmeans)

        winthinchainvar = np.mean(chainvars)
        betweenchainvar = Nsamples*np.var(chainmeans,0,ddof=1)

        GR = (((Nsamples-1)/Nsamples) * winthinchainvar + (1/Nsamples) * betweenchainvar) / winthinchainvar
        return GR
                    
#Including plotting function for the contour ellipse to let us easily plot the demo/example.
def confidence_ellipse(mean, cov, ax=None, n_std=1.0, edgecolor='black',facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse

    Parameters
    ----------

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if ax is None: ax = plt.gca()
        
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, edgecolor=edgecolor,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)
