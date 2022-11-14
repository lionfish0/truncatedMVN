from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.optimize import minimize
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
        self.startpoint = None #needs to be computed.
        #self.fast_populate_truncnorm()
        

    def getboundaries(self,x,axis,update_axis):
        """
        For a location 'x', and a given axis direction (specified by 'axis'),
             find the points of these planes nearest to x along the axis passing through x.

        This algorithm assumes x is in a valid location, so doesn't check the 'direction' of each normal. Instead it
             just looks for the nearest plane-axis-crossing point on each side of x along the axis.

        For speed we use self.previous and self.previousx.
         - self.previous is the previous value of self.Phi @ self.x
         - self.previousx is the value of x[update_axis] before it was modified
         - the update_axis parameter specifies which axis we need to update (typically this will be axis-1)

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
        #PhiTimesx = self.previous - self.Phi[:,update_axis]*self.previousx + self.Phi[:,update_axis]*x[update_axis]
        PhiTimesx = self.previous + self.Phi[:,update_axis]*(x[update_axis]-self.previousx)
        vs = -(PhiTimesx - self.Phi[:,axis]*x[axis])/P
        #vs = -((self.Phi[:,:axis] @ x[:axis]) + (self.Phi[:,(axis+1):] @ x[(axis+1):]))/P
        #print(np.max(np.abs(vs-vs2)))
        lowv = vs[vs<x[axis]]
        highv = vs[vs>x[axis]]
        if len(lowv)==0: 
            lowv = -1000000
        else:
            lowv = np.max(lowv)
        if len(highv)==0: 
            highv = 1000000
        else:
            highv = np.min(highv)
        self.previous = PhiTimesx
        self.previousx = x[axis]
        return [lowv,highv]
        
             
        
    def getnearestpoint(self,w,x,margin=1e-4):
        """Find nearest point on plane (passing through origin) defined by normal 'w', to point x.
        Adds an additional little margin, to get it to lie on the positive side."""
        w = w / np.sqrt(np.sum(w**2))
        d = w.T@x
        d-=margin
        p = x - d * w
        return p

    def findstartpoint(self,margin=1e-4):
        """Finds the start point by minimising the distance from the mean, while constrained by
        the planes (plus the margin).
        """
        if self.verbose: print("Finding Start Point")
        fun = lambda x: np.sum((x-self.mean)**2)
        cons = ({'type': 'ineq', 'fun': lambda x:  self.Phi@x-margin})
        res = minimize(fun, self.mean, method='SLSQP',
                       constraints=cons)
        if not np.all(self.Phi @ res.x > 0):
            raise NoValidStartPointFoundError("No valid start location has been found. Specify one, using 'initx' and/or check the domain contains non-zero space (i.e. that the truncations don't completely occlude the space.")
        return res.x
        
    def old_findstartpoint(self,margin=1e-8):
        """Finds the start point by starting at the mean, and moving to the nearest point on each plane.
        """
        startx = self.mean.copy()
        if self.verbose: print("Finding Start Point")
        for it in range(10000):
            if self.verbose: print(".",end="")
            v = self.Phi @ startx
            idx = np.where(v<margin)[0] #get planes that we are on the wrong side of
            if len(idx)==0: 
                if self.verbose: print("")
                return startx
            startx = self.getnearestpoint(self.Phi.T[:,idx[0]],startx)

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

     
    def fast_populate_truncnorm(self):
        res = 300
        self.fast_samples = 2000
        self.samps = np.zeros([res,res,self.fast_samples])
        self.indexes = np.zeros([res,res]).astype(int)
        self.vals = np.linspace(-300,300,res)
        for i,a in enumerate(self.vals):
            print("%3d/%3d\r" % (i,len(self.vals)),end="")
            for j,b in enumerate(self.vals):
                if b<=a: continue
                s = self.sample_truncnorm(a,b,loc=0,scale=1,size=self.fast_samples)
                self.samps[i,j,:] = s
        print("")
               
    
    def fast_sample_truncnorm(self,a,b,loc,scale):
        try: 

            newa, newb = (a - loc) / scale, (b - loc) / scale
            wa = np.where(self.vals<=newa)[0][-1]
            wb = np.where(self.vals<=newb)[0][-1]+1
            startindex = self.indexes[wa,wb]
            s = self.samps[wa,wb,startindex:(startindex+30)]
            try:
                index = np.argmax((newa<s) & (s<newb))
            except ValueError:
                print("!",end="")
                #self.fast_populate_truncnorm()
                s = self.sample_truncnorm(self.vals[wa],self.vals[wb],loc=0,scale=1,size=self.fast_samples)
                self.samps[wa,wb,:] = s
                self.indexes[wa,wb] = 0
                return self.fast_sample_truncnorm(a,b,loc,scale)
            if (newa>=s[index]) & (s[index]>=newb):
                print("x",end="")
                raise IndexError("Not quickly finding solution")
            self.indexes[wa,wb] += index + 1
            return self.samps[wa,wb,index+startindex]*scale + loc
            
        except IndexError:
            print("m",end="")
            #print("m (newa=%0.2f,newb=%0.2f)" % (newa,newb),end="")
            return self.sample_truncnorm(a,b,loc=loc,scale=scale,size=1)



    def sample(self,initx=None,samples=10,usecaching=False):
        """
        Sample from the truncated MVN.
        Parameters:
            initx = the start location for sampling, if not set, then the tool tries to find a valid (non-zero/truncated) location
                        close to the MVN's mean.
            samples = number of samples (default 10).
            usecaching = whether to precache samples (the truncnorm rvs method is faster if you ask for lots of samples at once!)
        """
        if usecaching:
            if not hasattr(self,'fast_samples'): self.fast_populate_truncnorm()
        
        if initx is None: 
            if self.startpoint is None:
                self.startpoint = self.findstartpoint()
            initx = self.startpoint.copy()
        else:
            try:
                initx[0]
            except TypeError:
                raise TypeError("Need to pass a vector (of length equal to the number of dimensions) as initx location, or leave as None to compute automatically.")
            
        x = initx
        xs = []
        
        if self.verbose: print("Computing inverse, for conditional distributions.")
        invcov = np.linalg.inv(self.cov)
        
        if self.burnin == None:
            burnin = samples * self.thinning
        else:
            burnin = self.burnin
            
        if self.verbose: print("Sampling")            
        
        self.previous = self.Phi @ x
        self.previousx = x[-1]

        self.invprevious = invcov@(x-self.mean)
        self.previousxminusmean = x[-1] - self.mean[-1]
        
        for it in range(samples*self.thinning+burnin):
            if self.verbose: print("%5d/%5d [%s]\r" % (it,samples*self.thinning+burnin,"burn-in" if it<burnin else "samples"), end="")
            for axis in range(len(self.mean)):
                cond_var = 1/invcov[axis,axis] #self.cov[axis,axis] - remcovinvtimesrmmat[axis] @ remcovmat[axis].T
                #invcovmod = self.invprevious - invcov[:,axis-1]*self.previousxminusmean + invcov[:,axis-1]*(x[axis-1]-self.mean[axis-1])
                invcovmod = self.invprevious + invcov[:,axis-1]*((x[axis-1]-self.mean[axis-1])-self.previousxminusmean)
    
                self.invprevious = invcovmod
                self.previousxminusmean = x[axis] - self.mean[axis]
                cond_mean = x[axis] - invcovmod[axis]/invcov[axis,axis]

                bs = self.getboundaries(x,axis,axis-1)                
                
                if usecaching:
                    s = self.fast_sample_truncnorm(bs[0],bs[1],cond_mean,np.sqrt(cond_var))                
                else:
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
            
       
    def compute_gelman_rubin(self,Nsamples=100,Nchains=10,usecaching=False):
        """
        Computes the Gelman Rubin statistic. Note that we use the same start location, so this might
        cause an overestimate of the convergence.
        
        Returns a vector of GR statistics (one for each axis).
        
        These should all be about 1 (e.g. no more than 1.2).
        """
        chainmeans = []
        chainvars = []
        for chains in range(Nchains):
            if self.verbose: print("chain %d of %d\r" % (chains,Nchains),end="")
            samps = self.sample(samples=Nsamples,usecaching=usecaching)
            chainmeans.append(np.mean(samps,0))
            chainvars.append(np.var(samps,0,ddof=1))
        chainmeans = np.array(chainmeans)

        winthinchainvar = np.mean(chainvars)
        betweenchainvar = Nsamples*np.var(chainmeans,0,ddof=1)

        GR = (((Nsamples-1)/Nsamples) * winthinchainvar + (1/Nsamples) * betweenchainvar) / winthinchainvar
        return GR
             
             
class NotFindingEnoughSamplesDueToRejection(ValueError):
    '''raise this when the method tries for ages to find any samples with rejection sampling'''
    
class TruncMVNrejection(TruncMVN):
    def __init__(self,mean,cov,planes,verbose=False):
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
        self.verbose = verbose

    def sample(self,samples=10):
        """
        Sample from the truncated MVN.
        Parameters:
            samples = number of samples (default 10).
        """
        xs = np.zeros([0,len(self.mean)])
        for it in range(100):
            samps = np.random.multivariate_normal(self.mean,self.cov,1000)
            keep = np.min(self.Phi @ samps.T,0)>0
            xs = np.r_[xs,samps[keep,:]]
            if len(xs)>samples:
                return xs[:samples,:]
        raise NotFindingEnoughSamplesDueToRejection
        
class TruncMVNreparam(TruncMVN):
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
        self.L = np.linalg.cholesky(cov)
        self.invL = np.linalg.inv(self.L) #TODO Don't need to invert maybe if we have chole.?
        self.transformed_planes = planes @ self.L
        self.mean = mean.astype(float)
        self.cov = cov.astype(float)
        self.Phi = planes.astype(float)
        self.thinning = thinning
        self.burnin = burnin
        self.verbose = verbose
        self.transformed_cov = np.eye(len(cov))
        self.transformed_mean = mean @ self.invL.T        
        self.TMVN = TruncMVN(self.transformed_mean,self.transformed_cov,self.transformed_planes,thinning=thinning,burnin=burnin,verbose=verbose)
        

    def sample(self,initx=None,samples=10,usecaching=False):
        """
        Sample from the truncated MVN.
        Parameters:
            initx = the start location for sampling, if not set, then the tool tries to find a valid (non-zero/truncated) location
                        close to the MVN's mean.
            samples = number of samples (default 10).
            usecaching = whether to precache samples (the truncnorm rvs method is faster if you ask for lots of samples at once!)
        """
        return self.TMVN.sample(initx=initx,samples=samples,usecaching=usecaching) @ self.L.T
           
        

                    
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
