#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:59:09 2017

@author: nsde
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm as scipy_expm
from ddtn.helper.utility import get_dir, load_obj, save_obj, make_hashable
from ddtn.helper.math import null, create_grid

#%%
class setup_CPAB_transformer:
    def __init__(self,   ncx = 2, 
                         ncy = 2, 
                         valid_outside = True,
                         zero_trace = False, 
                         zero_boundary = False,
                         name = 'cpab_basis',
                         override = False):
        """
        Main class for setting up cpab_transformer object. The main purpose of
        calling this class is to produce a file "cbap_basis.pkl" that contains
        all information needed for the transformation.
        
        Arguments:
            ncx:            number of rectangular cells in x direction
            ncy:            number of rectangular cells in y direction
            valid_outside:  boolean, determines if transformation is valid
                            outside the image region
            zero_trace:     boolean, if true the transformation is area 
                            preserving <--> each affine transformation have
                            zero trace
            zero_boundary:  boolean, if true the velocity at the image boundary
                            is constrained to be zero. NOTE: zero_boundary and
                            valid_outside cannot both be True or False at the
                            same time
            name:           str, name for the created bases file. Default is
                            'cpab_basis', but can be used to create multiple
                            basis files for easy switch between them
            override:       if True, then a new basis will be saved to 
                            'cbap_basis.pkl' even if it already exists
        """
        
        # We cannot have zero boundary and valid_outside at the same time
        assert valid_outside != zero_boundary, '''valid_outside and zero_boundary
            cannot both be active or deactive at the same time, CHOOSE'''
        
        # Domain information
        self.valid_outside = valid_outside
        self.zero_trace = zero_trace
        self.zero_boundary = zero_boundary
        self.minbound = [-1, -1]
        self.maxbound = [1, 1]
        self.ncx = ncx
        self.ncy = ncy
        self.nC = 4*ncx*ncy
        self.inc_x = (self.maxbound[0] - self.minbound[0]) / self.ncx
        self.inc_y = (self.maxbound[1] - self.minbound[1]) / self.ncy
        self.Ashape = [2,3]
        self.Asize = np.prod(self.Ashape)
        dir_loc = get_dir(__file__)
        self.filename = dir_loc + '/../' + name
        
        # Try to load file with basis and vertices
        try:    
            file = load_obj(self.filename)
            if override:
                raise print('File ' + name + '.pkl already exist, ' \
                            'but override == True, ' \
                            'so updating basis with new settings')
            # File found -> load information
            self.valid_outside = file['valid_outside']
            self.zero_trace = file['zero_trace']
            self.zero_boundary = file['zero_boundary']
            self.B = file['B']
            self.nConstrains = file['nConstrains']
            self.cells_multiidx = file['cells_multiidx']
            self.cells_verts = file['cells_verts']
            self.ncx = file['ncx']
            self.ncy = file['ncy']
            self.nC = 4*self.ncx*self.ncy
            self.inc_x = (self.maxbound[0] - self.minbound[0]) / self.ncx
            self.inc_y = (self.maxbound[1] - self.minbound[1]) / self.ncy
            loaded = True
        except: # Else create it
            # Call tessalation and get vertices of cells
            self.cells_multiidx, self.cells_verts  = self.tessalation()
            
            # Find shared vertices (edges) where a continuity constrain needs to hold
            self.shared_v, self.shared_v_idx = self.find_shared_verts()
            
            # If the transformation should be valid outside of the image domain, 
            # calculate the auxiliary points and add them to the edges where a 
            # continuity constrain should be
            if self.valid_outside:
                shared_v_outside, shared_v_idx_outside = self.find_shared_verts_outside()
                if shared_v_outside.size != 0:
                    self.shared_v = np.concatenate((self.shared_v, shared_v_outside))
                    self.shared_v_idx = np.concatenate((self.shared_v_idx, shared_v_idx_outside))
            
            # Create L
            L = self.create_continuity_constrains()
            
            # Update L with extra constrains if needed
            if self.zero_trace:
                Ltemp = self.create_zero_trace_constrains()
                L = np.vstack((L, Ltemp))
            
            if self.zero_boundary:
                Ltemp = self.create_zero_boundary_constrains()
                L = np.vstack((L, Ltemp))
            
            # Number of constrains
            self.nConstrains = L.shape[0]
            
            # Find the null space of L, which is the basis B
            self.B = null(L)
            
            # Save all information
            save_obj({
                      'B': self.B,
                      'D': self.B.shape[0],
                      'd': self.B.shape[1],
                      'nConstrains': self.nConstrains, 
                      'cells_multiidx': self.cells_multiidx,
                      'cells_verts': self.cells_verts,
                      'nC': self.nC,
                      'ncx': self.ncx,
                      'ncy': self.ncy,
                      'inc_x': self.inc_x,
                      'inc_y': self.inc_y,
                      'minbound': self.minbound, 
                      'maxbound': self.maxbound,
                      'valid_outside': self.valid_outside,
                      'zero_trace': self.zero_trace,
                      'zero_boundary': self.zero_boundary
                     }, self.filename)
            loaded = False
        
        # Get shapes of PA space and CPA space
        self.D, self.d = self.B.shape
        
        # Print information about basis
        print(70*'-')
        if loaded:
            print('Loaded file ' + name + '.pkl, ' \
                  'containing tessalation with settings:')
        else:
            print('Creating file ' + name +'.pkl, ' \
                  'containing tessalation with settings:')
        print('    nx = {0}, ny = {1}'.format(self.ncx, self.ncy))
        print('    valid outside     = {0}'.format(self.valid_outside))
        print('    zero boundary     = {0}'.format(self.zero_boundary))
        print('    volume preserving = {0}'.format(self.zero_trace))
        print('With these settings, theta.shape = {0}x1'.format(self.B.shape[1]))
        print(70*'-')
            
        
    def tessalation(self):
        """ Finds the coordinates of all cell vertices """
        xmin, ymin = self.minbound
        xmax, ymax = self.maxbound
        Vx = np.linspace(xmin, xmax, self.ncx+1)
        Vy = np.linspace(ymin, ymax, self.ncy+1)
        cells_x = [ ]
        cells_x_verts = [ ]
        for i in range(self.ncy):
            for j in range(self.ncx):
                ul = tuple([Vx[j],Vy[i],1])
                ur = tuple([Vx[j+1],Vy[i],1])
                ll = tuple([Vx[j],Vy[i+1],1])
                lr = tuple([Vx[j+1],Vy[i+1],1])
                
                center = [(Vx[j]+Vx[j+1])/2,(Vy[i]+Vy[i+1])/2,1]
                center = tuple(center)                 
                
                cells_x_verts.append((center,ul,ur))  # order matters!
                cells_x_verts.append((center,ur,lr))  # order matters!
                cells_x_verts.append((center,lr,ll))  # order matters!
                cells_x_verts.append((center,ll,ul))  # order matters!                

                cells_x.append((j,i,0))
                cells_x.append((j,i,1))
                cells_x.append((j,i,2))
                cells_x.append((j,i,3))
        
        return  cells_x, np.asarray(cells_x_verts)
    
    def find_shared_verts(self):
        """ Find all pair of cells that share a vertices that encode continuity
            constrains inside the domain
        """
        nC = self.nC
        shared_v = [ ]
        shared_v_idx = [ ]
        for i in range(nC):
            for j in range(nC):
                vi = make_hashable(self.cells_verts[i])
                vj = make_hashable(self.cells_verts[j])
                shared_verts = set(vi).intersection(vj)
                if len(shared_verts) == 2 and (j,i) not in shared_v_idx:
                    shared_v.append(list(shared_verts))
                    shared_v_idx.append((i,j))
                
        return np.array(shared_v), shared_v_idx
    
    def find_shared_verts_outside(self):
        """ Find all pair of cells that share a vertices that encode continuity
            constrains outside the domain
        """
        shared_v = [ ]
        shared_v_idx = [ ]

        left =   np.zeros((self.nC, self.nC), np.bool)    
        right =  np.zeros((self.nC, self.nC), np.bool) 
        top =    np.zeros((self.nC, self.nC), np.bool) 
        bottom = np.zeros((self.nC, self.nC), np.bool) 

        for i in range(self.nC):
            for j in range(self.nC):
                
                vi = make_hashable(self.cells_verts[i])
                vj = make_hashable(self.cells_verts[j])
                shared_verts = set(vi).intersection(vj)
                
                mi = self.cells_multiidx[i]
                mj = self.cells_multiidx[j]
        
                # leftmost col, left triangle, adjacent rows
                if  mi[0]==mj[0]==0 and \
                    mi[2]==mj[2]==3 and \
                    np.abs(mi[1]-mj[1])==1: 
                        
                    left[i,j]=True
                
                # rightmost col, right triangle, adjacent rows                 
                if  mi[0]==mj[0]==self.ncx-1 and \
                    mi[2]==mj[2]==1 and \
                    np.abs(mi[1]-mj[1])==1: 

                    right[i,j]=True
                
                # uppermost row, upper triangle , adjacent cols                    
                if  mi[1]==mj[1]==0 and \
                    mi[2]==mj[2]==0 and \
                    np.abs(mi[0]-mj[0])==1:
                        
                    top[i,j]=True
                
                # lowermost row, # lower triangle, # adjacent cols            
                if  mi[1]==mj[1]==self.ncy-1 and \
                    mi[2]==mj[2]==2 and \
                    np.abs(mi[0]-mj[0])==1:
                        
                    bottom[i,j]=True
                                
                if  len(shared_verts) == 1 and \
                    any([left[i,j],right[i,j],top[i,j],bottom[i,j]]) and \
                    (j,i) not in shared_v_idx:
                        
                    v_aux = list(shared_verts)[0] # v_aux is a tuple
                    v_aux = list(v_aux) # Now v_aux is a list (i.e. mutable)
                    if left[i,j] or right[i,j]:
                        v_aux[0]-=10 # Create a new vertex  with the same y
                    elif top[i,j] or bottom[i,j]:
                        v_aux[1]-=10 # Create a new vertex  with the same x
                    else:
                        raise ValueError("WTF?")                        
                    shared_verts = [tuple(shared_verts)[0], tuple(v_aux)]
                    shared_v.append(shared_verts)
                    shared_v_idx.append((i,j))
        
        return np.array(shared_v), shared_v_idx
        
        
        
    def create_continuity_constrains(self):
        """ Based on the vertices found that are shared by cells, construct
            continuity constrains 
        """
        Ltemp = np.zeros(shape=(0,6*self.nC))
        count = 0
        for i,j in self.shared_v_idx:
    
            # Row 1 [x_a^T 0_{1x3} -x_a^T 0_{1x3}]
            row1 = np.zeros(shape=(6*self.nC))
            row1[(6*i):(6*(i+1))] = np.append(np.array(self.shared_v[count][0]), 
                                              np.zeros((1,3)))
            row1[(6*j):(6*(j+1))] = np.append(-np.array(self.shared_v[count][0]), 
                                              np.zeros((1,3)))
            
            # Row 2 [0_{1x3} x_a^T 0_{1x3} -x_a^T]
            row2 = np.zeros(shape=(6*self.nC))
            row2[(6*i):(6*(i+1))] = np.append(np.zeros((1,3)), 
                                              np.array(self.shared_v[count][0]))
            row2[(6*j):(6*(j+1))] = np.append(np.zeros((1,3)), 
                                              -np.array(self.shared_v[count][0]))
            
            # Row 3 [x_b^T 0_{1x3} -x_b^T 0_{1x3}]
            row3 = np.zeros(shape=(6*self.nC))
            row3[(6*i):(6*(i+1))] = np.append(np.array(self.shared_v[count][1]), 
                                              np.zeros((1,3)))
            row3[(6*j):(6*(j+1))] = np.append(-np.array(self.shared_v[count][1]), 
                                              np.zeros((1,3)))
            
            # Row 4 [0_{1x3} x_b^T 0_{1x3} -x_b^T]
            row4 = np.zeros(shape=(6*self.nC))
            row4[(6*i):(6*(i+1))] = np.append(np.zeros((1,3)), 
                                              np.array(self.shared_v[count][1]))
            row4[(6*j):(6*(j+1))] = np.append(np.zeros((1,3)), 
                                              -np.array(self.shared_v[count][1]))
                        
            Ltemp = np.vstack((Ltemp, row1, row2, row3, row4))
            
            count += 1
        
        return Ltemp
        
    def create_zero_trace_constrains(self):
        """ Construct zero trace (volume perservation) constrains """
        Ltemp = np.zeros(shape=(self.nC, 6*self.nC))
        for c in range(self.nC):
            Ltemp[c,(6*c):(6*(c+1))] = np.array([1,0,0,0,1,0])
        return Ltemp
        
    def create_zero_boundary_constrains(self):
        """ Construct zero boundary i.e. fixed boundary constrains. Note that 
            points on the upper and lower bound can still move to the left and 
            right and points on the left and right bound can still move up 
            and down. Thus, they are only partial zero. 
        """
        xmin, ymin = self.minbound
        xmax, ymax = self.maxbound
        Ltemp = np.zeros(shape=(0,6*self.nC))
        for c in range(self.nC):
            for v in self.cells_verts[c]:
                if(v[0] == xmin or v[0] == xmax): 
                    row = np.zeros(shape=(6*self.nC))
                    row[(6*c):(6*(c+1))] = np.append(np.zeros((1,3)),v)
                    Ltemp = np.vstack((Ltemp, row))
                if(v[1] == ymin or v[1] == ymax): 
                    row = np.zeros(shape=(6*self.nC))
                    row[(6*c):(6*(c+1))] = np.append(v,np.zeros((1,3)))
                    Ltemp = np.vstack((Ltemp, row))
        return Ltemp
    
    def get_size_theta(self):
        return self.d
        
    def theta2Avees(self, theta):
        """ Calculate Avees = B*theta, where Avees will be a Dx1 vector with the
            row-by-row flatten affine transformations {A_1, A_2, ..., A_nC}
        """
        Avees = self.B.dot(theta)
        return Avees
        
    def Avees2As(self, Avees):
        """ Reshape the output of theta2Avees into a 3D matrix with shape
            (nC, 2, 3) i.e As[0] will be the affine transformation A_0 belonging
            to cell 0
        """
        As = np.reshape(Avees, (self.nC, self.Ashape[0], self.Ashape[1]))
        return As
    
    def As2squareAs(self, As):
        """ Concatenate a zero to each affine transformation, such that they
            become square matrices
        """
        squareAs = np.zeros(shape=(self.nC, 3, 3))
        squareAs[:,:-1,:] = As
        return squareAs
    
    def find_cell_idx(self, p):
        # Given a point p, finds which cell it belongs to
        p = p[0:2] - self.minbound # Move with respect to the center
        
        p0 = np.fmin(self.ncx*self.inc_x-1e-8, np.fmax(0.0, p[0]))
        p1 = np.fmin(self.ncy*self.inc_y-1e-8, np.fmax(0.0, p[1]))
        
        xmod = np.mod(p0, self.inc_x)
        ymod = np.mod(p1, self.inc_y)
        
        x = xmod / self.inc_x
        y = ymod / self.inc_y
        
        def mymin(a, b):
            return a if a < b else np.round(b)
        
        cell_idx = 4 * (mymin(self.ncx-1, (p0 - xmod) / self.inc_x) + 
                        mymin(self.ncy-1, (p1 - ymod) / self.inc_y) * self.ncx)
        
        # Out of bound (left)
        if(p[0] <= 0):
            if(p[1] <= 0 and p[1] / self.inc_y < p[0] / self.inc_x):
                pass
            elif(p[1] >= self.ncy * self.inc_y and p[1] / self.inc_y - self.ncy > -p[0] / self.inc_x):
                cell_idx += 2
            else:
                cell_idx += 3
            return int(cell_idx)
            
        # Out of bound (right)
        if(p[0] >= self.ncx * self.inc_x):
            if(p[1] <= 0 and -p[1] / self.inc_y > p[0] / self.inc_x - self.ncx):
                pass
            elif(p[1] >= self.ncy * self.inc_y and p[1] / self.inc_y - self.ncy > p[0] / self.inc_x - self.ncx):
                cell_idx += 2
            else:
                cell_idx += 1
            return int(cell_idx)
        
        # Out of bound (up)
        if(p[1] <= 0):
            return int(cell_idx)
        
        # Out of bound (bottom)
        if(p[1] >= self.ncy * self.inc_y):
            cell_idx += 2
            return int(cell_idx)
        
        # In bound
        if(x < y):
            if(1-x < y):
                cell_idx += 2
            else:
                cell_idx += 3
        elif(1-x < y):
            cell_idx += 1
        
        return int(cell_idx)
    
    def sample_grid(self, nb_points = 1000):
        """ Samples nb_points in both directions within the image domain and 
            returns a matrix of size (nb_points^2, 2), where each row is point
        """
        return create_grid(self.minbound, self.maxbound, [nb_points, nb_points])
    
    def sample_grid_outside(self, nb_points = 1000, procentage = 0.1):
        """ Similar to sample_grid, however this samples from a extension of the
            image domain where procentage * image domain is added around the
            original image domain
        """
        x_ext = procentage * (self.maxbound[0] - self.minbound[0])
        y_ext = procentage * (self.maxbound[1] - self.minbound[1])
        return create_grid([self.minbound[0] - x_ext, self.minbound[1] - y_ext],
                           [self.maxbound[0] + x_ext, self.maxbound[1] + y_ext], 
                           [nb_points, nb_points])
        
    def sample_grid_image(self, imagesize):
        """ Similar to sample_grid, just with varing sample size in x,y direction """
        return create_grid(self.minbound, self.maxbound, imagesize)
    
    def visualize_tessalation(self, outside = False):
        """ Visualize the tessalation. Outside determine if only the tessalation
            is evaluated on the image domain (False) or also outside of the domain
        """
        nb_points = 500
        if outside:
            points = self.sample_grid_outside(nb_points, 0.2)
        else:
            points = self.sample_grid(nb_points)
        idx = np.zeros(points.shape[1], dtype = np.int)
        count = 0
        for p in points.T:
            idx[count] = self.find_cell_idx(p)
            count += 1
        idx_disp = np.reshape(idx, (nb_points, nb_points))
        plt.imshow(idx_disp)
        plt.axis('off')
        plt.colorbar()
        plt.title('Tessalation [{}, {}]'.format(self.ncx, self.ncy), fontsize = 25)
    
    
    def sample_theta_without_prior(self, n = 1):
        """ Sample a random parameterization vector theta from a multivariate
            normal distribution with zero mean and 0.5*I covariance matrix """
        theta = np.random.multivariate_normal(np.zeros(self.d), np.identity(self.d), n)
        return theta
    
    def sample_theta_with_prior(self, n = 1):
        # Extract centers
        centers = np.mean(self.cells_verts[:,:,:2], axis=1)
        
        # Compute distance between centers
        norms = np.linalg.norm(centers,axis=1)**2
        dist_c = norms[:,np.newaxis] + norms[np.newaxis,:] - 2*np.dot(centers, centers.T)
        
        # Construct covariance matrix on original parameter space
        cov_avees = np.zeros((6*self.nC, 6*self.nC))
        for i in range(self.nC):
            for j in range(self.nC):
                    cov_avees[6*i:6*i+6, 6*j:6*j+6] = np.diag(np.repeat(np.exp(-dist_c[i,j]),6))
        
        # Calculate covariance matrix for theta space
        cov_theta = np.dot(self.B.T, np.dot(cov_avees, self.B))
        
        # Sample values
        theta = np.random.multivariate_normal(np.zeros(self.d), cov_theta, n)
        return theta
    
    def calc_v(self, theta, points):
        """ For a given parametrization theta and a matrix of 2D points, calculate 
            the corresponding velocity field at all points
        """
        # Construct affine transformations
        Avees = self.theta2Avees(theta)
        As = self.Avees2As(Avees)
        v = np.zeros((points.shape[1],2))
        
        # For all points, find the cell index and calculate velocity
        count = 0
        for p in points.T:
            p = np.append(p,1)
            idx = self.find_cell_idx(p)
            v[count] = np.dot(As[idx], p)
            count += 1
        
        return v
    
    def calcTrans(self, theta, points):
        nP = points.shape[1]
        nstep = 50
        dT = 1.0/nstep
        
        # Transform points to homogeneuos coordinates
        newpoints = np.concatenate((points, np.ones((1, nP))), axis=0)
        
        # Construct affine transformations
        Avees = self.theta2Avees(theta)
        As = self.Avees2As(Avees)
        Asquare = self.As2squareAs(As)
        
        # Construct mappings
        Trels = np.array([scipy_expm(dT*Asquare[i]) for i in range(self.nC)])
        
        # Transform points using the mappings
        for i in range(nP):
            for t in range(nstep):
                idx = self.find_cell_idx(newpoints[:,i])
                newpoints[:,i] = Trels[idx].dot(newpoints[:,i])
        
        return newpoints[:2,:]
    
    def visualize_vectorfield(self, theta):
        """ Visualize the velocity field as two heatmaps """
        nb_points = 500
        points = self.sample_grid(nb_points)
        v = self.calc_v(theta, points)
        vmax = np.max(np.abs(v))
        vmin = -vmax
        vx = v[:,0].reshape((nb_points, nb_points))
        vy = v[:,1].reshape((nb_points, nb_points))
        
        plt.figure()
        plt.subplot(121)
        plt.imshow(vx.copy(), vmin = vmin, vmax = vmax, interpolation="Nearest")
        plt.axis('off')
        plt.title('v_x', fontsize = 25)
        plt.colorbar()
        
        plt.subplot(122)
        plt.imshow(vy.copy(), vmin = vmin, vmax = vmax, interpolation="Nearest")
        plt.axis('off')
        plt.title('v_y', fontsize = 25)
        plt.colorbar()
        
    def visualize_vectorfield_arrow(self, theta):
        """ Visualize the velocity field as a single arrow plot """
        nb_points = 20
        points = self.sample_grid(nb_points)
        v = self.calc_v(theta, points)
        plt.figure()
        plt.quiver(points[0,:], points[1,:], v[:,0], v[:,1], scale=5)
        plt.axis([-1.5, 1.5, -1.5, 1.5])
        plt.axis('equal')
        plt.title('Velocity field')

#%%
if __name__ == '__main__':
    # Create/load basis
    s = setup_CPAB_transformer(2, 2, 
                               valid_outside=True, 
                               zero_trace=False,
                               zero_boundary=False,
                               override=False)
    
    # Show tessalation
    s.visualize_tessalation(outside=True)
    
    # Sample random transformation
    theta = s.sample_theta_without_prior(1)
    theta = np.reshape(theta, (-1, 1))
    
    # Show velocity field
    s.visualize_vectorfield(theta)
    s.visualize_vectorfield_arrow(theta)
    