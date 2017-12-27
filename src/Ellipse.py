# Plot 1-Sigma Ellipses
import numpy

def ellipse(mean, covariance, sigma=4.605):
    # Plots a confidence ellipse around 2D Gaussians.
    # Converted to Julia from the Matlab function:
    # http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/

    # Calculate the eigenvectors and eigenvalues
    eigenval, eigenvec = numpy.linalg.eig(covariance);

    if eigenval[0] > eigenval[1]:
        largest_eigenval = eigenval[0]
        largest_eigenvec = eigenvec[:, 0]
        smallest_eigenval = eigenval[1]
        smallest_eigenvec = eigenvec[:,1]
    else:
        largest_eigenval = eigenval[1]
        largest_eigenvec = eigenvec[:, 1]
        smallest_eigenval = eigenval[0]
        smallest_eigenvec = eigenvec[:,0]
    

    angle = numpy.arctan2(largest_eigenvec[1], largest_eigenvec[0])

    if angle < 0.0:
        angle = angle + 2*numpy.pi
    

    chisquare_val = numpy.sqrt(sigma)

    theta_grid = numpy.linspace(0.0,2.0*numpy.pi)
    phi = angle
    X0=mean[0]
    Y0=mean[1]
    a=chisquare_val*numpy.sqrt(largest_eigenval)
    b=chisquare_val*numpy.sqrt(smallest_eigenval)

    # the ellipse in x and y coordinates
    ellipse_x_r  = a*numpy.cos( theta_grid )
    ellipse_y_r  = b*numpy.sin( theta_grid )

    #Define a rotation matrix
    R = [[numpy.cos(phi), numpy.sin(phi)], [-numpy.sin(phi), numpy.cos(phi)]]

    # rotate the ellipse to some angle phi
    r_ellipse = numpy.zeros([2, len(ellipse_x_r)])
    for k in range(len(ellipse_x_r)):
        r_ellipse[:, k] = numpy.matmul(R, [ellipse_x_r[k], ellipse_y_r[k]]) + mean
    
    return r_ellipse[0,:][:], r_ellipse[1,:][:]
    

 
