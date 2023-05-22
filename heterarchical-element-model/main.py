import numpy as np
import matplotlib.pyplot as plt
import json5

def swap(i, j, arr):
    new_arr = arr.copy()
    for a in new_arr:
        a[i[0],i[1],i[2],:], a[j[0],j[1],j[2],:] = a[j[0],j[1],j[2],:], a[i[0],i[1],i[2],:]
    return new_arr

def move_overhangs(P,s,x,v,a):
    for i in range(1,P["grid"]["nx"]-1): # TODO: IGNORING ALL BOUNDARIES FOR NOW
        for j in range(1,P["grid"]["ny"]-1): # TODO: IGNORING ALL BOUNDARIES FOR NOW
            for k in range(P["grid"]["nm"]): # TODO: IGNORING ALL BOUNDARIES FOR NOW
                if ~np.isnan(s[i,j,k]):
                    # move anything that needs to be moved
                    if x[i,j,k,0] > P["grid"]["dx"][0]/2.: # RIGHT
                        if np.isnan(s[i+1,j,k]):
                            x[i,j,k,0], x[i+1,j,k,0] = 0, x[i,j,k,0]-P["grid"]["dx"][0]
                            # s,v = swap([i, j, k], [i+1, j, k], [s,v])
                            s[i,j,k],   s[i+1,j,k]   = s[i+1,j,k],   s[i,j,k]
                            v[i,j,k,:], v[i+1,j,k,:] = v[i+1,j,k,:], v[i,j,k,:]
                            # print(v[i,j,k,0])
                        else:
                            x[i,j,k,0] = P["grid"]["dx"][0]/2.
                            v[i,j,k,0] = 0
                    elif x[i,j,k,0] < -P["grid"]["dx"][0]/2.: # LEFT
                        if np.isnan(s[i-1,j,k]):
                            x[i,j,k,0], x[i-1,j,k,0] = 0, x[i,j,k,0]+P["grid"]["dx"][0]
                            # s,v = swap([i, j, k], [i-1, j, k], [s,v])
                            s[i,j,k],   s[i-1,j,k]   = s[i-1,j,k],   s[i,j,k]
                            v[i,j,k,:], v[i-1,j,k,:] = v[i-1,j,k,:], v[i,j,k,:]
                            # print(v[i,j,k,0])
                        else:
                            x[i,j,k,0] = - P["grid"]["dx"][0]/2.
                            v[i,j,k,0] = 0
                    elif x[i,j,k,1] > P["grid"]["dx"][1]/2.: # UP
                        if np.isnan(s[i,j+1,k]):
                            x[i,j,k,1], x[i,j+1,k,1] = 0, x[i,j,k,1]-P["grid"]["dx"][1]
                            # s,v = swap([i, j, k], [i, j+1, k], [s,v])
                            s[i,j,k],   s[i,j+1,k]   = s[i,j+1,k],   s[i,j,k]
                            v[i,j,k,:], v[i,j+1,k,:] = v[i,j+1,k,:], v[i,j,k,:]
                            # print(v[i,j,k,1])
                        else:
                            x[i,j,k,1] = P["grid"]["dx"][1]/2.
                            v[i,j,k,1] = 0
                    elif x[i,j,k,1] < -P["grid"]["dx"][1]/2.: # DOWN
                        if np.isnan(s[i,j-1,k]):
                            x[i,j,k,1], x[i,j-1,k,1] = 0, x[i,j,k,1]+P["grid"]["dx"][1]
                            # print(v[i,j,k,1])
                            # s,v = swap([i, j, k], [i, j-1, k], [s,v])
                            # print(v[i,j-1,k,1])
                            s[i,j,k],   s[i,j-1,k]   = s[i,j-1,k],   s[i,j,k]
                            v[i,j,k,:], v[i,j-1,k,:] = v[i,j-1,k,:], v[i,j,k,:]
                        else:
                            x[i,j,k,1] = - P["grid"]["dx"][1]/2.
                            v[i,j,k,1] = 0
    return s, x, v, a

def update_forces(P,s,x,v):
    F = np.zeros_like(x) # forces
    for i in range(1,P["grid"]["nx"]-1): # TODO: IGNORING ALL BOUNDARIES FOR NOW
        for j in range(1,P["grid"]["ny"]-1): # TODO: IGNORING ALL BOUNDARIES FOR NOW
            for k in range(P["grid"]["nm"]): # TODO: IGNORING ALL BOUNDARIES FOR NOW                            
                if ~np.isnan(s[i,j,k]):
                
                    # body forces
                    F[i,j,k,0] +=  P["mat"]["m_cell"]*P["gravity"]["mag"]*np.sin(np.deg2rad(P["gravity"]["theta"]))
                    F[i,j,k,1] += -P["mat"]["m_cell"]*P["gravity"]["mag"]*np.cos(np.deg2rad(P["gravity"]["theta"]))

                    # elastic forces
                    for ax in [0,1]:
                        for dir in [-1,1]:
                            target = [i,j,k]
                            target[ax] += dir
                            if ~np.isnan(s[target[0],target[1],target[2]]):
                                E_star = P["mat"]["E"] # TODO: If materials have different stiffness, calculate it properly
                                R_e = (s[i,j,k] + s[target[0],target[1],target[2]])/(s[i,j,k]*s[target[0],target[1],target[2]]) # equivalent radius of contact particles —-- what does this mean in this context and is this even something worth calculating??
                                delta = np.abs(x[i,j,k,ax] - x[target[0],target[1],target[2],ax])# - dx[dir]
                                if delta > 0:
                                    this_force = dir*np.sqrt(16/9)*np.sqrt(E_star*R_e)*(delta)**(1.5)
                                    F[i,j,k,dir] -= this_force
                                    F[target[0],target[1],target[2],dir] += this_force
    return F

def main(input_filename):
    P = json5.load(open(input_filename, 'r'))

    if P["IC"]["type"] == "random":
        s = np.random.rand(P["grid"]["nx"],P["grid"]["ny"],P["grid"]["nm"])
        s[ np.random.rand(*s.shape) > P["IC"]["fill"] ] = np.nan

    if P["IC"]["boundary"]:
        s[:,0,:] = P["IC"]["max"]
        s[:,-1,:] = P["IC"]["max"]
        s[0,:,:] = P["IC"]["max"]
        s[-1,:,:] = P["IC"]["max"]

    nt = int(P["time"]["t_max"] / P["time"]["dt"])
    P["grid"]["dx"] = [P["grid"]["Lx"] / P["grid"]["nx"], P["grid"]["Ly"] / P["grid"]["ny"]]
    x_ref = [np.linspace(0,P["grid"]["Lx"],P["grid"]["nx"]), np.linspace(0,P["grid"]["Ly"],P["grid"]["ny"])]
    X, Y = np.meshgrid(x_ref[0], x_ref[1], indexing='ij')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    X_c = (X[1:,1:] + X[:-1,1:])/2.
    Y_c = (Y[1:,1:] + Y[1:,:-1])/2.

    empty_c = np.nan*np.ones_like(X_c)
    empty = np.nan*np.ones_like(X)

    V_tot = P["grid"]["Lx"]*P["grid"]["Ly"]*P["grid"]["Lz"] # total volume of entire grid
    V_cell = V_tot/(P["grid"]["nx"]*P["grid"]["ny"]*P["grid"]["nm"]) # volume of an individual cell
    P["mat"]["m_cell"] = V_cell * P["mat"]["rho"] # mass of an individual cell  

    x = np.zeros((P["grid"]["nx"],P["grid"]["ny"],P["grid"]["nm"],2)) # position of particle centres relative to centre of the cells
    v = np.zeros_like(x) # velocity of particle centres
    a = np.zeros_like(x) # acceleration of particle centres

    for tstep in range(nt):
        s, x, v, a = move_overhangs(P,s,x,v,a)
        F = update_forces(P,s,x,v)
        
        
        a = F / P["mat"]["m_cell"]
        v += P["time"]["dt"] * a
        x += P["time"]["dt"] * v
        # print(v)

        if P["debug"]:
            plt.ion()
            plt.clf()
            plt.title(f"t = {tstep*P['time']['dt']:0.3f}")
            # plt.imshow(s[:,:,0])
            plt.pcolormesh(X_c,Y_c,empty_c,edgecolors='lightgray')
            plt.pcolormesh(X,Y,empty,edgecolors='k')
            plt.scatter(X_flat+x[...,0].flatten(),Y_flat+x[...,1].flatten(),c=s.flatten(),cmap='viridis',vmin=0,vmax=P["IC"]["max"])
            plt.quiver(X+x[:,:,0,0],Y+x[:,:,0,1],F[:,:,0,0],F[:,:,0,1])
            plt.colorbar()
            plt.pause(1e-2)

if __name__ == '__main__':
    input_filename = '../json5/test.json5'
    main(input_filename)