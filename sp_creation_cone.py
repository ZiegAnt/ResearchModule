import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import norm
import pylab as plt
import random
import pandas as pd 
#%matplotlib qt

plt.close('all')


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

def truncated_cone(p0, p1, R0, R1):
    """
    Based on https://stackoverflow.com/a/39823124/190597 (astrokeat)
    """
    # vector in direction of axis
    v = p1 - p0
    # find magnitude of vector
    mag = norm(v)
    # unit vector in direction of axis
    v = v / mag
    # make some vector not in the same direction as v
    not_v = np.array([1, 1, 0])
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])
    # make vector perpendicular to v
    n1 = np.cross(v, not_v)
    # print n1,'\t',norm(n1)
    # normalize n1
    n1 /= norm(n1)
    # make unit vector perpendicular to v and n1
    n2 = np.cross(v, n1)
    # surface ranges over t from 0 to length of axis and 0 to 2*pi
    n = 100
    t = np.linspace(0, mag, n)
    theta = np.linspace(0, 2 * np.pi, n)
    # use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)
    R = np.linspace(R0, R1, n)
    # generate coordinates for surface
    X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    ax.plot_surface(X, Y, Z, color='blue', linewidth=0, antialiased=False)
    return X, Y, Z

###############################################################################
# Important parameters
# cone parameters
A0 = np.array([50, 50, 45]) # upper axis point
A1 = np.array([50, 50, 5]) # lower axis point
R_upper = 30 # radius at upper end
R_lower = 20 # radius at lower end

# model dimensions
x_min = 0
x_max = 150
y_min = 0
y_max = 150
z_min = 0
z_max = 60

# Noise parameters
noise_mag_cone = 1 # in + and - direction

# surface_points parameters
n_sp = 3 # number of surface points in 1 dimension

# settings strongly consolidated sandstone formation
css_z = 5
noise_mag_css = 1

# settings sandstone formation
ss_z = 35
noise_mag_ss = 1

# settings soil formation
soil_z = 44
noise_mag_soil = 0.5

###############################################################################
# Generating Cone structure
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)
[x,y,z] = truncated_cone(A0, A1, R_upper, R_lower)
plt.show()

###############################################################################
# Adding random Noise to cone geometry
dim = np.size(x,0)

noise_cone_x = (np.random.random((dim, dim))-0.5)*2*noise_mag_cone
noise_cone_y = (np.random.random((dim, dim))-0.5)*2*noise_mag_cone

x_n = x + noise_cone_x
y_n = y + noise_cone_y

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(x_n, y_n, z, color='blue', linewidth=0, antialiased=False)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)
plt.show()

###############################################################################
# Flatten for conversion into CSV 
sp_diatreme_x = x_n.flatten().tolist()
sp_diatreme_y = y_n.flatten().tolist()
sp_diatreme_z = z.flatten().tolist()

# Add axis endpoints 
sp_diatreme_x = sp_diatreme_x[::24] + [A0[0],A1[0]]
sp_diatreme_y = sp_diatreme_y[::24] + [A0[1],A1[1]]
sp_diatreme_z = sp_diatreme_z[::24] + [A0[2],A1[2]]
###############################################################################
# surface_points of strongly consolidated Sandstone Layer (sp_css)
x_arr = np.linspace(x_min, x_max, n_sp) 
y_arr = np.linspace(y_min, y_max, n_sp)
sp_css_x = []
sp_css_y = []
sp_css_z = []

for i in x_arr:
    for j in y_arr:
      sp_css_x.append(i)
      sp_css_y.append(j)
      sp_css_z.append(css_z+random.uniform(-noise_mag_css, noise_mag_css))
      
ax.scatter(sp_css_x,sp_css_y,sp_css_z)
        
###############################################################################
# surface_points of Sandstone Layer (sp_ss)
sp_ss_x = []
sp_ss_y = []
sp_ss_z = []

for i in x_arr:
    for j in y_arr:
      sp_ss_x.append(i)
      sp_ss_y.append(j)
      sp_ss_z.append(ss_z)
      
ax.scatter(sp_ss_x,sp_ss_y,sp_ss_z)

###############################################################################
# surface_points of Soil Layer (sp_soil)
sp_soil_x = []
sp_soil_y = []
sp_soil_z = []

for i in x_arr:
    for j in y_arr:
      sp_soil_x.append(i)
      sp_soil_y.append(j)
      sp_soil_z.append(soil_z+random.uniform(-noise_mag_soil, noise_mag_soil))
      
ax.scatter(sp_soil_x,sp_soil_y,sp_soil_z)
        

###############################################################################
# merging all surface_points into panda dataframe
# Adding all coordinate lists into 1 starting with oldest and finishing with youngest
# i.e., css + ss + diatreme + soil

x_coordinate = sp_css_x + sp_ss_x + sp_diatreme_x + sp_soil_x
y_coordinate = sp_css_y + sp_ss_y + sp_diatreme_y + sp_soil_y
z_coordinate = sp_css_z + sp_ss_z + sp_diatreme_z + sp_soil_z
formation_list = ['CSS' for i in sp_css_x] + ['SS' for i in sp_ss_x] + ['Diatreme' for i in sp_diatreme_x] + ['Soil' for i in sp_soil_x]

sp_df_dict = {'X':x_coordinate, 'Y':y_coordinate, 'Z':z_coordinate,'formation':formation_list}
sp_df = pd.DataFrame(data=sp_df_dict)
#sp_df.to_csv('RM_surface_points.csv')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(sp_css_x,sp_css_y,sp_css_z)
ax.scatter(sp_ss_x,sp_ss_y,sp_ss_z)
ax.scatter(sp_diatreme_x[::10],sp_diatreme_y[::10],sp_diatreme_z[::10])
ax.scatter(sp_soil_x,sp_soil_y,sp_soil_z)
plt.show()


x_coo_ori = [sp_css_x[0],sp_soil_x[0],A0[0],A1[0],sp_diatreme_x[125]]
y_coo_ori = [sp_css_y[0],sp_soil_y[0],A0[1],A1[1],sp_diatreme_y[125]]
z_coo_ori = [sp_css_z[0],sp_soil_z[0],A0[2],A1[2],sp_diatreme_z[125]]
az_ori = [0,0,0,0,0]
dip_ari = [0,0,0,0,90]
pol_ori = [1,1,1,1,1]
formation_ori = ['CSS','Soil','Diatreme','Diatreme','Diatreme']

ori_df_dict = {'X':x_coo_ori, 'Y':y_coo_ori, 'Z':z_coo_ori,'azimuth':az_ori ,'dip':dip_ari ,'polarity':pol_ori ,'formation':formation_ori}
ori_df = pd.DataFrame(data=ori_df_dict)
#ori_df.to_csv('RM_orientations.csv')











