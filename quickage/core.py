
##################
## MAIN CODE ###
##############

#imports
import os
import astropy.coordinates as astro_coord
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import G

#gala imports
import gala.coordinates as gc
import gala.potential as gp
import gala.dynamics as gd
from gala.units import galactic




#set paths
CODE_PATH = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
DATA_FOLDER= CODE_PATH + '/data/'



#define a coordinate frame
#coordinate frame
_ = astro_coord.galactocentric_frame_defaults.set('v4.0')
v_sun = astro_coord.CartesianDifferential([11.1, 220 + 24.0, 7.25]*u.km/u.s)
galcen_frame =astro_coord.Galactocentric(galcen_distance=8.2*u.kpc,
                                    galcen_v_sun=v_sun)

#potential
pot=gp.MilkyWayPotential()
H = gp.Hamiltonian(pot)


def load_samples():
	"""
	Read bensby et al, luck et al, cassagrande et al., spocs
	"""

	bensby=ascii.read(DATA_FOLDER+'/bensby_30pc_gaia1.txt', \
	                        names ='ra              de    \
	                                [Fe/H]       age1     l_age1     u_age1       \
	                                ra_gaia              de_gaia        plx       eplx    \
	                                pmra      epmra       pmde      epmde      \
	                                rv        erv'.split()).to_pandas()

	luck=ascii.read(DATA_FOLDER+'/luck_30pc_gaia1.txt', \
	               names='ra              de     [Fe/H]       age1     l_age1    \
	               u_age1         ra_gaia         de_gaia        plx       eplx      \
	               pmra      epmra       pmde      epmde         rv   \
	               erv '.split()).to_pandas()

	casgr=ascii.read(DATA_FOLDER+'/casagrande_30pc_gaia1.txt', \
	                 names='ra              de     [Fe/H]      ageMP     age16P     \
	                 age84P              ra_gaia              de_gaia        \
	                 plx       eplx       pmra      epmra       pmde      \
	                 epmde         rv        erv'.split()).to_pandas()
	spocs=ascii.read(DATA_FOLDER+'/spocs_30pc_gaia1.txt', \
	                 names=' ra              de     [Fe/H]       age1     l_age1     u_age1          \
	                 ra_gaia              de_gaia        plx       eplx       pmra   \
	                 epmra       pmde      epmde         rv        erv '.split()).to_pandas()

	comb=pd.concat([bensby, luck, spocs, casgr.rename(columns={'ageMP':'age1'})])
	return comb
	                 

def get_phase_space(ra, dec, pmracosdec, pmdec, distance, rv ):
	#get phase space position of an object in our coordinate frame
    #ra, dec in degree
    #proper motions in mas/yr
    #distance in pc
    #rv in km/s
    coord=SkyCoord(ra=ra*u.degree, dec=dec*u.degree,  
               pm_ra_cosdec= pmracosdec *u.mas/u.yr, pm_dec=pmdec*u.mas/u.yr, \
               distance=distance*u.pc, 
              radial_velocity= rv*u.km/u.s)
    #phase space position
    pos=gd.PhaseSpacePosition(coord.transform_to(galcen_frame).cartesian)
    
    return coord, pos

def compute_actions(pos, plot_all_orbit=False, alpha=1., print_pericenter=False):

	"""
	Purpose compute actions based on the orbit of a star

	"""
    orbit=gp.Hamiltonian(pot).integrate_orbit(pos, dt=3*u.Myr, t1=0*u.Myr, \
                                              t2=2.5*u.Gyr)
    #plot 
    orbit_to_plot=orbit[:,0]
    if plot_all_orbit: orbit_to_plot=orbit
    oplot=orbit_to_plot.cylindrical.plot( components=['rho', 'z', 'v_z'],  \
                                      units=[u.pc, u.pc, u.km/u.s] ,alpha=alpha, c='#0074D9')
    #documentation: http://gala.adrian.pw/en/latest/dynamics/actionangle.html
    toy_potential = gd.fit_isochrone(orbit[:,0])
    print (toy_potential)
    print (np.shape(orbit.z))
    result = [gd.find_actions(orbit[:,idx], N_max=10, toy_potential=toy_potential) \
              for idx in tqdm(np.arange(np.shape(orbit)[-1]))]
    if  print_pericenter:
        apos=[orbit[:,idx].apocenter() for idx in tqdm(np.arange(np.shape(orbit)[-1]))]
        peris=[orbit[:,idx].pericenter() for idx in tqdm(np.arange(np.shape(orbit)[-1]))]
        print ('apocenter --- {} +/- {}'.format(np.nanmedian(u.Quantity(apos)),\
                                                np.nanstd(u.Quantity(apos))))
        print ('pericenter --- {} +/- {}'.format(np.nanmedian(u.Quantity(peris)),\
                                                 np.nanstd(u.Quantity(peris))))
    return pd.DataFrame.from_records(result), oplot


 def estimate_age(source_coord, source_metal, use_jz=False, plot=False):

 	#source_coord must be a dictionary with the following keywords
 	#ra: ra in degree
 	#dec: dec in degree
 	#pmra: tuple in mas/yr (pmra, pmra_unc)
 	#pmdec: tuple in mas/yr (pmdec, pmdec_unc)
 	#distance: must a distance in pc (dist, dist_unc)
 	#rv: tuple in km/s (rv, rv_unc)

 	#source_metal: a tuple of [Fe/H] and uncertainty

 	#use_jz: keyword to use vertical action as additional constraints

 	#returns: age and posterior plots

 	Scoord={'ra':source_coord['ra'], \
              'dec': source_coord['dec'],\
              'pmra':np.random.normal(source_coord['pmra'][0],source_coord['pmra'][1], 1000),
              'pmdec':np.random.normal(source_coord['pmdec'][0],source_coord['pmdec'][1], 1000),
              'distance':np.random.normal(source_coord['distance'][0],source_coord['distance'][1], 1000),
              'rv': np.random.normal(source_coord['rv'][0],source_coord['rv'][1], 1000)}

    #read in data
    data= load_samples()
    data_coord, data_pos=get_phase_space(data.ra_gaia.values, data.de_gaia.values,\
                             data.pmracosdec.values, \
                data.pmde.values, 1000/data.plx.values, data.rv.values )

    source_coord, source_pos=get_phase_space(	Scoord['ra'], 	Scoord['dec'],\
                       	Scoord['pmra']*np.cos(	Scoord['dec']*u.degree), \
                       	Scoord['pmdec'], 	Scoord['distance'], Scoord['rv'])

   	#if use jz
   	if use_jz:
   		#compute stellar orbits, add in option to load pre-computed orbits
   		data_res=compute_actions(data_pos, plot_all_orbit=False)
   		data_actions=np.vstack(data_res[0]['actions'].apply(lambda x: np.array(x)).values)

   		#don't need
		#data_angles=np.vstack(data_res[0]['angles'].apply(lambda x: np.array(x)).values)
		#data_freqs=np.vstack(data_res[0]['freqs'].apply(lambda x: np.array(x)).values)


   		data['Jr']=data_actions[:,0]*1000 #units (kpc$^2$/Gyr)
		data['Jphi']=data_actions[:,1]*1000 
		data['Jz']=data_actions[:,2]*1000
		data['vtot']=((data_coord.transform_to(galcen_frame).v_x**2+
		                data_coord.transform_to(galcen_frame).v_y**2+
		                data_coord.transform_to(galcen_frame).v_z**2)**0.5).value
		data['v_x']=data_coord.transform_to(galcen_frame).v_x.value
		data['v_y']=data_coord.transform_to(galcen_frame).v_y.value
		data['v_z']=data_coord.transform_to(galcen_frame).v_z.value

		#idem for the source
   		source_res=compute_actions(source_pos, plot_all_orbit=True, alpha=1.)
   		source_actions=np.vstack(source_res[0]['actions'].apply(lambda x: np.array(x)).values)
   		#forget about angles and other things


