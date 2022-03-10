
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
#from astropy.constants import G

#gala imports
#import gala.coordinates as gc
import gala.potential as gp
import gala.dynamics as gd
#from gala.units import galactic
import pandas as pd
from astropy.io import ascii
import matplotlib as mpl
from .plots import seaborn 

#set paths
CODE_PATH = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
DATA_FOLDER= CODE_PATH + '/data/'

#define a coordinate frame
#not sure where this system came from ? Sarah's values 
#I have used this system to compute Jz so can't change it really
_ = astro_coord.galactocentric_frame_defaults.set('v4.0')
v_sun = astro_coord.CartesianDifferential([11.1, 220 + 24.0, 7.25]*u.km/u.s)
galcen_frame =astro_coord.Galactocentric(galcen_distance=8.2*u.kpc,
                                    galcen_v_sun=v_sun)


#potential
pot=gp.MilkyWayPotential()
H = gp.Hamiltonian(pot)


def load_schneider_samples(use_jz=False):
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
            
    luck['pmracosdec']= luck.pmra*np.cos(luck.de*u.degree)
    bensby['pmracosdec']= bensby.pmra*np.cos(bensby.de*u.degree)
    spocs['pmracosdec']= spocs.pmra*np.cos(spocs.de*u.degree)
    casgr['pmracosdec']= casgr.pmra*np.cos(casgr.de*u.degree)
    
    data=pd.concat([bensby, luck, spocs, casgr]).rename(columns={'ageBP':'age1'})
    
    data_coord, data_pos=get_phase_space(data.ra_gaia.values, data.de_gaia.values, data.pmracosdec.values, data.pmde.values, 1000/data.plx.values, data.rv.values )
    
    data['vtot']=((data_coord.transform_to(galcen_frame).v_x**2+data_coord.transform_to(galcen_frame).v_y**2+data_coord.transform_to(galcen_frame).v_z**2)**0.5).value


    data['v_x']=data_coord.transform_to(galcen_frame).v_x.value
    data['v_y']=data_coord.transform_to(galcen_frame).v_y.value
    data['v_z']=data_coord.transform_to(galcen_frame).v_z.value

    #compute actions if necessary
    if use_jz:
        data_res=compute_actions(data_pos, plot_all_orbit=False)
        data_actions=np.vstack(data_res[0]['actions'].apply(lambda x: np.array(x)).values)
        #don't need
        #data_angles=np.vstack(data_res[0]['angles'].apply(lambda x: np.array(x)).values)
        #data_freqs=np.vstack(data_res[0]['freqs'].apply(lambda x: np.array(x)).values)
        data['Jr']=data_actions[:,0]#*1000 #units (kpc$^2$/Myr)
        data['Jphi']=data_actions[:,1]#*1000 
        data['Jz']=data_actions[:,2]#*1000
        
    return data  

#def load_galah_sample():
#    from astropy.table import Table
#    merged_file='/volumes/LaCie/galah_merged.fits'
#    galah= Table.read(merged_file).to_pandas()
#    galah['vtot']= (galah['vx_gala (km/s)']**2+galah['vy_gala (km/s)']**2+galah['vz_gala (km/s)']**2)**0.5
#    return galah.rename(columns={'vx_gala (km/s)': 'v_x', \
#                      'vy_gala (km/s)': 'v_y',  \
#                      'vz_gala (km/s)': 'v_z',
#                     'fe_h': '[Fe/H]', 
#                    'Jz (kpc2/Myr)': 'Jz',
#                    'age_bstep': 'age1'})

def load_galah_sample():
    return pd.read_csv(DATA_FOLDER+'/galah_lite.csv.gz')

def get_phase_space(ra, dec, pmracosdec, pmdec, distance, rv ):
    """
	get phase space position of an object in our coordinate frame
    ra, dec in degree
    proper motions in mas/yr
    distance in pc
    rv in km/s
    """
    coord=astro_coord.SkyCoord(ra=ra*u.degree, dec=dec*u.degree,  
               pm_ra_cosdec= pmracosdec *u.mas/u.yr, pm_dec=pmdec*u.mas/u.yr, \
               distance=distance*u.pc, 
              radial_velocity= rv*u.km/u.s)
    #phase space position
    pos=gd.PhaseSpacePosition(coord.transform_to(galcen_frame).cartesian)
    
    return coord, pos

def compute_actions(pos, plot_all_orbit=False, alpha=1., print_pericenter=False):
    """
	Purpose compute actions based on the orbit of a star

	These default settings are used for all the samples, 
	they sample at least one orbit of the majority of the stars
    
    These are the default paramaters that I used to compute actions on the main samples

	"""
    orbit=gp.Hamiltonian(pot).integrate_orbit(pos, dt=3*u.Myr, t1=0*u.Myr, \
                                              t2=2.5*u.Gyr)
    #plot 
    orbit_to_plot=orbit[:,0]
    oplot=None
    if plot_all_orbit: 
        orbit_to_plot=orbit
        oplot=orbit_to_plot.cylindrical.plot( components=['rho', 'z', 'v_z'],  units=[u.pc, u.pc, u.km/u.s] ,alpha=alpha, c='#0074D9')
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

def estimate_age(source_coord, source_metal, nsigma=3, use_jz=False, plot_kde=False, data_set='galah', plot=False, file_plot=None, file_data=None, export_data=False):
    """
 	source_coord must be a dictionary with the following keywords
 	ra: ra in degree
 	dec: dec in degree
 	pmra: tuple in mas/yr (pmra, pmra_unc)
 	pmdec: tuple in mas/yr (pmdec, pmdec_unc)
 	distance: must a distance in pc (dist, dist_unc)
 	rv: tuple in km/s (rv, rv_unc)

 	source_metal: a tuple of [Fe/H] and uncertainty

 	use_jz: keyword to use vertical action as additional constraints

 	returns: age and posterior plots
 	uncertainties must be reasonable, not zero
 	"""
    
    #compute source coordinate 
    Scoord={'ra':source_coord['ra'], \
              'dec': source_coord['dec'],\
              'pmra':np.random.normal(source_coord['pmra'][0],source_coord['pmra'][1], 1000),
              'pmdec':np.random.normal(source_coord['pmdec'][0],source_coord['pmdec'][1], 1000),
              'distance':np.random.normal(source_coord['distance'][0],source_coord['distance'][1], 1000),
              'rv': np.random.normal(source_coord['rv'][0],source_coord['rv'][1], 1000)}

    source_coord, source_pos=get_phase_space(Scoord['ra'], 	Scoord['dec'],\
                       	Scoord['pmra']*np.cos(	Scoord['dec']*u.degree), \
                       	Scoord['pmdec'], 	Scoord['distance'], Scoord['rv'])


    if data_set=='galah': 
        data= load_galah_sample()

    if data_set=='schneider': 
        data=load_schneider_samples(use_jz=use_jz)
        
    #if use jz
    total_cut=[]

    if use_jz:

        source_res=compute_actions(source_pos, plot_all_orbit=False, alpha=1.)
        source_actions=np.vstack(source_res[0]['actions'].apply(lambda x: np.array(x)).values)
        mean_source_jz= np.nanmedian(source_actions[:,-1])
        std_source_jz= np.nanstd(source_actions[:,-1])
   		#forget about angles and frequencies
        print (mean_source_jz, std_source_jz)

   		#compute boolean vertical_actions within uncertainties
        jz_cut= np.logical_and(data.Jz < mean_source_jz+ nsigma* std_source_jz, \
        	data.Jz > mean_source_jz- nsigma* std_source_jz)
        total_cut.append(jz_cut)

   	#kinematics, metallicity cuts

   	#total velocity of the source
    source_total_v=(source_coord.transform_to(galcen_frame).v_x**2+
            source_coord.transform_to(galcen_frame).v_y**2+
                source_coord.transform_to(galcen_frame).v_z**2)**0.5

    vr= ((source_coord.transform_to(galcen_frame).v_x**2+
                source_coord.transform_to(galcen_frame).v_y**2)**0.5).value
                 
    vz=(source_coord.transform_to(galcen_frame).v_z).value


   	#compute only kinematics within the velocity ellipse of the star
    kinematic_cut=data.vtot < (np.nanmedian(source_total_v).value)# + nsigma*np.nanmedian(source_total_v).value)
    total_cut.append(kinematic_cut)

	#metallicity within n-sigma
    metallicity_cut= np.logical_and(data['[Fe/H]'] > source_metal[0]-nsigma*source_metal[-1],
									data['[Fe/H]'] < source_metal[0]+nsigma*source_metal[-1])
    total_cut.append(metallicity_cut)

	#selection
    selection= np.logical_and.reduce(total_cut)
    nans= ~np.isnan(data.age1)
    MEDIAN_AGE=np.nanmedian(data.age1[np.logical_and(selection, nans)])
    STD_AGE=[np.percentile(data.age1[selection].dropna(), 16), \
	         np.percentile(data.age1[selection].dropna(), 84)]
    if plot:
        fig, ax=plt.subplots( figsize=(8, 6))
        if not  plot_kde:
            _=ax.hist(data.age1, histtype='step', bins='auto', lw=3, density=True, \
                      linestyle='--', color='r', label='Full sample')
            _=ax.hist(data.age1[selection],  lw=3, density=True, \
                      linestyle='-', histtype='step', color='k', bins=32, label='Selected')
        if plot_kde:
            _= seaborn.kdeplot(data.age1.values, lw=3, linestyle='--', color='r', label='Full sample',\
             ax=ax, common_grid=True, multiple="stack", alpha=0.5)
            _= seaborn.kdeplot(data.age1[selection].values, lw=4, linestyle='-', color='k', \
                label='Selected', ax=ax, common_grid=True,multiple="stack", alpha=0.5)


        ax.axvspan(STD_AGE[0], STD_AGE[-1], alpha=0.2, color='blue')
        ax.set(xlabel='Age (Gyr)', ylabel='Normalized Density')
        plt.legend()
        ax.minorticks_on()
        plt.savefig(file_plot)

        #print(data.columns)
        
        fig, ax=plt.subplots(ncols=2, figsize=(12, 4))
        ax[0].scatter((data.v_x**2+data.v_y**2)**0.5, data.v_z, s=0.1,  c=data.age1, \
                      marker='+',  cmap='plasma', vmin=0, vmax=13)

        ax[0].errorbar(np.nanmedian(vr), np.nanmedian(vz), xerr=np.nanstd(vr), \
            yerr=np.nanstd(vz), fmt='o', ms=15, c='k')

        ax[0].set(xlabel=r'$(V_x^2+v_y^2)^{0.5}$ (km/s) ', ylabel=r'$V_z$ (km/s)')

        #plot Jz instead
        if use_jz:    
            ax[1].scatter(data['[Fe/H]'],  data.Jz, s=0.5,  c=data.age1, \
                          marker='+',  cmap='plasma', vmin=0, vmax=13)
            ax[1].errorbar(source_metal[0], mean_source_jz, xerr=source_metal[-1],\
                       yerr=std_source_jz, marker='o', ms=15, c='k')
            ax[1].set(  xlabel='[Fe/H]', \
               ylabel=r'J$_z$ (kpc$^2$/Myr) ', ylim=[-0.1, 1.1])
        

        #plot vertical velocity instead   
        if not use_jz:
            ax[1].scatter(data['[Fe/H]'],  data['v_z'], s=0.5,  c=data.age1, \
                          marker='+',  cmap='viridis_r', vmin=0, vmax=13)
            ax[1].errorbar(source_metal[0], np.nanmedian(vz), xerr=source_metal[-1],\
                       yerr=np.nanstd(vz), marker='o', ms=15, c='k')
            ax[1].set(  xlabel='[Fe/H]', \
               ylabel=r'V$_z$ (km/s) ')
        
        
        norm= mpl.colors.Normalize(vmin=0,vmax=13)
        mp=mpl.cm.ScalarMappable(norm=norm, cmap='viridis_r')
        cax = fig.add_axes([1.01, 0.25, .015, 0.7])
        cbar=plt.colorbar(mp, cax=cax, orientation='vertical')
        cbar.ax.set_ylabel(r'Age (Gyr)', fontsize=18)
        plt.tight_layout()
        for a in ax: a.minorticks_on()
        plt.savefig(file_plot.replace('.', '_scatter_'), rasterized=True, bbox_inches='tight')


    return { 'median_age':MEDIAN_AGE,
			'std_age': (MEDIAN_AGE-STD_AGE[0], STD_AGE[1]-MEDIAN_AGE),
			'posterior': data.age1[selection].values }







